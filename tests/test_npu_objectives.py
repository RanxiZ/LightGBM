"""Standalone real-CPU/NPU objective, evaluation and lifecycle regression.

Examples::

    python tests/test_npu_objectives.py --summary-output objectives-summary.json
    python tests/test_npu_objectives.py --scenario cross_entropy_lambda
    python tests/test_npu_objectives.py --mape-reference-only

The hand-calculated MAPE Hessian/gain reference runs first on CPU and can run
alone with --mape-reference-only. Only the parity/lifecycle checks require a real
NPU and the intended LightGBM installation. No build, library replacement,
device auto-detection, CPU retry, skips, or tolerance fallback is performed.
This module is import-safe. The categorical companion supplies only
its logger and strict tree/float comparison helpers; it does not run its main.

Coverage is deliberately small: dense numeric data, 256 training / 64 validation
rows, eight features, two rounds (six trees for three-class objectives), and five
leaves. regression_l2 additionally exercises reset_parameter/update, on-disk
save/load and two continued rounds. This is parity testing, not a mathematical
reference for every built-in loss, a performance benchmark, or coverage of all
objective parameters, sparse/categorical inputs, distributed training, GPU/CUDA,
early stopping, refit, or custom multiclass gradients. Python custom gradients
and built-in objective/metric calculations may run on the host; runtime logs
prove that tree construction and ordinary training score updates use the NPU.
"""

import argparse
import json
from pathlib import Path
import re
import tempfile
import time
import traceback

import lightgbm as lgb
import numpy as np

from test_npu_categorical import CaptureLogger, assert_float, compare_trees


NUM_ROUNDS = 2
NUM_LEAVES = 5
ATOL = RTOL = 2e-5


def scenarios():
    """Explicit metric names make a missing evaluation path a hard failure."""
    for name, metrics, extra in (
        ("regression_l2", ["l2", "l1", "rmse"], {}),
        ("regression_l1", ["l1", "l2"], {}),
        ("huber", ["huber", "l2"], {"alpha": 0.8}),
        ("fair", ["fair", "l1"], {"fair_c": 0.7}),
        ("quantile", ["quantile", "l1"], {"alpha": 0.7}),
        ("mape", ["mape", "l1"], {}),
        ("poisson", ["poisson", "l2"], {"poisson_max_delta_step": 0.7}),
        ("gamma", ["gamma", "gamma_deviance"], {}),
        ("tweedie", ["tweedie", "l2"], {"tweedie_variance_power": 1.4}),
        ("binary", ["binary_logloss", "binary_error", "auc"], {"sigmoid": 1.3}),
        ("cross_entropy", ["cross_entropy", "kullback_leibler"], {}),
        ("cross_entropy_lambda", ["cross_entropy_lambda"], {}),
        ("multiclass", ["multi_logloss", "multi_error"], {"num_class": 3}),
        ("multiclassova", ["multi_logloss", "multi_error"], {"num_class": 3, "sigmoid": 1.3}),
        ("lambdarank", ["ndcg", "map"], {"eval_at": [1, 3], "lambdarank_truncation_level": 5}),
        ("rank_xendcg", ["ndcg", "map"], {"eval_at": [1, 3]}),
        ("custom", ["l2", "l1"], {}),
    ):
        yield dict(name=name, metrics=metrics, extra=extra)


def make_data(name):
    rng = np.random.default_rng(17031)
    x = rng.normal(size=(320, 8)).astype(np.float32)
    signal = 1.6 * x[:, 0] - 0.9 * x[:, 1] + 0.6 * x[:, 2] * x[:, 3]
    signal += 0.12 * rng.normal(size=len(x))
    weights = rng.uniform(0.6, 1.7, len(x)).astype(np.float32)
    group = None
    if name in ("lambdarank", "rank_xendcg"):
        # Contiguous eight-document queries, each with all four relevance levels.
        # Query-constant row weights exercise weighted ranking and metrics.
        y = np.empty(len(x), dtype=np.int32)
        for start in range(0, len(x), 8):
            order = np.argsort(signal[start:start + 8], kind="stable")
            y[start + order] = np.repeat(np.arange(4), 2)
            weights[start:start + 8] = rng.uniform(0.6, 1.7)
        group = np.full(len(x) // 8, 8, dtype=np.int32)
    elif name in ("multiclass", "multiclassova"):
        y = np.digitize(signal, [-0.65, 0.7]).astype(np.int32)
    elif name == "binary":
        y = (signal > 0.1).astype(np.int32)
    elif name in ("cross_entropy", "cross_entropy_lambda"):
        y = (0.05 + 0.9 / (1.0 + np.exp(-signal))).astype(np.float32)
    elif name == "poisson":
        y = rng.poisson(np.exp(0.3 + 0.35 * signal)).astype(np.float32)
    elif name in ("mape", "gamma", "tweedie"):
        y = np.exp(0.5 + 0.35 * signal).astype(np.float32)
        if name == "tweedie":
            y[::11] = 0.0  # Compound Poisson domain includes exact zeros.
    else:
        y = signal.astype(np.float32)
    train_group = None if group is None else group[:32]
    valid_group = None if group is None else group[32:]
    return (dict(x=x[:256], y=y[:256], weight=weights[:256], group=train_group),
            dict(x=x[256:], y=y[256:], weight=weights[256:], group=valid_group))


def custom_objective(prediction, dataset):
    """Weighted squared loss; custom objectives must apply weights themselves."""
    weight = dataset.get_weight().astype(np.float64)
    gradient = (prediction - dataset.get_label()) * weight
    return gradient, weight.copy()


def output_mean(prediction, dataset):
    # Built-in objectives pass CONVERTED scores to feval, custom ones raw scores.
    # Use a nonconstant reduction for multiclass (sum would always be one).
    values = prediction if prediction.ndim == 1 else prediction[:, 0]
    return "weighted_output_mean", float(np.average(values, weights=dataset.get_weight())), False


def parameters(spec, device):
    params = dict(objective=custom_objective if spec["name"] == "custom" else spec["name"],
                  metric=spec["metrics"], device_type=device, verbosity=1,
                  num_threads=2, num_leaves=NUM_LEAVES, max_bin=15,
                  learning_rate=0.15, min_data_in_leaf=8, min_data_in_bin=1,
                  force_col_wise=True, deterministic=True, feature_pre_filter=False,
                  use_quantized_grad=False, seed=17, data_random_seed=17,
                  bagging_seed=17, feature_fraction_seed=17, objective_seed=17,
                  feature_fraction=1.0, bagging_fraction=1.0,
                  lambda_l1=0.03, lambda_l2=0.4, bin_construct_sample_cnt=10000)
    params.update(spec["extra"])
    return params


def datasets(parts, params):
    train, valid = parts
    data = lgb.Dataset(train["x"], label=train["y"], weight=train["weight"],
                       group=train["group"], params=params, free_raw_data=False)
    validation = lgb.Dataset(valid["x"], label=valid["y"], weight=valid["weight"],
                             group=valid["group"], reference=data, free_raw_data=False)
    return data, validation


def converted_output(raw, spec):
    """Independent interpretation of raw margins, including non-probability outputs."""
    name = spec["name"]
    if name in ("poisson", "gamma", "tweedie"):
        return np.exp(raw)
    if name == "cross_entropy_lambda":
        # A normalized exponential rate, NOT a probability. Per-row probability
        # would be -expm1(-weight * rate), and is not Booster.predict's output.
        return np.logaddexp(0.0, raw)
    if name == "multiclass":
        exp = np.exp(raw - np.max(raw, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    if name in ("binary", "cross_entropy", "multiclassova"):
        # One-vs-all uses independent sigmoids, NOT softmax or row normalization.
        return 1.0 / (1.0 + np.exp(-spec["extra"].get("sigmoid", 1.0) * raw))
    return raw.copy()


def expected_metrics(spec):
    names = set()
    for metric in spec["metrics"]:
        if metric in ("ndcg", "map"):
            names.update(f"{metric}@{cutoff}" for cutoff in spec["extra"]["eval_at"])
        else:
            names.add(metric)
    return names | {"weighted_output_mean"}


def metrics_dict(evaluations, spec, context):
    result = {}
    for dataset, metric, value, higher_better in evaluations:
        key = f"{dataset}.{metric}"
        assert key not in result and np.isfinite(value), (context, key, value)
        result[key] = dict(value=float(value), higher_better=bool(higher_better))
    expected = {f"{dataset}.{metric}" for dataset in ("train", "valid")
                for metric in expected_metrics(spec)}
    assert set(result) == expected, (context, set(result), expected)
    return result


def snapshot(model, parts, spec, context, evaluations):
    trees = model.dump_model()["tree_info"]
    nclasses = spec["extra"].get("num_class", 1)
    assert model.num_model_per_iteration() == nclasses, context
    assert len(trees) == model.current_iteration() * nclasses, context
    assert all(1 < tree["num_leaves"] <= NUM_LEAVES for tree in trees), context
    result = dict(trees=trees, metrics=metrics_dict(evaluations, spec, context))
    for index, (label, part) in enumerate(zip(("train", "valid"), parts)):
        raw = model.predict(part["x"], raw_score=True, num_threads=2)
        prediction = model.predict(part["x"], num_threads=2)
        leaves = model.predict(part["x"], pred_leaf=True, num_threads=2)
        cached = model._Booster__inner_predict(data_idx=index).copy()
        shape = (len(part["x"]), nclasses) if nclasses > 1 else (len(part["x"]),)
        assert raw.shape == prediction.shape == cached.shape == shape, (context, label, shape)
        # GetPredictAt/__inner_predict converts objective output and reshapes
        # class-major buffers to (rows, classes). It is NOT a raw-margin API.
        assert_float(cached, prediction, f"{context}.{label}.cached_converted", output=True)
        assert_float(prediction, converted_output(raw, spec),
                     f"{context}.{label}.transformation", output=True)
        # Reconstruct raw margins from dumped leaf values, in iteration-major,
        # class-minor tree order. This independently checks raw prediction layout.
        reconstructed = np.zeros((len(part["x"]), nclasses))
        for tree_index, tree in enumerate(trees):
            leaf_values = {}

            def visit(node):
                if "split_index" in node:
                    visit(node["left_child"])
                    visit(node["right_child"])
                else:
                    leaf_values[node.get("leaf_index", 0)] = node["leaf_value"]

            visit(tree["tree_structure"])
            reconstructed[:, tree_index % nclasses] += np.array(
                [leaf_values[int(leaf)] for leaf in leaves[:, tree_index]])
        if nclasses == 1:
            reconstructed = reconstructed[:, 0]
        assert_float(raw, reconstructed, f"{context}.{label}.raw_leaf_sum", output=True)
        mean = output_mean(prediction, _WeightedPart(part))[1]
        assert_float(result["metrics"][f"{label}.weighted_output_mean"]["value"], mean,
                     f"{context}.{label}.feval_output")
        result[label] = dict(raw=raw, prediction=prediction, cached=cached, leaves=leaves)
    return result


class _WeightedPart:
    """Minimal weight interface for independently evaluating the custom metric."""
    def __init__(self, part):
        self.part = part

    def get_weight(self):
        return self.part["weight"]


def check_npu_logs(messages, new_trees, context, name, initialised=True):
    text = "\n".join(messages)
    if initialised:
        assert "NPU runtime initialised on device" in text, (context, text)
        assert "NPU tree learner cached device bins" in text, (context, text)
    searches = re.findall(r"NPU split search: split_calls=(\d+) \(\+(\d+) this tree\), leaves=(\d+)", text)
    assert len(searches) == len(new_trees), (context, searches, len(new_trees))
    assert all(int(delta) > 0 and int(leaves) > 1 for _, delta, leaves in searches), (context, searches)
    assert "runtime winners populate best_split_per_leaf_ used by tree splits" in text, context
    updates = re.findall(r"NPU score update: device_calls=(\d+), partition_rows_downloaded=(\d+) this tree", text)
    assert len(updates) == len(new_trees), (context, updates, len(new_trees))
    calls = [int(call) for call, _ in updates]
    assert all(right == left + 1 for left, right in zip(calls, calls[1:])), (context, calls)
    renewals = text.count("materialized host rows")
    if name in ("regression_l1", "quantile", "mape"):
        # Leaf renewal explicitly downloads each training row once. These
        # compatibility transfers precede the still-device-side score update
        # and are included in the per-tree counter, unlike ordinary objectives.
        assert renewals == len(new_trees), (context, renewals, len(new_trees))
        assert all(int(downloaded) == 256 for _, downloaded in updates), (context, updates)
    else:
        assert renewals == 0, (context, text)
        assert all(int(downloaded) == 0 for _, downloaded in updates), (context, updates)
    assert "fallback" not in text.lower() and "falling back" not in text.lower(), (context, text)
    evidence = dict(split_searches=len(searches), score_updates=len(updates),
                    partition_rows_downloaded=sum(int(downloaded) for _, downloaded in updates),
                    compatibility_leaf_renewals=renewals,
                    runtime_initialised="NPU runtime initialised on device" in text)
    print(context, "runtime_evidence=", json.dumps(evidence, sort_keys=True), flush=True)
    return evidence


def compare_snapshots(cpu, npu, context):
    compare_trees(cpu["trees"], npu["trees"], context)
    max_error = 0.0
    for label in ("train", "valid"):
        for field in ("raw", "prediction", "cached"):
            assert_float(npu[label][field], cpu[label][field], f"{context}.{label}.{field}", output=True)
            max_error = max(max_error, float(np.max(np.abs(npu[label][field] - cpu[label][field]))))
        np.testing.assert_array_equal(npu[label]["leaves"], cpu[label]["leaves"],
                                      err_msg=f"{context}.{label}.leaves")
    assert cpu["metrics"].keys() == npu["metrics"].keys(), context
    for key in cpu["metrics"]:
        assert cpu["metrics"][key]["higher_better"] == npu["metrics"][key]["higher_better"], (context, key)
        assert_float(npu["metrics"][key]["value"], cpu["metrics"][key]["value"], f"{context}.{key}")
    return max_error


def train_stage(spec, device, parts, params, logger, report, stage, init_model=None):
    data, valid = datasets(parts, params)
    snapshots = []
    history = {}
    logger.messages.clear()
    start = time.perf_counter()

    def capture(env):
        assert env.model.params["device_type"] == device, (stage, env.model.params)
        snapshots.append(snapshot(env.model, parts, spec,
                                  f"{spec['name']}.{device}.{stage}.round{env.iteration + 1}",
                                  env.evaluation_result_list))

    model = lgb.train(params, data, num_boost_round=NUM_ROUNDS,
                      valid_sets=[data, valid], valid_names=["train", "valid"],
                      init_model=init_model, feval=output_mean, keep_training_booster=True,
                      callbacks=[lgb.record_evaluation(history), capture])
    assert len(snapshots) == NUM_ROUNDS, (stage, len(snapshots))
    expected_iterations = NUM_ROUNDS if init_model is None else 2 * NUM_ROUNDS + 1
    assert model.current_iteration() == expected_iterations, (stage, model.current_iteration())
    expected = expected_metrics(spec)
    assert set(history) == {"train", "valid"}, (stage, history)
    for dataset in history:
        assert set(history[dataset]) == expected, (stage, history)
        for metric, values in history[dataset].items():
            assert len(values) == NUM_ROUNDS, (stage, metric, values)
            assert_float(values, [item["metrics"][f"{dataset}.{metric}"]["value"] for item in snapshots],
                         f"{stage}.{dataset}.{metric}.history")
    stage_report = dict(seconds=time.perf_counter() - start, requested_device=device,
                        effective_device=model.params["device_type"], metric_history=history,
                        tree_leaves=[tree["num_leaves"] for tree in snapshots[-1]["trees"]],
                        logs=list(logger.messages))
    report[stage] = stage_report
    if device == "npu":
        new_trees = snapshots[-1]["trees"][-NUM_ROUNDS * spec["extra"].get("num_class", 1):]
        stage_report["runtime_evidence"] = check_npu_logs(
            logger.messages, new_trees, f"{spec['name']}.{device}.{stage}", spec["name"])
    return model, snapshots


def check_roundtrip(model, path, parts, spec, before, context):
    model.save_model(str(path))
    restored = lgb.Booster(model_file=str(path))
    compare_trees(before["trees"], restored.dump_model()["tree_info"], context)
    assert restored.current_iteration() == model.current_iteration(), context
    for label, part in zip(("train", "valid"), parts):
        for field, kwargs in (("raw", dict(raw_score=True)), ("prediction", {})):
            assert_float(restored.predict(part["x"], num_threads=2, **kwargs), before[label][field],
                         f"{context}.{label}.{field}", output=True)
        np.testing.assert_array_equal(restored.predict(part["x"], pred_leaf=True, num_threads=2),
                                      before[label]["leaves"], err_msg=f"{context}.{label}.leaves")


def run_device(spec, device, logger, report, directory):
    parts = make_data(spec["name"])
    params = parameters(spec, device)
    model, results = train_stage(spec, device, parts, params, logger, report, "initial")
    if spec["name"] != "regression_l2":
        return results
    # Reset a live training booster, not a prediction-only deserialized booster.
    reset = dict(learning_rate=0.07, lambda_l2=0.9, num_leaves=4)
    logger.messages.clear()
    model.reset_parameter(reset)
    assert model.params["device_type"] == device, model.params
    assert all(model.params[key] == value for key, value in reset.items()), model.params
    old_count = len(results[-1]["trees"])
    assert model.update() is False, "representative reset update unexpectedly finished"
    reset_snapshot = snapshot(model, parts, spec, f"{device}.reset",
                              model.eval_train(feval=output_mean) + model.eval_valid(feval=output_mean))
    assert len(reset_snapshot["trees"]) == old_count + 1, device
    assert reset_snapshot["trees"][-1]["num_leaves"] <= reset["num_leaves"], device
    assert_float(reset_snapshot["trees"][-1]["shrinkage"], reset["learning_rate"], f"{device}.reset.learning_rate")
    report["reset"] = dict(parameters=reset, metrics=reset_snapshot["metrics"], logs=list(logger.messages))
    if device == "npu":
        report["reset"]["runtime_evidence"] = check_npu_logs(
            logger.messages, reset_snapshot["trees"][old_count:], f"{device}.reset",
            spec["name"], initialised=False)
    results.append(reset_snapshot)
    path = directory / f"{device}.txt"
    check_roundtrip(model, path, parts, spec, reset_snapshot, f"{device}.save_load")
    report["save_load"] = dict(status="passed", format="on-disk LightGBM text model")
    continued_params = dict(params, **reset)
    continued, resumed = train_stage(spec, device, parts, continued_params, logger, report,
                                     "continued", init_model=str(path))
    assert continued.current_iteration() == NUM_ROUNDS + 1 + NUM_ROUNDS, device
    compare_trees(reset_snapshot["trees"], resumed[-1]["trees"][:old_count + 1], f"{device}.continued.prefix")
    results.extend(resumed)
    # Also compare continued serialized outputs, not only the pre-resume model.
    check_roundtrip(continued, path, parts, spec, resumed[-1], f"{device}.continued.save_load")
    return results


def run_mape_reference(logger, report):
    """Check weighted Hessians and gain against arithmetic, not device parity."""
    x = np.array([0] * 12 + [1] * 4, dtype=np.float32).reshape(-1, 1)
    y = np.array([2] * 12 + [8] * 4, dtype=np.float32)
    initial = np.full(len(y), 4.0)
    weight_cases = (
        ("unweighted", None),
        ("ones", np.ones(len(y), dtype=np.float32)),
        ("twos", np.full(len(y), 2.0, dtype=np.float32)),
        ("nonuniform", np.array([1] * 12 + [4] * 4, dtype=np.float32)),
    )
    for layout in ("col", "row"):
        for name, weight in weight_cases:
            context = f"mape_reference.{layout}_wise.{name}"
            case = dict(name=name, layout=f"{layout}_wise", status="running")
            report["cases"].append(case)
            # LightGBM stores gradients in float32 and accumulates in double.
            # Keep the oracle independent of dumped weights and gain, allowing
            # the existing float tolerance for the float32 dump of split_gain.
            w = np.ones(len(y), dtype=np.float32) if weight is None else weight
            gradient = (np.sign(initial - y) / np.maximum(1.0, np.abs(y)) * w).astype(np.float32)
            expected_g = np.array([gradient[:12].sum(dtype=np.float64),
                                   gradient[12:].sum(dtype=np.float64)])
            expected_h = np.array([w[:12].sum(dtype=np.float64),
                                   w[12:].sum(dtype=np.float64)])
            expected_gain = float(np.sum(expected_g ** 2 / expected_h)
                                  - expected_g.sum() ** 2 / expected_h.sum())
            if name == "nonuniform":
                assert_float(expected_g, [6.0, -2.0], context + ".gradient_oracle")
                assert_float(expected_h, [12.0, 16.0], context + ".hessian_oracle")
                assert_float(expected_gain, 36.0 / 12.0 + 4.0 / 16.0 - 16.0 / 28.0,
                             context + ".gain_oracle")
            params = dict(objective="mape", metric="None", device_type="cpu",
                          verbosity=1, num_threads=2, num_leaves=2, learning_rate=1.0,
                          lambda_l1=0.0, lambda_l2=0.0, min_data_in_leaf=1,
                          min_data_in_bin=1, feature_pre_filter=False,
                          use_quantized_grad=False, boost_from_average=False,
                          force_col_wise=layout == "col", force_row_wise=layout == "row")
            logger.messages.clear()
            data = lgb.Dataset(x, label=y, weight=weight, init_score=initial, params=params)
            model = lgb.train(params, data, num_boost_round=1)
            trees = model.dump_model()["tree_info"]
            assert model.params["device_type"] == "cpu", (context, model.params)
            assert len(trees) == 1 and trees[0]["num_leaves"] == 2, (context, trees)
            root = trees[0]["tree_structure"]
            assert root["split_feature"] == 0 and root["decision_type"] == "<=", (context, root)
            assert 0 <= root["threshold"] < 1, (context, root)
            leaves = [root["left_child"], root["right_child"]]
            assert all("split_index" not in leaf for leaf in leaves), (context, leaves)
            counts = [leaf["leaf_count"] for leaf in leaves]
            actual_h = [leaf["leaf_weight"] for leaf in leaves]
            case.update(expected_gradient=expected_g.tolist(), expected_hessian=expected_h.tolist(),
                        expected_gain=expected_gain, split_gain=root["split_gain"],
                        leaf_weight=actual_h, leaf_count=counts)
            # Check gain independently of leaf_weight/count agreement. MAPE's
            # renewed leaf values are residual medians, not -G/H.
            failures = []
            for field, actual, expected in (("split_gain", root["split_gain"], expected_gain),
                                            ("leaf_weight", actual_h, expected_h)):
                try:
                    assert_float(actual, expected, context + "." + field)
                except AssertionError:
                    failures.append(traceback.format_exc())
            try:
                assert counts == [12, 4], (context, counts)
            except AssertionError:
                failures.append(traceback.format_exc())
            case["status"] = "failed" if failures else "passed"
            if failures:
                case["errors"] = failures
            print(context, case["status"].upper(), json.dumps(case, sort_keys=True), flush=True)
    failed = [f"{case['layout']}.{case['name']}" for case in report["cases"]
              if case["status"] != "passed"]
    assert not failed, ("MAPE CPU reference failures", failed)
    report["status"] = "passed"


def main():
    specs = list(scenarios())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[spec["name"] for spec in specs],
                        help="Run just one objective; regression_l2 includes lifecycle checks.")
    parser.add_argument("--mape-reference-only", action="store_true",
                        help="Run only the hand-calculated CPU MAPE Hessian/gain reference; no NPU required.")
    parser.add_argument("--summary-output", type=Path, default=Path("npu_objectives_summary.json"))
    args = parser.parse_args()
    if args.scenario:
        specs = [spec for spec in specs if spec["name"] == args.scenario]
    if args.mape_reference_only:
        specs = []
    logger = CaptureLogger()
    lgb.register_logger(logger)
    library = str(lgb.basic._LIB._name)
    print("lightgbm_library=", library, flush=True)
    summary = dict(status="running", library=library, lightgbm_version=lgb.__version__,
                   train_rows=256, validation_rows=64, features=8, num_boost_round=NUM_ROUNDS,
                   max_bin=15, num_leaves=NUM_LEAVES, num_threads=2, use_quantized_grad=False,
                   tolerances=dict(gain_metric_weight_atol=ATOL, gain_metric_weight_rtol=RTOL,
                                   output_prediction_atol=ATOL, output_prediction_rtol=0),
                   coverage_limits=__doc__, mape_reference_only=args.mape_reference_only,
                   mape_reference=dict(status="running", cases=[]), scenarios=[])

    def save():
        args.summary_output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    save()
    try:
        run_mape_reference(logger, summary["mape_reference"])
    except Exception:
        summary["mape_reference"].update(status="failed", error=traceback.format_exc(),
                                         last_logs=list(logger.messages))
        for case in summary["mape_reference"]["cases"]:
            if case["status"] == "running":
                case["status"] = "failed"
        summary["status"] = "failed"
        save()
        raise
    save()
    for spec in specs:
        report = dict(name=spec["name"], metrics=spec["metrics"], parameters=spec["extra"],
                      status="running", cpu={}, npu={})
        summary["scenarios"].append(report)
        try:
            with tempfile.TemporaryDirectory(prefix="lightgbm-npu-objectives-") as temp:
                directory = Path(temp)
                cpu = run_device(spec, "cpu", logger, report["cpu"], directory)
                npu = run_device(spec, "npu", logger, report["npu"], directory)
                assert len(cpu) == len(npu), spec["name"]
                error = max(compare_snapshots(left, right, f"{spec['name']}.snapshot{index + 1}")
                            for index, (left, right) in enumerate(zip(cpu, npu)))
            report.update(status="passed", prediction_max_abs_error=error)
            print(spec["name"], "PASS prediction_max_abs_error=", error, flush=True)
        except Exception:
            report.update(status="failed", error=traceback.format_exc(), last_logs=list(logger.messages))
            summary["status"] = "failed"
            save()
            raise
        save()
    summary["status"] = "passed"
    save()
    if args.mape_reference_only:
        print("PASS: CPU MAPE hand-calculated Hessian/gain reference (8 cases)", flush=True)
    else:
        print("PASS: CPU MAPE reference and real CPU/NPU objective structures, raw/converted/cached outputs, train/valid metrics and lifecycle", flush=True)
    print("summary_output=", args.summary_output.resolve(), flush=True)


if __name__ == "__main__":
    main()