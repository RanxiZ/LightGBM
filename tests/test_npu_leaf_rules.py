"""Standalone CPU/NPU path_smooth and max_delta_step parity regression.

Run against the intended, already-built installation with::

    python tests/test_npu_leaf_rules.py --summary-output leaf-rules-summary.json

This script never builds, chooses, or replaces a library. Missing NPU support
is a failure, not a skip. Tree structure / thresholds / leaf assignments are
exact; floating gains and predictions use the existing categorical helper's
2e-5 tolerances. Positive and negative leaf clipping, and negative (disabled)
max_delta_step, are all covered. Refit follows the existing CPU semantics,
including its host leaf-renewal path; it is not an independent refit oracle.
"""

import argparse
import json
from pathlib import Path
import re
import tempfile
import traceback

import lightgbm as lgb
import numpy as np

from test_npu_categorical import (
    CaptureLogger, assert_float, compare_trees, make_data, split_nodes,
)


NUM_TREES = 3
LEARNING_RATE = 0.2


def base_params():
    return dict(
        objective="regression", verbosity=1, num_threads=2, num_leaves=5,
        learning_rate=LEARNING_RATE, boost_from_average=False,
        min_data_in_leaf=8, min_sum_hessian_in_leaf=1e-3, min_data_in_bin=1,
        max_bin=15, feature_pre_filter=False, force_col_wise=True,
        use_quantized_grad=False, seed=173, data_random_seed=173,
        bin_construct_sample_cnt=10000, lambda_l1=0.0, lambda_l2=0.0,
        max_cat_to_onehot=2, max_cat_threshold=4, min_data_per_group=8,
        cat_smooth=0.0, cat_l2=7.5, path_smooth=0.0, max_delta_step=0.0,
    )


def dataset(kind):
    if kind == "numeric":
        rng = np.random.default_rng(1701)
        x = rng.normal(size=(320, 4)).astype(np.float32)
        y = (2.1 + 5.3 * x[:, 0] + 2.7 * (x[:, 1] > 0.2)
             - 1.8 * x[:, 2] ** 2 + 0.04 * x[:, 3])
        x[rng.random(x.shape) < 0.04] = np.nan
        return x, y, []
    cardinality = 3 if kind == "onehot" else 9
    x, y = make_data(cardinality, 1702 + cardinality,
                     counts=[40] * cardinality, missing=True)
    return x, y, [0, 1, 4, 5]


def leaf_values(node):
    if "split_index" in node:
        yield from leaf_values(node["left_child"])
        yield from leaf_values(node["right_child"])
    else:
        yield node["leaf_value"]


def snapshot(model, x):
    return dict(trees=model.dump_model()["tree_info"],
                predictions=model.predict(x, raw_score=True),
                leaves=model.predict(x, pred_leaf=True))


def equivalent(cpu, npu, context):
    compare_trees(cpu["trees"], npu["trees"], context)
    np.testing.assert_array_equal(cpu["leaves"], npu["leaves"], err_msg=context)
    assert_float(npu["predictions"], cpu["predictions"], context + ".predictions", output=True)


def runtime_evidence(messages, expected_trees, name):
    text = "\n".join(messages)
    assert "NPU runtime initialised on device" in text, (name, text)
    assert "NPU tree learner cached device bins" in text, (name, text)
    searches = re.findall(
        r"NPU split search: split_calls=(\d+) \(\+(\d+) this tree\), leaves=(\d+)", text)
    assert len(searches) == expected_trees, (name, searches, text)
    assert all(int(delta) > 0 and int(leaves) > 1 for _, delta, leaves in searches), (name, searches)
    assert "runtime winners populate best_split_per_leaf_ used by tree splits" in text, (name, text)
    updates = re.findall(
        r"NPU score update: device_calls=(\d+), partition_rows_downloaded=(\d+) this tree", text)
    assert len(updates) >= expected_trees, (name, updates, text)
    assert all(int(calls) > 0 and int(rows) == 0 for calls, rows in updates), (name, updates)
    assert "fallback" not in text.lower() and "falling back" not in text.lower(), (name, text)
    # Retain actual backend messages, not a synthetic assertion-success string.
    return [message for message in messages if "NPU" in message]


def train(device, x, y, categories, params, logger, weight=None, rounds=NUM_TREES):
    logger.messages.clear()
    data = lgb.Dataset(x, label=y, weight=weight, categorical_feature=categories,
                       params=params, free_raw_data=False)
    model = lgb.train(dict(params, device_type=device), data, num_boost_round=rounds,
                      keep_training_booster=True)
    result = snapshot(model, x)
    evidence = runtime_evidence(logger.messages, rounds, device) if device == "npu" else []
    assert len(result["trees"]) == rounds, (device, result["trees"])
    return model, data, result, evidence


def scenarios():
    # Each branch gets isolated smoothing, isolated clipping, and both together.
    for kind in ("numeric", "onehot", "grouped"):
        for rule, extra in (
            ("smooth", dict(path_smooth=60.0)),
            ("clip", dict(max_delta_step=2.5)),
            ("combined", dict(path_smooth=35.0, max_delta_step=2.5,
                              lambda_l1=0.7, lambda_l2=1.9)),
            ("bagged_weighted", dict(path_smooth=25.0, max_delta_step=3.2,
                                     lambda_l1=0.4, lambda_l2=2.3,
                                     bagging_fraction=0.77, bagging_freq=1,
                                     bagging_seed=79, histogram_pool_size=0.000001)),
        ):
            yield dict(name=kind + "_" + rule, kind=kind, rule=rule, extra=extra)
    yield dict(name="negative_limit_disabled", kind="numeric", rule="disabled",
               extra=dict(max_delta_step=-2.5))


def lifecycle(cpu_model, npu_model, cpu_data, npu_data, x, y, categories,
              weight, logger, report):
    # Save/load both models, preserving structure and prediction parity.
    with tempfile.TemporaryDirectory(prefix="npu-leaf-rules-") as directory:
        restored = []
        for device, model in (("cpu", cpu_model), ("npu", npu_model)):
            path = Path(directory) / (device + ".txt")
            model.save_model(path)
            loaded = lgb.Booster(model_file=path)
            equivalent(snapshot(model, x), snapshot(loaded, x), "serialization." + device)
            restored.append(snapshot(loaded, x))
        equivalent(*restored, "serialization.parity")
    report["serialization"] = True

    # Refit is deliberately compared with CPU, not with newly trained trees.
    # The implementation inherits SerialTreeLearner's existing leaf rules.
    refit_y = (0.83 * y + 0.31).astype(np.float32)
    refitted = []
    for device, model, data in (("cpu", cpu_model, cpu_data), ("npu", npu_model, npu_data)):
        logger.messages.clear()
        refit = model.refit(x, refit_y, decay_rate=0.35, reference=data,
                            categorical_feature=categories, weight=weight,
                            free_raw_data=False)
        refitted.append(snapshot(refit, x))
        assert not np.allclose(refitted[-1]["predictions"], snapshot(model, x)["predictions"],
                               rtol=0, atol=2e-5), (device, "refit had no effect")
        if device == "npu":
            text = "\n".join(logger.messages)
            assert "NPU runtime initialised on device" in text, text
            assert "fallback" not in text.lower() and "falling back" not in text.lower(), text
            report["refit_logs"] = list(logger.messages)
    equivalent(*refitted, "refit.parity")
    report["refit"] = True


def run_spec(spec, logger, report):
    x, y, categories = dataset(spec["kind"])
    # Labels are stored as float32 by Dataset; use the same values everywhere.
    y = np.asarray(y, dtype=np.float32)
    params = base_params()
    if spec["kind"] == "onehot":
        params["max_cat_to_onehot"] = 8
    params.update(spec["extra"])
    weight = None
    if spec["rule"] == "bagged_weighted":
        weight = np.random.default_rng(1777).uniform(0.55, 1.8, len(y)).astype(np.float32)
    cpu_model, cpu_data, cpu, _ = train("cpu", x, y, categories, params, logger, weight)
    npu_model, npu_data, npu, evidence = train("npu", x, y, categories, params, logger, weight)
    equivalent(cpu, npu, spec["name"])
    report["npu_logs"] = evidence
    nodes = [node for tree in cpu["trees"] for node in split_nodes(tree["tree_structure"])]
    assert any(node["split_index"] > 0 for node in nodes), (spec["name"], "no non-root search")
    expected_type = "<=" if spec["kind"] == "numeric" else "=="
    assert any(node["decision_type"] == expected_type for node in nodes), (spec["name"], nodes)
    if spec["kind"] == "grouped":
        assert any(node["decision_type"] == "==" and "||" in node["threshold"] for node in nodes), spec

    # Compare with the same regularization, weights, bins and bagging, changing
    # only these two leaf rules. Acceptance without changing outputs must fail.
    baseline_params = dict(params, path_smooth=0.0, max_delta_step=0.0)
    _, _, baseline, _ = train("cpu", x, y, categories, baseline_params, logger, weight)
    if spec["rule"] == "disabled":
        equivalent(baseline, cpu, "negative_limit.cpu")
        equivalent(baseline, npu, "negative_limit.npu")
    else:
        assert not np.allclose(cpu["predictions"], baseline["predictions"], rtol=0, atol=2e-5), spec
        report["changes_vs_disabled"] = True
    if spec["rule"] == "clip":
        values = np.asarray([value for tree in npu["trees"]
                             for value in leaf_values(tree["tree_structure"])])
        limit = LEARNING_RATE * params["max_delta_step"]
        assert np.max(np.abs(values)) <= limit + 2e-5, (spec, values)
        # Exercise both saturation directions, not just a positive parameter.
        assert np.any(np.isclose(values, limit, rtol=0, atol=2e-5)), (spec, values)
        assert np.any(np.isclose(values, -limit, rtol=0, atol=2e-5)), (spec, values)
        report["clipped_both_signs"] = True
    if spec["rule"] == "combined":
        lifecycle(cpu_model, npu_model, cpu_data, npu_data, x, y, categories,
                  weight, logger, report)
    report["parameters"] = params
    report["tree_count"] = len(npu["trees"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-output", type=Path, default=Path("npu_leaf_rules_summary.json"))
    args = parser.parse_args()
    logger = CaptureLogger()
    lgb.register_logger(logger)
    summary = dict(status="running", library=str(lgb.basic._LIB._name),
                   num_boost_round=NUM_TREES, scenarios=[])
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    for spec in scenarios():
        report = dict(name=spec["name"], status="running")
        summary["scenarios"].append(report)
        try:
            run_spec(spec, logger, report)
            report["status"] = "passed"
            print(spec["name"], "PASS", flush=True)
        except Exception:
            report.update(status="failed", error=traceback.format_exc(), logs=list(logger.messages))
            summary["status"] = "failed"
            args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
            raise
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    summary["status"] = "passed"
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print("PASS: CPU/NPU leaf-rule parity, clipping both signs, save/load and refit", flush=True)


if __name__ == "__main__":
    main()