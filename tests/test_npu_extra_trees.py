"""Real-device CPU/NPU extra_trees parity regression, using an existing library.

Run after rebuilding the NPU backend::

    python tests/test_npu_extra_trees.py --summary-output extra-trees-summary.json

No builds, library swaps, backend auto-detection, CPU fallbacks, or device skips
are performed. NPU execution must be demonstrated by the captured runtime logs.
"""

import argparse
import json
from pathlib import Path
import re
import time
import traceback

import lightgbm as lgb
import numpy as np

from test_npu_categorical import (
    CaptureLogger,
    FLOAT_ATOL,
    FLOAT_RTOL,
    OUTPUT_ATOL,
    assert_float,
    category_set,
    compare_trees,
    make_data,
    make_probes,
    split_nodes,
)


NUM_TREES = 3
NUM_LEAVES = 5
EXTRA_SEEDS = (3, 18, 41)


def scenarios():
    yield dict(name="numeric", kind="numeric", extra={})
    yield dict(name="numeric_nan", kind="numeric", missing=True, extra={})
    yield dict(name="numeric_sparse_nan", kind="numeric", missing=True, sparse=True, extra={})
    yield dict(name="numeric_zero_missing", kind="numeric", missing=True,
               extra=dict(zero_as_missing=True, path_smooth=1.7, max_delta_step=2.0))
    # The two-bin numeric branch must not advance the RNG. Three bins advances
    # it even though NextInt(0, 1) necessarily returns zero.
    for bins in (2, 3):
        yield dict(name=f"numeric_{bins}_bins", kind="numeric", narrow=bins,
                   seed_effect=False, extra=dict(max_bin=bins))
    yield dict(name="categorical_onehot", kind="categorical", cardinality=3,
               missing=True, extra=dict(max_cat_to_onehot=8))
    yield dict(name="categorical_grouped_filtered", kind="categorical", cardinality=9,
               counts=[140, 110, 90, 70, 55, 35, 15, 8, 3], missing=True,
               extra=dict(max_cat_to_onehot=2, max_cat_threshold=4, cat_smooth=16.0,
                          cat_l2=3.0, min_data_per_group=12))
    yield dict(name="mixed_bynode_bytree", kind="categorical", cardinality=9, missing=True,
               extra=dict(max_cat_to_onehot=2, feature_fraction=0.8,
                          feature_fraction_bynode=0.5))
    yield dict(name="mixed_bagging_small_cache", kind="categorical", cardinality=9, missing=True,
               extra=dict(max_cat_to_onehot=2, feature_fraction_bynode=0.8,
                          bagging_fraction=0.73, bagging_freq=1, bagging_seed=47,
                          histogram_pool_size=0.000001, path_smooth=2.3))


def data_for(spec):
    if spec["kind"] == "categorical":
        x, y = make_data(spec["cardinality"], 1021, counts=spec.get("counts"),
                         missing=spec.get("missing", False))
        return x, y, make_probes(spec["cardinality"]), [0, 1, 4, 5]
    rng = np.random.default_rng(1022)
    z = rng.normal(size=(480, 4)).astype(np.float32)
    if spec.get("sparse"):
        # Positive nonzeros make the omitted most-frequent zero bin logical
        # bin 0. With NaNs this exercises the forward residual candidate too.
        z = np.abs(z)
        z[rng.random(z.shape) < 0.72] = 0.0
    if spec.get("narrow"):
        # Isolate the low-bin branch, with both omitted and stored bin zero
        # layouts. Constant columns before useful columns also check that RNG
        # seeds use internal feature indices rather than original indices.
        z = (z > 0).astype(np.float32) if spec["narrow"] == 2 else np.round(z).clip(-1, 1)
    if spec.get("missing"):
        z[rng.random(z.shape) < 0.13] = np.nan
        z[rng.random(z.shape) < 0.28] = 0.0
    clean = np.nan_to_num(z)
    y = 6.3 * clean[:, 0] - 3.7 * clean[:, 1] + 1.1 * clean[:, 2] ** 2
    y += 2.9 * np.isnan(z[:, 0]) + 0.07 * clean[:, 3] + rng.normal(0, 0.02, len(z))
    x = np.column_stack((np.ones(len(z)), z[:, 0], np.full(len(z), np.nan), z[:, 1:])).astype(np.float32)
    probes = x[:96].copy()
    probes[::4, 1] = np.nan
    probes[1::4, 1] = 0
    probes[2::4, 1] = -8
    probes[3::4, 1] = 8
    return x, y, probes, []


def base_params(spec, seed):
    params = dict(objective="regression", verbosity=1, num_threads=2,
                  num_leaves=NUM_LEAVES, learning_rate=0.15, min_data_in_leaf=8,
                  min_data_in_bin=1, max_bin=15, feature_pre_filter=False,
                  force_col_wise=True, use_quantized_grad=False, extra_trees=True,
                  extra_seed=seed, seed=17, data_random_seed=17,
                  feature_fraction_seed=131, bin_construct_sample_cnt=10000,
                  lambda_l1=0.1, lambda_l2=0.4, max_cat_to_onehot=2,
                  max_cat_threshold=8, min_data_per_group=8, cat_smooth=0.0, cat_l2=0.0)
    params.update(spec["extra"])
    return params


def check_runtime(messages, snapshots, context):
    text = "\n".join(messages)
    assert "NPU runtime initialised on device" in text, (context, text)
    assert "NPU tree learner cached device bins" in text, (context, text)
    searches = re.findall(
        r"NPU split search: split_calls=(\d+) \(\+(\d+) this tree\), leaves=(\d+)", text)
    assert len(searches) == NUM_TREES, (context, searches, text)
    assert all(int(delta) > 0 and 1 <= int(leaves) <= NUM_LEAVES
               for _, delta, leaves in searches), (context, searches)
    assert "runtime winners populate best_split_per_leaf_ used by tree splits" in text, context
    # Random thresholds can be invalid. Such searches legitimately produce a
    # root-only tree, which GBDT can discard after its first iteration.
    split_attempts = sum(int(leaves) > 1 for _, _, leaves in searches)
    split_trees = sum(tree["num_leaves"] > 1 for tree in snapshots[-1]["trees"])
    assert split_attempts == split_trees, (context, searches, split_trees)
    updates = re.findall(
        r"NPU score update: device_calls=(\d+), partition_rows_downloaded=(\d+) this tree", text)
    assert len(updates) == split_attempts, (context, updates, split_attempts)
    assert [int(calls) for calls, _ in updates] == list(range(1, split_attempts + 1)), (context, updates)
    assert all(int(downloaded) == 0 for _, downloaded in updates), (context, updates)
    assert "materialized host rows" not in text, (context, text)
    assert "fallback" not in text.lower() and "falling back" not in text.lower(), (context, text)
    return dict(split_searches=len(searches), split_trees=split_trees,
                root_only_searches=NUM_TREES - split_attempts, score_updates=len(updates),
                partition_rows_downloaded=0)


def compare_snapshots(expected, actual, context):
    assert len(actual) == len(expected) == NUM_TREES, context
    for index, (left, right) in enumerate(zip(expected, actual)):
        where = f"{context}.round{index + 1}"
        compare_trees(left["trees"], right["trees"], where)
        for field in ("train", "probes", "cached"):
            assert_float(right[field], left[field], where + "." + field, output=True)
        for field in ("train_leaves", "probe_leaves"):
            np.testing.assert_array_equal(right[field], left[field], err_msg=where + "." + field)


def structure_signature(snapshots):
    def node_signature(node):
        if "split_index" not in node:
            return (node.get("leaf_index"), node["leaf_count"])
        return (node["split_feature"], node["decision_type"], node["threshold"],
                node["default_left"], node["missing_type"],
                node_signature(node["left_child"]), node_signature(node["right_child"]))
    return tuple(node_signature(tree["tree_structure"]) for tree in snapshots[-1]["trees"])


def train(spec, device, seed, logger, report, reset=None):
    x, y, probes, categorical = data_for(spec)
    params = base_params(spec, seed)
    snapshots = []
    logger.messages.clear()
    start = time.perf_counter()
    data = lgb.Dataset(x, label=y, categorical_feature=categorical,
                       free_raw_data=False, params=params)
    initial_params = dict(params, device_type=device)
    if reset == "toggle":
        initial_params["extra_trees"] = False

    def capture(model):
        prediction = model.predict(x, raw_score=True)
        cached = model._Booster__inner_predict(data_idx=0).copy()
        assert_float(cached, prediction, f"{spec['name']}.{device}.cached", output=True)
        snapshots.append(dict(trees=model.dump_model()["tree_info"], train=prediction,
                              cached=cached, probes=model.predict(probes, raw_score=True),
                              train_leaves=model.predict(x, pred_leaf=True),
                              probe_leaves=model.predict(probes, pred_leaf=True)))

    if reset is None:
        model = lgb.train(initial_params, data, num_boost_round=NUM_TREES,
                          keep_training_booster=True, callbacks=[lambda env: capture(env.model)])
    else:
        model = lgb.train(initial_params, data, num_boost_round=1, keep_training_booster=True)
        capture(model)
        # CPU SetFeatureInfo<false, true> reseeds on EVERY ResetConfig, not
        # merely a changed extra_seed. The unrelated-lambda reset catches that.
        changes = dict(lambda_l2=0.65) if reset == "same_seed" else (
            dict(extra_seed=seed + 9) if reset == "new_seed" else dict(extra_trees=True))
        model.reset_parameter(changes)
        for _ in range(NUM_TREES - 1):
            model.update()  # A random invalid threshold may legitimately finish this round.
            capture(model)
    assert len(snapshots) == NUM_TREES, (spec["name"], device)
    trees = snapshots[-1]["trees"]
    assert 1 <= len(trees) <= NUM_TREES, (spec["name"], trees)
    assert all(1 <= tree["num_leaves"] <= NUM_LEAVES for tree in trees), spec["name"]
    if spec.get("narrow"):
        bins = [data.feature_num_bin(i) for i in range(x.shape[1])]
        assert all(value <= spec["narrow"] for value in bins), (spec["name"], bins)
        assert spec["narrow"] in bins, (spec["name"], bins)
    if spec["extra"].get("bagging_freq"):
        for tree in trees:
            if tree["num_leaves"] > 1:
                assert 0 < tree["tree_structure"]["internal_count"] < len(x), spec["name"]
    report.update(seconds=time.perf_counter() - start,
                  num_bins=[data.feature_num_bin(i) for i in range(x.shape[1])],
                  tree_leaves=[tree["num_leaves"] for tree in trees])
    if device == "npu":
        report["runtime_evidence"] = check_runtime(logger.messages, snapshots, spec["name"])
        report["logs"] = list(logger.messages)
    restored = lgb.Booster(model_str=model.model_to_string())
    compare_trees(trees, restored.dump_model()["tree_info"], f"{spec['name']}.{device}.serialized")
    for field, leaves, matrix in (("train", "train_leaves", x), ("probes", "probe_leaves", probes)):
        assert_float(restored.predict(matrix, raw_score=True), snapshots[-1][field],
                     f"{spec['name']}.{device}.serialized.{field}", output=True)
        np.testing.assert_array_equal(restored.predict(matrix, pred_leaf=True), snapshots[-1][leaves])
    return snapshots


def run_scenario(spec, logger, report):
    results = {}
    all_nodes = []
    report["runs"] = []
    for seed in EXTRA_SEEDS:
        pair = {}
        for device in ("cpu", "npu"):
            entry = dict(device=device, extra_seed=seed)
            report["runs"].append(entry)
            pair[device] = train(spec, device, seed, logger, entry)
        compare_snapshots(pair["cpu"], pair["npu"], f"{spec['name']}.seed{seed}")
        results[seed] = pair["cpu"]
        all_nodes.extend(node for tree in pair["cpu"][-1]["trees"]
                         for node in split_nodes(tree["tree_structure"]))
    # Permit a root-only result for any individual seed/round, but forbid an
    # entire scenario passing without actually exercising tree splits.
    assert all_nodes, (spec["name"], "no seed produced a split")
    if spec["kind"] == "categorical":
        categories = [node for node in all_nodes if node["split_feature"] == 0 and node["decision_type"] == "=="]
        assert categories, (spec["name"], "main categorical feature never split")
        if spec["cardinality"] == 3:
            assert all(len(category_set(node)) == 1 for node in categories), spec["name"]
        if "grouped_filtered" in spec["name"]:
            assert any(len(category_set(node)) > 1 for node in categories), spec["name"]
    seed_effect = len({structure_signature(value) for value in results.values()}) > 1
    if spec.get("seed_effect", True):
        assert seed_effect, (spec["name"], "extra_seed never changed thresholds/structure/leaves")
    report["extra_seed_changes_structure"] = seed_effect
    for device in ("cpu", "npu"):
        entry = dict(device=device, extra_seed=EXTRA_SEEDS[0], repeat=True)
        report["runs"].append(entry)
        repeated = train(spec, device, EXTRA_SEEDS[0], logger, entry)
        compare_snapshots(results[EXTRA_SEEDS[0]], repeated, spec["name"] + ".repeat." + device)
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-output", type=Path, default=Path("npu_extra_trees_summary.json"))
    args = parser.parse_args()
    logger = CaptureLogger()
    lgb.register_logger(logger)
    summary = dict(status="running", library=str(lgb.basic._LIB._name),
                   num_boost_round=NUM_TREES, num_leaves=NUM_LEAVES, extra_seeds=EXTRA_SEEDS,
                   tolerances=dict(gain_weight_rtol=FLOAT_RTOL, gain_weight_atol=FLOAT_ATOL,
                                   output_prediction_rtol=0, output_prediction_atol=OUTPUT_ATOL), scenarios=[])
    print("lightgbm_library=", summary["library"], flush=True)

    def save():
        args.summary_output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    save()
    specs = list(scenarios())
    jobs = [(spec, None) for spec in specs]
    jobs += [(specs[0], reset) for reset in ("same_seed", "new_seed", "toggle")]
    grouped_spec = next(spec for spec in specs if spec["name"] == "categorical_grouped_filtered")
    jobs += [(grouped_spec, "same_seed")]
    for spec, reset in jobs:
        report = dict(name=spec["name"] + (".reset_" + reset if reset else ""),
                      parameters=spec["extra"], status="running")
        summary["scenarios"].append(report)
        try:
            if reset is None:
                run_scenario(spec, logger, report)
            else:
                report["runs"] = []
                pair = {}
                for device in ("cpu", "npu"):
                    entry = dict(device=device, extra_seed=EXTRA_SEEDS[0], reset=reset)
                    report["runs"].append(entry)
                    pair[device] = train(spec, device, EXTRA_SEEDS[0], logger, entry, reset=reset)
                compare_snapshots(pair["cpu"], pair["npu"], report["name"])
                assert any(tree["num_leaves"] > 1 for tree in pair["cpu"][-1]["trees"]), report["name"]
                report["status"] = "passed"
            print(report["name"], "PASS", flush=True)
        except Exception:
            report.update(status="failed", error=traceback.format_exc(), last_logs=list(logger.messages))
            summary["status"] = "failed"
            save()
            raise
        save()
    summary["status"] = "passed"
    save()
    print("PASS: CPU/NPU extra_trees structures, leaves, scores, seeds, repeatability, serialization and resets", flush=True)
    print("summary_output=", args.summary_output.resolve(), flush=True)


if __name__ == "__main__":
    main()