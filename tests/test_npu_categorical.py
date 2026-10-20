"""Standalone, real-NPU categorical parity regression (no fallback or skips).

Run with the intended LightGBM installation, for example::

    python tests/test_npu_categorical.py --summary-output categorical-summary.json

This deliberately does not select, build, or replace the LightGBM library.
A missing NPU backend/device is a failure. The JSON summary is updated after
each scenario, and includes the failing scenario and captured logs on failure.
"""

import argparse
import json
from pathlib import Path
import re
import time
import traceback

import lightgbm as lgb
import numpy as np


NUM_TREES = 2
NUM_LEAVES = 5
# Gains in the model dump pass through float32; exact float comparison (or a
# very small absolute-only gain tolerance) would create false failures.
FLOAT_RTOL = 2e-5
FLOAT_ATOL = 2e-5
OUTPUT_ATOL = 2e-5
CATEGORICAL_FEATURES = [0, 1, 4, 5]
FEATURE_NAMES = ["category", "aux_category", "numeric", "noise", "constant", "all_missing"]


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))

    def warning(self, message):
        self.messages.append(str(message))


def assert_float(actual, expected, context, output=False):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, (context, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), context
    np.testing.assert_allclose(
        actual, expected, rtol=0 if output else FLOAT_RTOL,
        atol=OUTPUT_ATOL if output else FLOAT_ATOL, err_msg=context,
    )


def category_set(node):
    assert node["decision_type"] == "==", node
    values = tuple(int(value) for value in node["threshold"].split("||"))
    assert len(values) == len(set(values)) and all(value >= 0 for value in values), node
    return frozenset(values)


def compare_node(cpu, npu, context):
    assert cpu.keys() == npu.keys(), (context, cpu.keys(), npu.keys())
    if "split_index" not in cpu:
        # A single-leaf tree omits leaf_index and leaf_weight in LightGBM's
        # JSON dump. Their presence must still agree between the two models.
        assert "split_index" not in npu, context
        for field in ("leaf_index", "leaf_count"):
            if field in cpu:
                assert cpu[field] == npu[field], (context, field, cpu[field], npu[field])
        assert "leaf_count" in cpu, context
        assert_float(npu["leaf_value"], cpu["leaf_value"], context + ".leaf_value", output=True)
        if "leaf_weight" in cpu:
            assert_float(npu["leaf_weight"], cpu["leaf_weight"], context + ".leaf_weight")
        return
    for field in ("split_index", "split_feature", "decision_type", "default_left",
                  "missing_type", "internal_count"):
        assert cpu[field] == npu[field], (context, field, cpu[field], npu[field])
    if cpu["decision_type"] == "==":
        assert category_set(cpu) == category_set(npu), (context, cpu["threshold"], npu["threshold"])
    else:
        # Numeric thresholds come from identical dataset bin boundaries, so
        # they too must match exactly, rather than merely within tolerance.
        assert cpu["threshold"] == npu["threshold"], (context, "threshold", cpu, npu)
    for field in ("split_gain", "internal_weight", "internal_value"):
        assert_float(npu[field], cpu[field], context + "." + field, output=field == "internal_value")
    compare_node(cpu["left_child"], npu["left_child"], context + ".left")
    compare_node(cpu["right_child"], npu["right_child"], context + ".right")


def compare_trees(cpu, npu, context):
    assert len(cpu) == len(npu), (context, "tree_count", len(cpu), len(npu))
    for index, (cpu_tree, npu_tree) in enumerate(zip(cpu, npu)):
        where = f"{context}.tree[{index}]"
        for field in ("tree_index", "num_leaves", "num_cat"):
            assert cpu_tree[field] == npu_tree[field], (where, field, cpu_tree, npu_tree)
        assert_float(npu_tree["shrinkage"], cpu_tree["shrinkage"], where + ".shrinkage", output=True)
        compare_node(cpu_tree["tree_structure"], npu_tree["tree_structure"], where)


def split_nodes(node):
    if "split_index" in node:
        yield node
        yield from split_nodes(node["left_child"])
        yield from split_nodes(node["right_child"])


def make_data(cardinality, seed, counts=None, missing=False, zero_missing=False, bitset=False):
    rng = np.random.default_rng(seed)
    if counts is None:
        counts = np.full(cardinality, 72, dtype=np.int32)
    categories = np.repeat(np.arange(cardinality), counts).astype(np.float32)
    if missing or bitset:
        # Both NaN and distinct negative integers map to the categorical
        # missing bin. Bitset cases include this bin so all rare tail codes
        # survive the categorical mapper's 99% frequency retention rule.
        categories = np.concatenate((categories, [np.nan] * 12 + [-1.0] * 7 + [-7.0] * 5))
    categories = categories[rng.permutation(len(categories))]
    n = len(categories)
    aux = rng.integers(0, 3, n).astype(np.float32)
    numeric = rng.uniform(-1, 1, n).astype(np.float32)
    noise = rng.normal(size=n).astype(np.float32)
    if missing:
        numeric[rng.random(n) < 0.09] = np.nan
        aux[rng.random(n) < 0.06] = np.nan
    if zero_missing:
        numeric[rng.random(n) < 0.30] = 0
        noise[rng.random(n) < 0.25] = 0
    valid = np.isfinite(categories) & (categories >= 0)
    code = np.where(valid, categories, 0).astype(np.int32)
    if bitset:
        effect = np.where(code >= cardinality - 4, 11.0, -1.0 + 0.006 * code)
        numeric_scale = 0.7
    elif cardinality == 3:
        effect = np.asarray([-7.0, 1.0, 10.0])[code]
        numeric_scale = 3.0
    else:
        # Noncontiguous groups with small within-group differences avoid
        # accidental equal-gradient ordering/ties without losing group splits.
        effect = np.asarray([-7.0, 0.6, 9.0])[(code * 7) % 3] + code * 0.037
        numeric_scale = 1.7
    effect = np.where(valid, effect, -2.3)
    clean_numeric = np.nan_to_num(numeric)
    target = effect + numeric_scale * clean_numeric + 0.23 * np.nan_to_num(aux)
    if zero_missing:
        target += np.where(numeric == 0, 3.7, 0)
    target += 0.013 * noise + rng.normal(0, 0.025, n)
    x = np.column_stack((categories, aux, numeric, noise, np.ones(n), np.full(n, np.nan))).astype(np.float32)
    return x, target


def make_probes(cardinality):
    # Cartesian probes cover every known category, both sides of each bitset
    # boundary, unseen categories, missing/negative codes, and numeric missing
    # and zero values. There is no label-dependent filtering of predictions.
    rows = []
    for code in list(range(cardinality)) + [cardinality + 17, -1, -7, np.nan]:
        for value in (-1.0, 0.0, 1.0, np.nan):
            for aux in (0.0, 2.0, np.nan):
                rows.append([code, aux, value, 0.0, 1.0, np.nan])
    return np.asarray(rows, dtype=np.float32)


def scenarios():
    yield dict(name="onehot_mixed", cardinality=3, seed=801,
               extra=dict(max_cat_to_onehot=8), onehot=True, require_numeric=True)
    yield dict(name="grouped_regularized_limits", cardinality=9, seed=802,
               extra=dict(max_cat_to_onehot=2, max_cat_threshold=3,
                          min_data_per_group=100, cat_smooth=8.0, cat_l2=7.5), grouped=True)
    yield dict(name="nan_negative_categories", cardinality=9, seed=803, missing=True,
               extra=dict(max_cat_to_onehot=2), grouped=True)
    yield dict(name="zero_as_missing", cardinality=3, seed=804, missing=True, zero_missing=True,
               extra=dict(max_cat_to_onehot=8, zero_as_missing=True), onehot=True, require_numeric=True)
    yield dict(name="rare_unbalanced_weighted", cardinality=10, seed=805, missing=True,
               counts=[280, 140, 70, 35, 18, 9, 5, 3, 2, 1], weighted=True,
               extra=dict(max_cat_to_onehot=2, max_cat_threshold=4,
                          min_data_per_group=24, cat_smooth=16.0, cat_l2=12.0))
    for cardinality in (32, 33, 65):
        yield dict(name=f"bitset_{cardinality}", cardinality=cardinality,
                   seed=810 + cardinality, bitset=True,
                   # Unique descending counts make raw code k map to bin k+1.
                   counts=(12 + cardinality - np.arange(cardinality)).tolist(),
                   extra=dict(max_cat_to_onehot=2, max_cat_threshold=8), grouped=True)
    yield dict(name="bagging_small_cache", cardinality=9, seed=806, missing=True,
               # This budget forces HistogramPool's minimum of two cached
               # leaves, below the five leaves required by the coverage check.
               # Both rounds resample rows and reuse resident score buffers.
               extra=dict(max_cat_to_onehot=2, bagging_fraction=0.73, bagging_freq=1,
                          bagging_seed=91, histogram_pool_size=0.000001), bagging=True)
    yield dict(name="unsplittable", cardinality=9, seed=807, missing=True,
               extra=dict(max_cat_to_onehot=2, min_gain_to_split=1e9), unsplittable=True)


def check_coverage(spec, params, data, trees, nrows):
    nodes = [node for tree in trees for node in split_nodes(tree["tree_structure"])]
    main_cat = [node for node in nodes if node["split_feature"] == 0 and node["decision_type"] == "=="]
    if spec.get("unsplittable"):
        # GBDT retains only its first constant tree; subsequent empty trees
        # may be discarded even though two boosting rounds were requested.
        assert 1 <= len(trees) <= NUM_TREES and not nodes, (spec["name"], trees)
        return
    assert len(trees) == NUM_TREES and main_cat, (spec["name"], "categorical path not exercised")
    assert all(1 < tree["num_leaves"] <= NUM_LEAVES for tree in trees), spec["name"]
    if spec.get("onehot"):
        # The branch predicate uses num_bin, INCLUDING the dummy missing bin,
        # not just the number of observed nonnegative category values.
        assert data.feature_num_bin(0) == spec["cardinality"] + 1, spec["name"]
        assert data.feature_num_bin(0) <= params["max_cat_to_onehot"], spec["name"]
        assert all(len(category_set(node)) == 1 for node in main_cat), spec["name"]
    else:
        assert data.feature_num_bin(0) > params["max_cat_to_onehot"], spec["name"]
        assert all(len(category_set(node)) <= params["max_cat_threshold"] for node in main_cat), spec["name"]
    if spec.get("grouped"):
        assert any(len(category_set(node)) > 1 for node in main_cat), (spec["name"], main_cat)
    if spec.get("require_numeric"):
        assert any(node["split_feature"] == 2 and node["decision_type"] == "<=" for node in nodes), spec["name"]
    if spec.get("missing"):
        assert any(node["missing_type"] == "NaN" for node in main_cat), spec["name"]
    if spec.get("zero_missing"):
        assert any(node["split_feature"] == 2 and node["missing_type"] == "Zero" for node in nodes), spec["name"]
    if spec.get("bitset"):
        cardinality = spec["cardinality"]
        assert data.feature_num_bin(0) == cardinality + 1, spec["name"]
        root = trees[0]["tree_structure"]
        assert root["split_feature"] == 0 and cardinality - 1 in category_set(root), (spec["name"], root)
        # With descending category frequencies, the final code occupies
        # logical bin 32/33/65 as well as raw-code bit 31/32/64, respectively.
    if spec.get("bagging"):
        assert all(tree["num_leaves"] == NUM_LEAVES for tree in trees), spec["name"]
        assert all(0 < tree["tree_structure"]["internal_count"] < nrows for tree in trees), spec["name"]


def check_npu_logs(messages, trees, context):
    text = "\n".join(messages)
    assert "NPU runtime initialised on device" in text, (context, text)
    assert "NPU tree learner cached device bins" in text, (context, text)
    searches = re.findall(r"NPU split search: split_calls=(\d+) \(\+(\d+) this tree\), leaves=(\d+)", text)
    assert searches, (context, text)
    assert "runtime winners populate best_split_per_leaf_ used by tree splits" in text, context
    split_trees = sum(tree["num_leaves"] > 1 for tree in trees)
    if split_trees:
        assert len(searches) == NUM_TREES, (context, searches)
        assert all(int(delta) > 0 and int(leaves) > 1 for _, delta, leaves in searches), (context, searches)
    updates = re.findall(r"NPU score update: device_calls=(\d+), partition_rows_downloaded=(\d+) this tree", text)
    assert len(updates) == split_trees, (context, updates, split_trees)
    assert [int(calls) for calls, _ in updates] == list(range(1, split_trees + 1)), (context, updates)
    assert all(int(downloaded) == 0 for _, downloaded in updates), (context, updates)
    assert "materialized host rows" not in text, (context, text)
    assert "fallback" not in text.lower() and "falling back" not in text.lower(), (context, text)
    return dict(split_searches=len(searches), score_updates=len(updates), partition_rows_downloaded=0)


def run_device(spec, device, logger, report):
    x, y = make_data(spec["cardinality"], spec["seed"], counts=spec.get("counts"),
                     missing=spec.get("missing", False), zero_missing=spec.get("zero_missing", False),
                     bitset=spec.get("bitset", False))
    probes = make_probes(spec["cardinality"])
    weights = None
    if spec.get("weighted"):
        weights = np.random.default_rng(spec["seed"] + 1000).uniform(0.6, 1.8, len(x))
    params = dict(objective="regression", verbosity=1, num_threads=2, num_leaves=NUM_LEAVES,
                  learning_rate=0.15, min_data_in_leaf=8, min_data_in_bin=1, max_bin=127,
                  feature_pre_filter=False, force_col_wise=True, use_quantized_grad=False,
                  feature_fraction=1.0, seed=17, data_random_seed=17,
                  bin_construct_sample_cnt=10000, lambda_l1=0.1, lambda_l2=0.4,
                  max_cat_to_onehot=2, max_cat_threshold=16, min_data_per_group=8,
                  cat_smooth=0.0, cat_l2=0.0)
    params.update(spec["extra"])
    snapshots = []
    logger.messages.clear()
    start = time.perf_counter()
    data = lgb.Dataset(x, label=y, weight=weights, categorical_feature=CATEGORICAL_FEATURES,
                       feature_name=FEATURE_NAMES, params=params, free_raw_data=False)

    def snapshot(env):
        model = env.model
        prediction = model.predict(x, raw_score=True)
        cached = model._Booster__inner_predict(data_idx=0).copy()
        assert_float(cached, prediction, f"{spec['name']}.{device}.round{env.iteration}.cached", output=True)
        snapshots.append(dict(trees=model.dump_model()["tree_info"], train=prediction,
                              cached=cached, probes=model.predict(probes, raw_score=True),
                              train_leaves=model.predict(x, pred_leaf=True),
                              probe_leaves=model.predict(probes, pred_leaf=True)))

    # Device errors propagate. There is deliberately no retry on CPU, skip,
    # exception-to-success conversion, library swap, or backend auto-detection.
    model = lgb.train(dict(params, device_type=device), data, num_boost_round=NUM_TREES,
                      keep_training_booster=True, callbacks=[snapshot])
    assert len(snapshots) == NUM_TREES, (spec["name"], device, len(snapshots))
    trees = snapshots[-1]["trees"]
    check_coverage(spec, params, data, trees, len(x))
    report[device] = dict(seconds=time.perf_counter() - start, rows=len(x),
                          num_bins=[data.feature_num_bin(i) for i in range(x.shape[1])],
                          tree_leaves=[tree["num_leaves"] for tree in trees])
    if device == "npu":
        report[device]["runtime_evidence"] = check_npu_logs(logger.messages, trees, spec["name"])
        report[device]["logs"] = list(logger.messages)
    restored = lgb.Booster(model_str=model.model_to_string())
    compare_trees(trees, restored.dump_model()["tree_info"], f"{spec['name']}.{device}.roundtrip")
    for label, leaf_field, matrix in (("train", "train_leaves", x),
                                      ("probes", "probe_leaves", probes)):
        assert_float(restored.predict(matrix, raw_score=True), snapshots[-1][label],
                     f"{spec['name']}.{device}.roundtrip.{label}", output=True)
        np.testing.assert_array_equal(restored.predict(matrix, pred_leaf=True),
                                      snapshots[-1][leaf_field],
                                      err_msg=f"{spec['name']}.{device}.roundtrip.{label}.leaves")
    return snapshots


def run_scenario(spec, logger, report):
    results = {device: run_device(spec, device, logger, report) for device in ("cpu", "npu")}
    max_error = 0.0
    for index, (cpu, npu) in enumerate(zip(results["cpu"], results["npu"])):
        context = f"{spec['name']}.round{index + 1}"
        compare_trees(cpu["trees"], npu["trees"], context)
        for field in ("train", "probes", "cached"):
            assert_float(npu[field], cpu[field], context + "." + field, output=True)
            max_error = max(max_error, float(np.max(np.abs(npu[field] - cpu[field]))))
        for field in ("train_leaves", "probe_leaves"):
            np.testing.assert_array_equal(npu[field], cpu[field], err_msg=context + "." + field)
    report["prediction_max_abs_error"] = max_error
    report["status"] = "passed"
    print(spec["name"], "PASS prediction_max_abs_error=", max_error, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-output", type=Path, default=Path("npu_categorical_summary.json"))
    args = parser.parse_args()
    logger = CaptureLogger()
    lgb.register_logger(logger)
    library = str(lgb.basic._LIB._name)
    print("lightgbm_library=", library, flush=True)
    summary = dict(status="running", library=library, num_boost_round=NUM_TREES,
                   num_leaves=NUM_LEAVES,
                   tolerances=dict(gain_weight_rtol=FLOAT_RTOL, gain_weight_atol=FLOAT_ATOL,
                                   output_prediction_rtol=0, output_prediction_atol=OUTPUT_ATOL),
                   scenarios=[])

    def save():
        args.summary_output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    save()
    for spec in scenarios():
        report = dict(name=spec["name"], cardinality=spec["cardinality"],
                      parameters=spec["extra"], status="running")
        summary["scenarios"].append(report)
        try:
            run_scenario(spec, logger, report)
        except Exception:
            report.update(status="failed", error=traceback.format_exc(), last_logs=list(logger.messages))
            summary["status"] = "failed"
            save()
            raise
        save()
    summary["status"] = "passed"
    save()
    print("PASS: CPU/NPU categorical structures, cached scores, predictions, leaves and serialization", flush=True)
    print("summary_output=", args.summary_output.resolve(), flush=True)


if __name__ == "__main__":
    main()