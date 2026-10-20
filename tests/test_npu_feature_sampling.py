"""Standalone CPU/NPU feature_fraction_bynode parity regression.

This test consumes an already-built LightGBM installation. It intentionally does
not build, select, or replace a library, and it does not skip when NPU training
is unavailable.
"""

import argparse
import json
from pathlib import Path
import re
import traceback

import lightgbm as lgb
import numpy as np

from test_npu_categorical import FLOAT_ATOL, FLOAT_RTOL, assert_float, compare_trees, make_data


NUM_TREES = 5
NUM_LEAVES = 5


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))

    def warning(self, message):
        self.messages.append(str(message))


def equivalent(left, right):
    """Compare structure and leaf assignments, including prediction values."""
    compare_trees(left["trees"], right["trees"], "feature_sampling")
    np.testing.assert_array_equal(left["leaves"], right["leaves"])
    assert_float(left["predictions"], right["predictions"], "predictions", output=True)


def make_specs():
    # make_data() produces 6 columns (4 useful mixed/categorical/numeric
    # columns plus constant and all-missing columns), with 216--512 rows.
    return [
        dict(name="mixed_categorical", cardinality=3, seed=901, missing=True,
             counts=[72, 72, 72], extra=dict(max_cat_to_onehot=8)),
        dict(name="bytree_combined", cardinality=9, seed=902, counts=[32] * 9,
             extra=dict(feature_fraction=0.8, max_cat_to_onehot=2,
                        max_cat_threshold=4)),
        dict(name="bagging_tiny_cache", cardinality=3, seed=903, missing=True,
             counts=[72, 72, 72],
             extra=dict(max_cat_to_onehot=8, bagging_fraction=0.73, bagging_freq=1,
                        bagging_seed=47, histogram_pool_size=0.000001)),
        dict(name="interaction_constraints", cardinality=3, seed=904, missing=True,
             counts=[72, 72, 72],
             extra=dict(max_cat_to_onehot=8,
                        interaction_constraints="[[0, 1, 2], [3, 4, 5]]")),
    ]


def base_params(spec, bynode):
    params = dict(objective="regression", verbosity=1, num_threads=2,
                  num_leaves=NUM_LEAVES, learning_rate=0.15, min_data_in_leaf=8,
                  min_data_in_bin=1, max_bin=15, feature_pre_filter=False,
                  force_col_wise=True, use_quantized_grad=False,
                  feature_fraction=1.0, feature_fraction_bynode=bynode,
                  feature_fraction_seed=131, seed=17, data_random_seed=17,
                  bin_construct_sample_cnt=10000, lambda_l1=0.1, lambda_l2=0.4,
                  max_cat_to_onehot=2, max_cat_threshold=8,
                  min_data_per_group=8, cat_smooth=0.0, cat_l2=0.0)
    params.update(spec["extra"])
    return params


def check_runtime_logs(messages, name, expected_trees):
    text = "\n".join(messages)
    assert "NPU runtime initialised on device" in text, (name, text)
    assert "NPU tree learner cached device bins" in text, (name, text)
    searches = re.findall(
        r"NPU split search: split_calls=(\d+) \(\+(\d+) this tree\), leaves=(\d+)", text)
    assert len(searches) == expected_trees, (name, searches, text)
    assert all(int(delta) > 0 and int(leaves) > 1 for _, delta, leaves in searches), (name, searches)
    # This is evidence that the NPU path, rather than CPU fallback or mere
    # parameter acceptance, actually searched and built every requested tree.
    assert "runtime winners populate best_split_per_leaf_ used by tree splits" in text, name
    assert "fallback" not in text.lower() and "falling back" not in text.lower(), (name, text)


def train(spec, device, bynode, logger, reset=False):
    x, y = make_data(spec["cardinality"], spec["seed"], counts=spec.get("counts"),
                     missing=spec.get("missing", False))
    params = base_params(spec, bynode)
    data = lgb.Dataset(x, label=y, categorical_feature=[0, 1, 4, 5],
                       free_raw_data=False, params=params)
    logger.messages.clear()
    if reset:
        initial_params = dict(params, feature_fraction_bynode=1.0)
        model = lgb.train(dict(initial_params, device_type=device), data, num_boost_round=1,
                          keep_training_booster=True)
        model.reset_parameter({"feature_fraction_bynode": bynode})
        for _ in range(NUM_TREES - 1):
            assert not model.update(), "training unexpectedly stopped during reset regression"
    else:
        model = lgb.train(dict(params, device_type=device), data, num_boost_round=NUM_TREES,
                          keep_training_booster=True)
    result = dict(trees=model.dump_model()["tree_info"],
                  predictions=model.predict(x, raw_score=True),
                  leaves=model.predict(x, pred_leaf=True))
    if device == "npu":
        check_runtime_logs(logger.messages, "%s.%s" % (spec["name"], bynode), NUM_TREES)
    return result


def run_spec(spec, logger, report):
    # Test both fractions against 1.0 and compare CPU with NPU for every run.
    # The changed-behaviour assertion prevents a test that merely accepts the
    # option while continuing to train as if the fraction were one.
    baseline = {}
    changed = {}
    for fraction in (1.0, 0.5, 0.8):
        cpu = train(spec, "cpu", fraction, logger)
        npu = train(spec, "npu", fraction, logger)
        equivalent(cpu, npu)
        baseline[fraction] = cpu
        if fraction != 1.0:
            try:
                equivalent(cpu, baseline[1.0])
            except AssertionError:
                changed[fraction] = True
            else:
                changed[fraction] = False
    assert any(changed.values()), (spec["name"], "fractions did not change structure or leaves", changed)
    report["changed_vs_fraction1"] = changed

    # Run the changed configuration again to catch sampling-order or seed
    # state leakage between training sessions.
    repeat_cpu = train(spec, "cpu", 0.5, logger)
    repeat_npu = train(spec, "npu", 0.5, logger)
    equivalent(repeat_cpu, baseline[0.5])
    equivalent(repeat_npu, baseline[0.5])
    report["repeatability"] = True

    # Exercise NPUTreeLearner::ResetConfig without changing the data or seed.
    reset_cpu = train(spec, "cpu", 0.5, logger, reset=True)
    reset_npu = train(spec, "npu", 0.5, logger, reset=True)
    equivalent(reset_cpu, reset_npu)
    report["reset_config_parity"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-output", type=Path, default=Path("npu_feature_sampling_summary.json"))
    args = parser.parse_args()
    logger = CaptureLogger()
    lgb.register_logger(logger)
    summary = dict(status="running", library=str(lgb.basic._LIB._name),
                   fractions=[0.5, 0.8], num_boost_round=NUM_TREES, scenarios=[])
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    for spec in make_specs():
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
    print("PASS: CPU/NPU by-node feature sampling parity, repeatability, cache, bagging and reset", flush=True)


if __name__ == "__main__":
    main()