"""Real-NPU training parity, explicit config rejection and cache regression.

Use --config-boundaries-only to exercise unsupported NPU requests and CPU compatibility.
"""
import argparse
import sys
import time
import numpy as np
import lightgbm as lgb


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warning(self, message):
        self.messages.append(message)


logger = CaptureLogger()
lgb.register_logger(logger)

print("lightgbm_library=", lgb.basic._LIB._name, flush=True)
rng = np.random.default_rng(731)
x = rng.normal(size=(1024, 19)).astype(np.float32)
x[rng.random(x.shape) < 0.65] = 0
x[rng.random(x.shape) < 0.04] = np.nan
y = np.nan_to_num(x[:, 0]) * 3 - np.nan_to_num(x[:, 3]) + rng.normal(size=len(x)) * 0.1


def run_config_boundary_checks():
    """Confirm unsupported NPU requests fail instead of changing backend."""
    boundary_x = np.arange(64, dtype=np.float32).reshape(32, 2)
    boundary_y = boundary_x[:, 0] * 0.25 + boundary_x[:, 1]
    common = dict(objective="regression", verbosity=-1, num_leaves=3,
                  min_data_in_leaf=2, force_col_wise=True, num_threads=2)

    def expect_npu_failure(extra, text):
        logger.messages.clear()
        try:
            data = lgb.Dataset(boundary_x, label=boundary_y, free_raw_data=False)
            lgb.train(dict(common, device_type="npu", **extra), data, num_boost_round=1)
        except lgb.basic.LightGBMError as exc:
            message = str(exc)
            assert text in message, (text, message)
            return
        raise AssertionError("unsupported NPU configuration was accepted: " + text)

    expect_npu_failure({"use_quantized_grad": True}, "use_quantized_grad")
    expect_npu_failure({"linear_tree": True}, "linear_tree")

    # The same quantized option remains valid on CPU. This guards against a
    # global parser/config regression while NPU support is intentionally absent.
    data = lgb.Dataset(boundary_x, label=boundary_y, free_raw_data=False)
    model = lgb.train(dict(common, device_type="cpu", use_quantized_grad=True,
                           num_grad_quant_bins=16), data, num_boost_round=1)
    assert np.isfinite(model.predict(boundary_x)).all()
    print("PASS: NPU quantized-gradient and linear-tree requests are explicitly rejected; CPU quantized training remains valid", flush=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config-boundaries-only", action="store_true")
args = parser.parse_args()
if args.config_boundaries_only:
    run_config_boundary_checks()
    sys.exit(0)

for name, extra, weights in [
    ("sparse_nan", {}, None),
    ("zero_missing", {"zero_as_missing": True}, None),
    ("weighted", {}, rng.uniform(0.5, 2.0, len(x))),
    ("bagging_eviction", {"bagging_fraction": 0.75, "bagging_freq": 1,
                          "histogram_pool_size": 0.01}, None),
]:
    params = dict(objective="regression", verbosity=1, num_threads=2,
                  num_leaves=7, min_data_in_leaf=12, max_bin=15,
                  feature_fraction=0.8, seed=17, lambda_l1=0.1, lambda_l2=0.4,
                  force_col_wise=True, use_quantized_grad=False, **extra)
    predictions = {}
    trees = {}
    for device in ["cpu", "npu"]:
        data = lgb.Dataset(x, label=y, weight=weights, params=params, free_raw_data=False)
        start = time.perf_counter()
        logger.messages.clear()
        model = lgb.train(dict(params, device_type=device), data, num_boost_round=4,
                          keep_training_booster=True)
        predictions[device] = model.predict(x)
        # Inference traverses the saved tree; the cached training prediction
        # separately validates score updates and out-of-bag handling.
        cached = model._Booster__inner_predict(data_idx=0).copy()
        assert np.max(np.abs(cached - predictions[device])) < 1e-10, (name, device)
        if device == "npu":
            updates = [m for m in logger.messages if "NPU score update:" in m]
            assert len(updates) == 4, (name, updates)
            assert all("partition_rows_downloaded=0 this tree" in m for m in updates), updates
            assert not any("materialized host rows" in m for m in logger.messages)
        trees[device] = model.dump_model()["tree_info"]
        print(name, device, "seconds=", time.perf_counter() - start, flush=True)
    error = np.max(np.abs(predictions["npu"] - predictions["cpu"]))
    print(name, "prediction_max_abs_error=", error, flush=True)
    assert np.isfinite(predictions["npu"]).all()
    assert error < 2e-5, (name, error)
print("PASS: CPU/NPU training parity and zero row downloads for ordinary score updates", flush=True)

# Exercise consumers which historically depended on final CPU leaf indices.
for name, extra, target in [
    ("binary", {"objective": "binary"}, (y > np.median(y)).astype(np.int32)),
    ("multiclass", {"objective": "multiclass", "num_class": 3},
     np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))),
    ("l1_renew", {"objective": "regression_l1"}, y),
    ("root_only", {"objective": "regression", "min_data_in_leaf": 1000}, y),
]:
    params = dict(verbosity=1, num_threads=2, num_leaves=5, min_data_in_leaf=12,
                  max_bin=15, force_col_wise=True, use_quantized_grad=False, seed=17)
    params.update(extra)
    predictions = {}
    for device in ("cpu", "npu"):
        logger.messages.clear()
        data = lgb.Dataset(x, label=target, free_raw_data=False)
        model = lgb.train(dict(params, device_type=device), data, num_boost_round=2,
                          keep_training_booster=True)
        predictions[device] = model.predict(x)
        cached = model._Booster__inner_predict(data_idx=0).copy()
        assert np.max(np.abs(cached - predictions[device])) < 1e-10, (name, device)
        if device == "npu":
            updates = [m for m in logger.messages if "NPU score update:" in m]
            if name == "root_only":
                assert not updates
                assert all(t["num_leaves"] == 1 for t in model.dump_model()["tree_info"])
            elif name == "l1_renew":
                assert len(updates) == 2
                assert any("materialized host rows" in m for m in logger.messages)
            else:
                expected_updates = 6 if name == "multiclass" else 2
                assert len(updates) == expected_updates, (name, updates)
                assert all("partition_rows_downloaded=0 this tree" in m for m in updates)
        if name == "binary":
            # Refit accepts a host leaf-prediction array as an explicit API
            # input; verify this does not accidentally reuse stale device rows.
            refitted = model.refit(x, target, decay_rate=0.6)
            predictions[device + "_refit"] = refitted.predict(x)
    error = np.max(np.abs(predictions["cpu"] - predictions["npu"]))
    assert np.isfinite(predictions["npu"]).all() and error < 2e-5, (name, error)
    if name == "binary":
        assert np.max(np.abs(predictions["cpu_refit"] - predictions["npu_refit"])) < 2e-5
    print(name, "prediction_max_abs_error=", error, flush=True)
print("PASS: binary, multiclass, leaf renewal, refit and unsplit-root compatibility", flush=True)
