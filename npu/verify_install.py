"""Check installed paths, the environment-free qrun wrapper and both NPU modes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import lightgbm as lgb
import lightgbm.basic as basic
import numpy as np
import qlib
from ruamel.yaml import YAML


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.installation.read_text())
    library = Path(basic._LIB._name).resolve()
    assert library == Path(report["native_library"]).resolve(), str(library)
    assert hashlib.sha256(library.read_bytes()).hexdigest() == report["native_sha256"]
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / "npu/reference/qlib_overlay_manifest.json").read_text())
    installed = Path(qlib.__file__).parent
    for relative, expected in manifest.items():
        target = installed / Path(relative).relative_to("qlib")
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected["patched_sha256"], str(target)
    config = YAML(typ="safe").load(Path(report["workflow"]).read_text())
    params = config["task"]["model"]["kwargs"]
    assert params["device"] == "npu" and params["max_depth"] == -1
    assert params["gpu_use_dp"] is False
    records = {x["class"] for x in config["task"]["record"]}
    assert {"SignalRecord", "SigAnaRecord", "PortAnaRecord"} <= records
    clean = dict(os.environ)
    for key in ["LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "TBE_IMPL_PATH"]:
        clean.pop(key, None)
    for command in report["qrun"]:
        result = subprocess.run([command, "--help"], env=clean, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    rng = np.random.default_rng(20261110)
    x = rng.normal(size=(257, 7))
    y = x[:, 0] * 0.7 - x[:, 2] * 0.2
    modes = []
    for dp in [False, True]:
        p = {"objective": "mse", "device": "npu", "gpu_use_dp": dp, "max_depth": -1,
             "num_leaves": 4, "min_data_in_leaf": 5, "num_threads": 2, "verbosity": -1}
        model = lgb.train(p, lgb.Dataset(x, label=y), num_boost_round=3)
        prediction = model.predict(x, num_threads=2)
        assert prediction.shape == (257,) and np.isfinite(prediction).all()
        assert model.params["device"] == "npu"
        modes.append({"gpu_use_dp": dp, "trees": model.num_trees()})
    result = {"pass": True, "library": str(library), "sha256": report["native_sha256"],
              "qlib": str(installed), "qrun_clean_environment_pass": True, "npu_smoke_modes": modes}
    (args.installation.parent / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
