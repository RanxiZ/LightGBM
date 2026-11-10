"""Compare Alpha158 inputs and a trusted qrun model with the frozen reference."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from qlib.contrib.eva.alpha import calc_ic
from qlib.data.dataset.handler import DataHandlerLP

from reference.collect_inputs import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/workspace/qlib/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml"))
    parser.add_argument("--artifacts", type=Path, help="Trusted MLflow artifacts directory containing params.pkl and pred.pkl")
    parser.add_argument("--output", type=Path, default=Path("npu_reference_check.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    summary, _, _, dataset = collect(str(args.config))
    expected = json.loads((root / "reference/input_summary.json").read_text())
    result = {"input_fingerprints_match": summary == expected, "actual_inputs": summary}
    if summary != expected:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise SystemExit("The data differs from the tested reference; recorded metrics/timing are not a reproducibility claim for this dataset.")
    if args.artifacts:
        model = pickle.load((args.artifacts / "params.pkl").open("rb")).model
        assert model.params["device"] == "npu" and model.params["max_depth"] == -1
        assert model.params.get("gpu_use_dp", False) is False
        assert model.best_iteration == 13
        reference = json.loads((root / "reference/formal_acceptance.json").read_text())["runs"][0]["segments"]
        result["segments"] = {}
        for segment in ["valid", "test"]:
            frame = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_L if segment == "valid" else DataHandlerLP.DK_I)
            from qlib.workflow.record_temp import SignalRecord
            label = (dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L) if segment == "valid" else SignalRecord.generate_label(dataset)).iloc[:, 0]
            assert frame.index.equals(label.index)
            pred = model.predict(frame.to_numpy(), num_threads=20)
            assert np.isfinite(pred).all()
            if segment == "test":
                saved = pd.read_pickle(args.artifacts / "pred.pkl").iloc[:, 0]
                assert saved.index.equals(frame.index) and np.array_equal(saved.to_numpy(), pred)
            ic, ric = calc_ic(pd.Series(pred, index=frame.index), label)
            mask = np.isfinite(label.to_numpy())
            metrics = {"MSE": float(np.mean((pred[mask] - label.to_numpy()[mask]) ** 2))}
            for name, values in [("IC", ic), ("RankIC", ric)]:
                values = values.dropna()
                metrics[name] = float(values.mean())
                metrics[name + "IR"] = float(values.mean() / values.std(ddof=1))
            delta = {key: value - reference[segment]["metrics"][key] for key, value in metrics.items()}
            result["segments"][segment] = {"metrics": metrics, "delta_vs_reference": delta}
            assert all(abs(value) <= 1e-10 for value in delta.values()), delta
    result["pass"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Reference inputs and requested model checks passed:", args.output)


if __name__ == "__main__":
    main()
