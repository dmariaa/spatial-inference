from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine matched GNN and classical sensor metrics")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--experiment-index", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classical = pd.read_csv(args.analysis / "classical_metrics_by_sensor.csv")
    with np.load(args.experiment_index, allow_pickle=True) as index:
        groups = np.asarray(index["test_sensors"], dtype=np.int64)

    gnn_frames = []
    for group, sensors in enumerate(groups, start=1):
        path = args.analysis / "gnn" / f"group-{group:02d}-metrics.csv"
        frame = pd.read_csv(path)
        frame["group"] = group
        frame["sensor_group"] = "-".join(str(sensor) for sensor in sensors)
        frame["method"] = "GNN"
        gnn_frames.append(frame)
    metrics = pd.concat([classical, *gnn_frames], ignore_index=True)
    metrics["count"] = metrics["count"].astype(int)
    metrics = metrics.sort_values(["group", "sensor_id", "method"]).reset_index(drop=True)

    comparisons = []
    for (group, sensor_id), frame in metrics.groupby(["group", "sensor_id"], sort=True):
        by_method = frame.set_index("method")
        best_classical = by_method.loc[["DIP", "KRG", "IDW"], "mae"].idxmin()
        best_classical_mae = float(by_method.loc[best_classical, "mae"])
        gnn_mae = float(by_method.loc["GNN", "mae"])
        comparisons.append(
            {
                "group": group,
                "sensor_group": frame["sensor_group"].iloc[0],
                "sensor_id": sensor_id,
                "gnn_mae": gnn_mae,
                "best_classical_method": best_classical,
                "best_classical_mae": best_classical_mae,
                "gnn_minus_best_classical_mae": gnn_mae - best_classical_mae,
                "winner": "GNN" if gnn_mae < best_classical_mae else best_classical,
            }
        )
    comparison = pd.DataFrame.from_records(comparisons)

    metrics.to_csv(args.analysis / "all_methods_metrics_by_sensor.csv", index=False)
    comparison.to_csv(args.analysis / "gnn_vs_best_classical_by_sensor.csv", index=False)
    print(comparison.to_string(index=False))
    print("\nWinner counts:")
    print(comparison["winner"].value_counts().to_string())
    print("\nGNN delta by group (negative is better):")
    print(
        comparison.groupby("group")["gnn_minus_best_classical_mae"]
        .agg(["mean", "min", "max"])
        .to_string()
    )


if __name__ == "__main__":
    main()
