from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metraq_dip.utils.render_window_visualizations import (
    _baseline_surface,
    _experiment_file,
    _load_config,
    _load_window_arrays,
)
from metraq_dip.utils.sensor_error_analysis import (
    summarize_sensor_errors,
    validate_window_aggregates,
)
from metraq_gnn.data import load_aq_sensor_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export matched DIP/KRG/IDW errors per sensor")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=Path("output/experiments/basic/single_channel_supervision_NO2"),
    )
    parser.add_argument("--cache", type=Path, default=Path("cache/metraq/no2-2010-2024"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.experiments)
    expected = pd.read_csv(args.experiments / "results.csv")
    with np.load(args.experiments / "data.npz", allow_pickle=True) as index:
        groups = np.asarray(index["test_sensors"], dtype=np.int64)
        windows = pd.DatetimeIndex(pd.to_datetime(index["time_windows"].tolist()))

    cache = load_aq_sensor_cache(args.cache)
    node_by_sensor = {
        int(sensor_id): int(node)
        for sensor_id, node in zip(cache.sensor_ids, cache.sensor_node_indices, strict=True)
    }
    _, width = cache.grid_shape
    records: list[dict[str, object]] = []

    for group_index, sensors in enumerate(groups, start=1):
        sensor_group = "-".join(str(sensor) for sensor in sensors)
        for window_end in windows:
            experiment_path = _experiment_file(args.experiments, sensor_group, window_end)
            arrays = _load_window_arrays(experiment_path, config)
            surfaces = {
                "DIP": arrays["DIP"],
                "KRG": _baseline_surface(
                    train_data=arrays["train_data"],
                    val_data=arrays["val_data"],
                    train_mask=arrays["train_mask"],
                    val_mask=arrays["val_mask"],
                    method="KRG",
                ),
                "IDW": _baseline_surface(
                    train_data=arrays["train_data"],
                    val_data=arrays["val_data"],
                    train_mask=arrays["train_mask"],
                    val_mask=arrays["val_mask"],
                    method="IDW",
                ),
            }
            for sensor_id in sensors:
                node_index = node_by_sensor[int(sensor_id)]
                row, column = divmod(node_index, width)
                if not arrays["test_mask"][row, column]:
                    raise RuntimeError(f"sensor {sensor_id} is absent from {experiment_path}")
                target = float(arrays["test_data"][row, column])
                for method, surface in surfaces.items():
                    prediction = float(surface[row, column])
                    error = prediction - target
                    records.append(
                        {
                            "group": group_index,
                            "sensor_group": sensor_group,
                            "window_end": window_end,
                            "sensor_id": int(sensor_id),
                            "node_index": node_index,
                            "grid_row": row,
                            "grid_column": column,
                            "method": method,
                            "target": target,
                            "prediction": prediction,
                            "error": error,
                            "absolute_error": abs(error),
                            "squared_error": error * error,
                        }
                    )

    predictions = pd.DataFrame.from_records(records)
    validate_window_aggregates(predictions, expected)
    metrics = summarize_sensor_errors(predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "classical_predictions_by_sensor.csv", index=False)
    metrics.to_csv(args.output / "classical_metrics_by_sensor.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
