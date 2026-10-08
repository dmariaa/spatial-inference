from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metraq_dip.tools.interpolator import IdwInterpolator, KrigingInterpolator
from metraq_gnn.data import load_aq_sensor_cache, split_sensor_nodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IDW and Kriging on the GNN spatial split")
    parser.add_argument("--cache", type=Path, default=Path("cache/metraq/no2-2010-2024"))
    parser.add_argument("--output", type=Path, default=Path("output/gnn/classical-comparison"))
    parser.add_argument("--start", default="2024-01-01 23:00")
    parser.add_argument("--end", default="2024-12-31 23:00")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-nodes", type=int, default=4)
    parser.add_argument("--test-nodes", type=int, default=4)
    parser.add_argument("--max-windows", type=int)
    return parser.parse_args()


def aggregate_nodes(cache, time_position: int) -> tuple[np.ndarray, np.ndarray]:
    pollutant = 0
    values = np.zeros(int(cache.graph.num_nodes), dtype=np.float64)
    counts = np.zeros_like(values, dtype=np.int64)
    for sensor_position, node in enumerate(cache.sensor_node_indices):
        if cache.availability[time_position, sensor_position, pollutant]:
            values[int(node)] += float(cache.values[time_position, sensor_position, pollutant])
            counts[int(node)] += 1
    available = counts > 0
    values[available] /= counts[available]
    return values.reshape(cache.grid_shape), available.reshape(cache.grid_shape)


def evaluate_method(cache, *, context_mask, target_mask, positions, method):
    target_rows, target_cols = np.nonzero(target_mask)
    absolute_error = squared_error = absolute_target = 0.0
    target_count = 0
    rows = []
    for position in positions:
        values, available = aggregate_nodes(cache, int(position))
        observed = available & context_mask
        target_available = available & target_mask
        interpolator = method(values[None, ...], observed[None, ...])
        prediction = np.asarray(
            interpolator(target_cols.astype(float), target_rows.astype(float), mode="points"),
            dtype=np.float64,
        )
        selected = target_available[target_rows, target_cols] & np.isfinite(prediction)
        truth = values[target_rows, target_cols][selected]
        prediction = prediction[selected]
        difference = prediction - truth
        absolute_error += np.abs(difference).sum()
        squared_error += np.square(difference).sum()
        absolute_target += np.abs(truth).sum()
        target_count += difference.size
        rows.append(
            {
                "timestamp": cache.time_index[int(position)].isoformat(),
                "mae": float(np.abs(difference).mean()),
                "mse": float(np.square(difference).mean()),
                "target_count": int(difference.size),
            }
        )
    return {
        "mae": float(absolute_error / target_count),
        "rmse": float(np.sqrt(squared_error / target_count)),
        "mse": float(squared_error / target_count),
        "wape": float(absolute_error / absolute_target),
        "target_count": int(target_count),
    }, rows


def main() -> None:
    args = parse_args()
    cache = load_aq_sensor_cache(args.cache)
    train, validation, test = split_sensor_nodes(
        cache,
        validation_nodes=args.validation_nodes,
        test_nodes=args.test_nodes,
        seed=args.seed,
    )
    timestamps = cache.time_index
    first = int(timestamps.searchsorted(pd.Timestamp(args.start)))
    last = int(timestamps.searchsorted(pd.Timestamp(args.end), side="right"))
    positions = np.arange(first, last, dtype=np.int64)
    if args.max_windows is not None:
        if args.max_windows <= 0:
            raise ValueError("max-windows must be positive")
        positions = positions[: args.max_windows]
    context = train | validation
    args.output.mkdir(parents=True, exist_ok=True)
    sensor_nodes = cache.sensor_node_indices
    test_sensor_ids = cache.sensor_ids[test.reshape(-1)[sensor_nodes]].astype(int).tolist()
    validation_sensor_ids = cache.sensor_ids[validation.reshape(-1)[sensor_nodes]].astype(int).tolist()

    results = {}
    for name, method in (("IDW", IdwInterpolator), ("Kriging", KrigingInterpolator)):
        metrics, rows = evaluate_method(
            cache,
            context_mask=context,
            target_mask=test,
            positions=positions,
            method=method,
        )
        results[name] = metrics
        pd.DataFrame(rows).to_csv(args.output / f"{name.lower()}-windows.csv", index=False)
    summary = {
        "period_start": timestamps[positions[0]].isoformat(),
        "period_end": timestamps[positions[-1]].isoformat(),
        "windows": int(len(positions)),
        "test_sensor_ids": test_sensor_ids,
        "validation_sensor_ids": validation_sensor_ids,
        "context_nodes": int(context.sum()),
        "test_nodes": int(test.sum()),
        **results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
