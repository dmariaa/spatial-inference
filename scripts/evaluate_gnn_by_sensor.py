from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from metraq_gnn.data import SensorGraphWindowDataset, load_aq_sensor_cache, split_sensor_nodes
from metraq_gnn.evaluation import (
    collect_sensor_predictions,
    summarize_all_predictions,
    summarize_sensor_predictions,
)
from metraq_gnn.model import SensorToGridGNN, SpatioTemporalGNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GNN TEST errors by sensor")
    parser.add_argument("--cache", type=Path, default=Path("cache/metraq/no2-2010-2024"))
    parser.add_argument("--run", type=Path, required=True, help="Run directory with summary and checkpoint")
    parser.add_argument("--test-windows-index", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def selected_windows(dataset: SensorGraphWindowDataset, index_path: Path) -> Subset:
    with np.load(index_path, allow_pickle=True) as index:
        requested = pd.DatetimeIndex(pd.to_datetime(index["time_windows"].tolist()))
    available = {
        int(timestamp.value): position
        for position, timestamp in enumerate(dataset.window_end_times)
    }
    missing = requested[~requested.isin(dataset.window_end_times)]
    if len(missing):
        raise ValueError(f"{len(missing)} requested test timestamps are unavailable")
    return Subset(dataset, [available[int(timestamp.value)] for timestamp in requested])


def build_model(summary: dict[str, object], sample, edge_channels: int, output_channels: int):
    options = {
        "input_channels": int(sample.x.shape[-1]),
        "output_channels": output_channels,
        "edge_channels": edge_channels,
        "temporal_hidden_channels": 32,
        "graph_hidden_channels": 32,
        "attention_heads": 4,
        "dropout": 0.1,
    }
    if summary["architecture"] == "sensor-to-grid":
        return SensorToGridGNN(
            **options,
            local_refinement_layers=int(summary["local_refinement_layers"]),
        )
    return SpatioTemporalGNN(**options, graph_layers=2)


def main() -> None:
    args = parse_args()
    summary_path = args.run / "summary.json"
    checkpoint_path = args.run / "best-model.pt"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cache = load_aq_sensor_cache(args.cache)
    train_mask, validation_mask, test_mask = split_sensor_nodes(
        cache,
        validation_nodes=int(summary["validation_nodes"]),
        test_nodes=int(summary["test_nodes"]),
        seed=int(summary["seed"]),
        test_sensor_ids=summary["test_sensor_ids"],
    )
    dataset = SensorGraphWindowDataset(
        cache=cache,
        hours=int(summary["hours"]),
        normalization_mean=np.asarray(summary["normalization_mean"], dtype=np.float32),
        normalization_std=np.asarray(summary["normalization_std"], dtype=np.float32),
        include_time_features=bool(summary["time_features"]),
        seed=int(summary["seed"]),
        context_sensor_mask=train_mask | validation_mask,
        forbidden_context_mask=test_mask,
        target_sensor_mask=test_mask,
        split_start="2024-01-01 00:00",
        split_end="2024-12-31 23:00",
    )
    index_path = args.test_windows_index or Path(str(summary["test_windows_index"]))
    test_data = selected_windows(dataset, index_path)
    loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = build_model(
        summary,
        test_data[0],
        edge_channels=int(cache.graph.edge_attr.shape[-1]),
        output_channels=len(cache.magnitude_ids),
    )
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    predictions = collect_sensor_predictions(
        model,
        loader,
        mean=np.asarray(summary["normalization_mean"], dtype=np.float32),
        std=np.asarray(summary["normalization_std"], dtype=np.float32),
        device=args.device,
        sensor_ids=cache.sensor_ids,
        sensor_node_indices=cache.sensor_node_indices,
        node_positions=cache.graph.pos.cpu().numpy(),
        magnitude_ids=cache.magnitude_ids,
    )
    sensor_metrics = summarize_sensor_predictions(predictions)
    aggregate = summarize_all_predictions(predictions)
    expected = {
        key: summary[key]
        for key in aggregate
    }
    for key, value in aggregate.items():
        if not np.isclose(value, expected[key], rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"aggregate mismatch for {key}: {value} != {expected[key]}")

    predictions.to_csv(args.run / "test_predictions_by_sensor.csv", index=False)
    sensor_metrics.to_csv(args.run / "test_metrics_by_sensor.csv", index=False)
    print(sensor_metrics.to_string(index=False))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
