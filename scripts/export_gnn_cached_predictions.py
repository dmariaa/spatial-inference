from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from metraq_gnn.data import (
    SensorGraphWindowDataset,
    load_aq_sensor_cache,
    load_traffic_grid_cache,
    split_sensor_nodes,
)
from metraq_gnn.evaluation import collect_sensor_predictions, summarize_all_predictions
from metraq_gnn.model import SensorToGridGNN, SpatioTemporalGNN


def selected_test_windows(dataset, *, index_path: Path):
    with np.load(index_path, allow_pickle=True) as index:
        if "time_windows" not in index.files:
            raise ValueError("test windows index must contain 'time_windows'")
        requested = pd.DatetimeIndex(pd.to_datetime(index["time_windows"].tolist()))
    if requested.has_duplicates:
        raise ValueError("test windows index must not contain duplicate timestamps")
    available = {
        int(timestamp.value): position
        for position, timestamp in enumerate(dataset.window_end_times)
    }
    missing = [timestamp for timestamp in requested if int(timestamp.value) not in available]
    if missing:
        raise ValueError(f"{len(missing)} requested test timestamps are unavailable")
    return torch.utils.data.Subset(
        dataset,
        [available[int(timestamp.value)] for timestamp in requested],
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
@click.option("--run-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--cache", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--traffic-cache", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--test-windows-index", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output", type=click.Path(path_type=Path))
@click.option("--batch-size", type=click.IntRange(min=1), default=64)
@click.option("--num-workers", type=click.IntRange(min=0), default=0)
@click.option("--pin-memory/--no-pin-memory", default=None)
@click.option("--device")
def main(
    run_dir: Path,
    cache: Path,
    traffic_cache: Path,
    test_windows_index: Path,
    output: Path | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool | None,
    device: str | None,
) -> None:
    """Export physical-unit predictions from one trained cached GNN run."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "best-model.pt"
    if not checkpoint_path.exists():
        raise click.ClickException(f"missing checkpoint: {checkpoint_path}")
    test_sensor_ids = summary.get("test_sensor_ids")
    if not test_sensor_ids:
        raise click.ClickException("summary.json must contain explicit test_sensor_ids")

    aq_cache = load_aq_sensor_cache(cache)
    train_mask, validation_mask, test_mask = split_sensor_nodes(
        aq_cache,
        validation_nodes=int(summary["validation_nodes"]),
        test_nodes=int(summary["test_nodes"]),
        seed=int(summary["seed"]),
        test_sensor_ids=test_sensor_ids,
    )
    loaded_traffic = load_traffic_grid_cache(traffic_cache)
    dataset = SensorGraphWindowDataset(
        cache=aq_cache,
        hours=int(summary["hours"]),
        normalization_mean=np.asarray(summary["normalization_mean"], dtype=np.float32),
        normalization_std=np.asarray(summary["normalization_std"], dtype=np.float32),
        traffic_cache=loaded_traffic,
        traffic_normalization_mean=float(summary["traffic_normalization_mean"]),
        traffic_normalization_std=float(summary["traffic_normalization_std"]),
        include_time_features=bool(summary["time_features"]),
        seed=int(summary["seed"]),
        context_sensor_mask=train_mask | validation_mask,
        forbidden_context_mask=test_mask,
        target_sensor_mask=test_mask,
        split_start="2024-01-01 00:00",
        split_end="2024-12-31 23:00",
    )
    test_data = selected_test_windows(dataset, index_path=test_windows_index)
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    sample = test_data[0]
    model_options = {
        "input_channels": int(sample.x.shape[-1]),
        "output_channels": len(aq_cache.magnitude_ids),
        "edge_channels": int(aq_cache.graph.edge_attr.shape[-1]),
        "temporal_hidden_channels": 32,
        "graph_hidden_channels": 32,
        "attention_heads": 4,
        "dropout": 0.1,
    }
    if summary["architecture"] == "sensor-to-grid":
        model = SensorToGridGNN(
            **model_options,
            local_refinement_layers=int(summary["local_refinement_layers"]),
        )
    else:
        model = SpatioTemporalGNN(**model_options, graph_layers=2)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)

    predictions = collect_sensor_predictions(
        model,
        loader,
        mean=np.asarray(summary["normalization_mean"], dtype=np.float32),
        std=np.asarray(summary["normalization_std"], dtype=np.float32),
        device=resolved_device,
        sensor_ids=aq_cache.sensor_ids,
        sensor_node_indices=aq_cache.sensor_node_indices,
        node_positions=aq_cache.graph.pos.numpy(),
        magnitude_ids=aq_cache.magnitude_ids,
    )
    predictions.insert(0, "sensor_group", "-".join(str(value) for value in sorted(test_sensor_ids)))
    predictions.insert(4, "method", "GNN_TRAFFIC")
    destination = output or run_dir / "predictions.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    metrics = summarize_all_predictions(predictions)
    click.echo(json.dumps({"output": str(destination), **metrics}, indent=2))


if __name__ == "__main__":
    main()
