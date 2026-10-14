from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from metraq_dip.tools.random_tools import set_seed
from metraq_gnn.data import (
    SensorGraphWindowDataset,
    compute_training_normalization,
    compute_traffic_training_normalization,
    load_aq_sensor_cache,
    load_traffic_grid_cache,
    split_sensor_nodes,
)
from metraq_gnn.model import SensorToGridGNN, SpatioTemporalGNN
from metraq_gnn.tracking import WandbLogger
from metraq_gnn.trainer import GNNTrainer


def sampled(dataset, *, stride: int, maximum: int | None):
    if stride <= 0:
        raise ValueError("stride must be positive")
    indices = range(0, len(dataset), stride)
    if maximum is not None and maximum <= 0:
        raise ValueError("window limits must be positive")
    if maximum is not None:
        indices = list(indices)[:maximum]
    return Subset(dataset, indices)


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
    return Subset(dataset, [available[int(timestamp.value)] for timestamp in requested])


def evaluate_physical_units(model, loader, *, mean, std, device) -> dict[str, float]:
    model.eval()
    absolute_error = squared_error = absolute_target = 0.0
    count = 0
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch) * std_tensor + mean_tensor
            target = batch.y * std_tensor + mean_tensor
            difference = prediction[batch.target_mask] - target[batch.target_mask]
            absolute_error += float(difference.abs().sum())
            squared_error += float(difference.square().sum())
            absolute_target += float(target[batch.target_mask].abs().sum())
            count += int(difference.numel())
    return {
        "test_mae_physical": absolute_error / count,
        "test_rmse_physical": (squared_error / count) ** 0.5,
        "test_wape": absolute_error / absolute_target if absolute_target else float("nan"),
        "test_target_count": count,
    }


@click.command(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
@click.option("--cache", type=click.Path(path_type=Path), default=Path("cache/metraq/no2-2010-2024"))
@click.option(
    "--traffic-cache",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Optional hourly traffic grid cache; adds intensity and availability channels.",
)
@click.option("--output", type=click.Path(path_type=Path), default=Path("output/gnn/cached-development"))
@click.option("--hours", type=click.IntRange(min=1), default=24)
@click.option("--train-start", default="2010-01-01 00:00", help="Inclusive start timestamp for the TRAIN period.")
@click.option("--epochs", type=click.IntRange(min=1), default=100)
@click.option("--batch-size", type=click.IntRange(min=1), default=16)
@click.option("--stride", type=click.IntRange(min=1), default=1)
@click.option("--num-workers", type=click.IntRange(min=0), default=0)
@click.option("--pin-memory/--no-pin-memory", default=None, help="Override automatic CUDA-based pin-memory selection.")
@click.option("--mixed-precision", is_flag=True)
@click.option("--time-features", is_flag=True)
@click.option("--architecture", type=click.Choice(("local", "sensor-to-grid")), default="local")
@click.option(
    "--local-refinement-layers",
    type=click.IntRange(min=0),
    default=1,
    help="Number of local grid GAT layers after sensor-to-grid attention.",
)
@click.option("--target-fraction", type=click.FloatRange(min=0.0, max=1.0, min_open=True, max_open=True), default=0.25)
@click.option("--validation-nodes", type=click.IntRange(min=1), default=4)
@click.option("--test-nodes", type=click.IntRange(min=1), default=4)
@click.option(
    "--test-sensor-ids",
    type=int,
    multiple=True,
    help="Explicit sensor ID reserved for TEST; repeat for multiple IDs. Overrides --test-nodes.",
)
@click.option("--seed", type=int, default=42)
@click.option("--learning-rate", type=click.FloatRange(min=0.0, min_open=True), default=1e-3)
@click.option("--patience", type=click.IntRange(min=1), default=20)
@click.option("--device")
@click.option("--max-train-windows", type=click.IntRange(min=1))
@click.option("--max-validation-windows", type=click.IntRange(min=1))
@click.option("--max-test-windows", type=click.IntRange(min=1))
@click.option(
    "--test-windows-index",
    type=click.Path(path_type=Path),
    help="NPZ index containing the exact time_windows to evaluate.",
)
@click.option("--wandb", is_flag=True)
@click.option("--wandb-project", default="metraq-gnn-test")
@click.option("--wandb-entity", default="dmariaa-team")
@click.option("--wandb-name")
@click.option("--wandb-mode", type=click.Choice(("online", "offline", "disabled")), default="online")
@click.option("--progress/--no-progress", default=True, help="Show epoch and batch progress bars.")
def main(
    cache: Path,
    traffic_cache: Path | None,
    output: Path,
    hours: int,
    train_start: str,
    epochs: int,
    batch_size: int,
    stride: int,
    num_workers: int,
    pin_memory: bool | None,
    mixed_precision: bool,
    time_features: bool,
    architecture: str,
    local_refinement_layers: int,
    target_fraction: float,
    validation_nodes: int,
    test_nodes: int,
    test_sensor_ids: tuple[int, ...],
    seed: int,
    learning_rate: float,
    patience: int,
    device: str | None,
    max_train_windows: int | None,
    max_validation_windows: int | None,
    max_test_windows: int | None,
    test_windows_index: Path | None,
    wandb: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_name: str | None,
    wandb_mode: str,
    progress: bool,
) -> None:
    """Train the GNN with a cached AQ time series."""
    explicit_test_sensor_ids = list(test_sensor_ids) or None
    set_seed(seed)
    aq_cache = load_aq_sensor_cache(cache)
    train_mask, validation_mask, test_mask = split_sensor_nodes(
        aq_cache,
        validation_nodes=validation_nodes,
        test_nodes=test_nodes,
        seed=seed,
        test_sensor_ids=explicit_test_sensor_ids,
    )
    mean, std = compute_training_normalization(
        aq_cache,
        train_node_mask=train_mask,
        start=train_start,
        end="2022-12-31 23:00",
    )
    loaded_traffic_cache = None
    traffic_mean = traffic_std = None
    if traffic_cache is not None:
        loaded_traffic_cache = load_traffic_grid_cache(traffic_cache)
        traffic_mean, traffic_std = compute_traffic_training_normalization(
            loaded_traffic_cache,
            start=train_start,
            end="2022-12-31 23:00",
        )
    common = {
        "cache": aq_cache,
        "hours": hours,
        "normalization_mean": mean,
        "normalization_std": std,
        "traffic_cache": loaded_traffic_cache,
        "traffic_normalization_mean": traffic_mean,
        "traffic_normalization_std": traffic_std,
        "include_time_features": time_features,
        "seed": seed,
    }
    train_dataset = SensorGraphWindowDataset(
        **common,
        context_sensor_mask=train_mask,
        forbidden_context_mask=validation_mask | test_mask,
        target_fraction=target_fraction,
        split_start=train_start,
        split_end="2022-12-31 23:00",
    )
    validation_dataset = SensorGraphWindowDataset(
        **common,
        context_sensor_mask=train_mask,
        forbidden_context_mask=validation_mask | test_mask,
        target_sensor_mask=validation_mask,
        split_start="2023-01-01 00:00",
        split_end="2023-12-31 23:00",
    )
    test_dataset = SensorGraphWindowDataset(
        **common,
        context_sensor_mask=train_mask | validation_mask,
        forbidden_context_mask=test_mask,
        target_sensor_mask=test_mask,
        split_start="2024-01-01 00:00",
        split_end="2024-12-31 23:00",
    )
    train_data = sampled(train_dataset, stride=stride, maximum=max_train_windows)
    validation_data = sampled(
        validation_dataset, stride=stride, maximum=max_validation_windows
    )
    if test_windows_index is not None:
        if max_test_windows is not None:
            raise click.UsageError("--max-test-windows cannot be combined with --test-windows-index")
        test_data = selected_test_windows(test_dataset, index_path=test_windows_index)
    else:
        test_data = sampled(test_dataset, stride=stride, maximum=max_test_windows)
    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    sample = train_data[0]
    model_options = {
        "input_channels": int(sample.x.shape[-1]),
        "output_channels": len(aq_cache.magnitude_ids),
        "edge_channels": int(aq_cache.graph.edge_attr.shape[-1]),
        "temporal_hidden_channels": 32,
        "graph_hidden_channels": 32,
        "attention_heads": 4,
        "dropout": 0.1,
    }
    if architecture == "sensor-to-grid":
        model = SensorToGridGNN(
            **model_options,
            local_refinement_layers=local_refinement_layers,
        )
    else:
        model = SpatioTemporalGNN(**model_options, graph_layers=2)
    config = {
        "protocol": "development",
        "architecture": architecture,
        "local_refinement_layers": local_refinement_layers,
        "train_start": str(pd.Timestamp(train_start)),
        "train_period": f"{pd.Timestamp(train_start)} to 2022-12-31 23:00:00",
        "validation_period": "2023",
        "test_period": "2024",
        "hours": hours,
        "epochs": epochs,
        "batch_size": batch_size,
        "stride": stride,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "mixed_precision": mixed_precision,
        "time_features": time_features,
        "traffic_cache": str(traffic_cache) if traffic_cache is not None else None,
        "traffic_normalization_mean": traffic_mean,
        "traffic_normalization_std": traffic_std,
        "device_requested": device or "auto",
        "train_windows": len(train_data),
        "validation_windows": len(validation_data),
        "test_windows": len(test_data),
        "test_windows_index": (
            str(test_windows_index) if test_windows_index is not None else None
        ),
        "train_nodes": int(train_mask.sum()),
        "validation_nodes": int(validation_mask.sum()),
        "test_nodes": int(test_mask.sum()),
        "test_sensor_ids": (
            [int(sensor_id) for sensor_id in explicit_test_sensor_ids]
            if explicit_test_sensor_ids is not None else None
        ),
        "seed": seed,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "learning_rate": learning_rate,
    }
    logger = (
        WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            mode=wandb_mode,
            tags=["cached", "development-protocol"],
            config=config,
        )
        if wandb
        else None
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        trainer = GNNTrainer(
            model=model,
            learning_rate=learning_rate,
            optimization_loss="mae",
            patience=patience,
            device=device,
            mixed_precision=mixed_precision,
            logger=logger,
            show_progress=progress,
        )
        result = trainer.fit(
            train_loader,
            validation_loader,
            epochs=epochs,
            checkpoint_path=output / "best-model.pt",
        )
        test_normalized = trainer.evaluate(test_loader)
        test_physical = evaluate_physical_units(
            model,
            test_loader,
            mean=mean,
            std=std,
            device=trainer.device,
        )
        summary = config | {
            "best_epoch": result.best_epoch,
            "best_validation_mae_normalized": result.best_validation_loss,
            **{f"test_{key}_normalized": value for key, value in test_normalized.items()},
            **test_physical,
            "wandb_url": logger.url if logger is not None else None,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if logger is not None:
            logger.log_summary({key: value for key, value in summary.items() if isinstance(value, (int, float))})
        print(json.dumps(summary, indent=2))
    finally:
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    main()
