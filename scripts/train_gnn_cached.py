from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from metraq_dip.tools.random_tools import set_seed
from metraq_gnn.data import (
    SensorGraphWindowDataset,
    compute_training_normalization,
    load_aq_sensor_cache,
    split_sensor_nodes,
)
from metraq_gnn.model import SensorToGridGNN, SpatioTemporalGNN
from metraq_gnn.tracking import WandbLogger
from metraq_gnn.trainer import GNNTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GNN with a cached AQ time series")
    parser.add_argument("--cache", type=Path, default=Path("cache/metraq/no2-2010-2024"))
    parser.add_argument("--output", type=Path, default=Path("output/gnn/cached-development"))
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--time-features", action="store_true")
    parser.add_argument(
        "--architecture",
        choices=("local", "sensor-to-grid"),
        default="local",
    )
    parser.add_argument(
        "--local-refinement-layers",
        type=int,
        default=1,
        help="Number of local grid GAT layers after sensor-to-grid attention",
    )
    parser.add_argument("--target-fraction", type=float, default=0.25)
    parser.add_argument("--validation-nodes", type=int, default=4)
    parser.add_argument("--test-nodes", type=int, default=4)
    parser.add_argument(
        "--test-sensor-ids",
        type=int,
        nargs="+",
        help="Explicit sensor IDs reserved for TEST; overrides --test-nodes",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device")
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-validation-windows", type=int)
    parser.add_argument("--max-test-windows", type=int)
    parser.add_argument(
        "--test-windows-index",
        type=Path,
        help="NPZ index containing the exact time_windows to evaluate",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="metraq-gnn-test")
    parser.add_argument("--wandb-entity", default="dmariaa-team")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show epoch and batch progress bars (enabled by default)",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cache = load_aq_sensor_cache(args.cache)
    train_mask, validation_mask, test_mask = split_sensor_nodes(
        cache,
        validation_nodes=args.validation_nodes,
        test_nodes=args.test_nodes,
        seed=args.seed,
        test_sensor_ids=args.test_sensor_ids,
    )
    mean, std = compute_training_normalization(
        cache,
        train_node_mask=train_mask,
        start="2010-01-01 00:00",
        end="2022-12-31 23:00",
    )
    common = {
        "cache": cache,
        "hours": args.hours,
        "normalization_mean": mean,
        "normalization_std": std,
        "include_time_features": args.time_features,
        "seed": args.seed,
    }
    train_dataset = SensorGraphWindowDataset(
        **common,
        context_sensor_mask=train_mask,
        forbidden_context_mask=validation_mask | test_mask,
        target_fraction=args.target_fraction,
        split_start="2010-01-01 00:00",
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
    train_data = sampled(train_dataset, stride=args.stride, maximum=args.max_train_windows)
    validation_data = sampled(
        validation_dataset, stride=args.stride, maximum=args.max_validation_windows
    )
    if args.test_windows_index is not None:
        if args.max_test_windows is not None:
            raise ValueError("--max-test-windows cannot be combined with --test-windows-index")
        test_data = selected_test_windows(test_dataset, index_path=args.test_windows_index)
    else:
        test_data = sampled(test_dataset, stride=args.stride, maximum=args.max_test_windows)
    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = torch.cuda.is_available() if args.pin_memory is None else args.pin_memory
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    sample = train_data[0]
    model_options = {
        "input_channels": int(sample.x.shape[-1]),
        "output_channels": len(cache.magnitude_ids),
        "edge_channels": int(cache.graph.edge_attr.shape[-1]),
        "temporal_hidden_channels": 32,
        "graph_hidden_channels": 32,
        "attention_heads": 4,
        "dropout": 0.1,
    }
    if args.architecture == "sensor-to-grid":
        model = SensorToGridGNN(
            **model_options,
            local_refinement_layers=args.local_refinement_layers,
        )
    else:
        model = SpatioTemporalGNN(**model_options, graph_layers=2)
    config = {
        "protocol": "development",
        "architecture": args.architecture,
        "local_refinement_layers": args.local_refinement_layers,
        "train_period": "2010-2022",
        "validation_period": "2023",
        "test_period": "2024",
        "hours": args.hours,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "stride": args.stride,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "mixed_precision": args.mixed_precision,
        "time_features": args.time_features,
        "device_requested": args.device or "auto",
        "train_windows": len(train_data),
        "validation_windows": len(validation_data),
        "test_windows": len(test_data),
        "test_windows_index": (
            str(args.test_windows_index) if args.test_windows_index is not None else None
        ),
        "train_nodes": int(train_mask.sum()),
        "validation_nodes": int(validation_mask.sum()),
        "test_nodes": int(test_mask.sum()),
        "test_sensor_ids": (
            [int(sensor_id) for sensor_id in args.test_sensor_ids]
            if args.test_sensor_ids is not None
            else None
        ),
        "seed": args.seed,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "learning_rate": args.learning_rate,
    }
    logger = (
        WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            tags=["cached", "development-protocol"],
            config=config,
        )
        if args.wandb
        else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        trainer = GNNTrainer(
            model=model,
            learning_rate=args.learning_rate,
            optimization_loss="mae",
            patience=args.patience,
            device=args.device,
            mixed_precision=args.mixed_precision,
            logger=logger,
            show_progress=args.progress,
        )
        result = trainer.fit(
            train_loader,
            validation_loader,
            epochs=args.epochs,
            checkpoint_path=args.output / "best-model.pt",
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
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if logger is not None:
            logger.log_summary({key: value for key, value in summary.items() if isinstance(value, (int, float))})
        print(json.dumps(summary, indent=2))
    finally:
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    main()
