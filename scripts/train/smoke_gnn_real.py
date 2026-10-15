from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from metraq_dip.data.aq_backends import get_aq_backend_for_config
from metraq_dip.data.data import collect_data, collect_ensemble_data
from metraq_dip.tools.config_tools import load_session_config
from metraq_dip.tools.random_tools import set_seed
from metraq_gnn.data import (
    build_graph_window,
    build_grid_graph,
    split_training_sensor_masks,
)
from metraq_gnn.model import SpatioTemporalGNN
from metraq_gnn.tracking import WandbLogger
from metraq_gnn.trainer import GNNTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short GNN training test on real AQ data.")
    parser.add_argument(
        "--session",
        type=Path,
        default=Path("output/experiments/experiment_test"),
        help="Experiment session containing config.yaml and data.npz.",
    )
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-windows", type=int, default=1)
    parser.add_argument("--validation-windows", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Log the run to Weights & Biases.")
    parser.add_argument("--wandb-project", default="metraq-gnn-test")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = load_session_config(args.session / "config.yaml")
    config_values = config.model_dump()
    aq_backend = get_aq_backend_for_config(config_values)

    with np.load(args.session / "data.npz", allow_pickle=True) as manifest:
        test_sensors = [int(sensor) for sensor in manifest["test_sensors"][0]]
        candidate_windows = [pd.Timestamp(value) for value in manifest["time_windows"]]

    if args.train_windows <= 0 or args.validation_windows <= 0 or args.batch_size <= 0:
        raise ValueError("train-windows, validation-windows, and batch-size must be positive")
    requested_windows = args.train_windows + args.validation_windows
    selected_windows: list[pd.Timestamp] = []
    minimum_separation = pd.Timedelta(hours=config.hours)
    for candidate in candidate_windows[args.window_index:]:
        if all(abs(candidate - selected) >= minimum_separation for selected in selected_windows):
            selected_windows.append(candidate)
        if len(selected_windows) == requested_windows:
            break
    if len(selected_windows) != requested_windows:
        raise ValueError(
            f"Could select only {len(selected_windows)} non-overlapping windows, "
            f"but {requested_windows} were requested"
        )

    train_end_times = selected_windows[: args.train_windows]
    validation_end_times = selected_windows[args.train_windows :]
    end_time = train_end_times[0].to_pydatetime()

    def collect_window(window_end: pd.Timestamp):
        window_end = pd.Timestamp(window_end).to_pydatetime()
        window_start = window_end - pd.Timedelta(hours=config.hours - 1)
        return collect_data(
            start_date=window_start,
            end_date=window_end,
            add_meteo=False,
            add_time_channels=True,
            add_coordinates=True,
            add_traffic_data=False,
            pollutants=list(config.pollutants),
            test_sensors=test_sensors,
            normalize=True,
            aq_backend=aq_backend,
        )

    static_data = collect_window(end_time)
    split_data = collect_ensemble_data(
        data=static_data,
        number_of_noise_channels=0,
        number_of_val_sensors=4,
        add_distance_to_sensors=False,
        normalize=True,
        aq_backend=aq_backend,
    )

    graph = build_grid_graph(static_data["grid_ctx"])
    validation_mask = np.asarray(split_data["val_mask"], dtype=bool)
    test_mask = np.asarray(split_data["test_mask"], dtype=bool)
    forbidden_train = validation_mask | test_mask
    train_samples = []
    train_context_counts = []
    train_target_counts = []
    for index, window_end in enumerate(train_end_times):
        window_data = static_data if index == 0 else collect_window(window_end)
        pollutant_data = np.asarray(window_data["pollutant_data"], dtype=np.float32)
        final_availability = pollutant_data[1::2, -1].astype(bool)
        train_context, train_target = split_training_sensor_masks(
            split_data["train_mask"],
            final_availability,
            target_fraction=0.25,
            generator=torch.Generator().manual_seed(args.seed + index),
        )
        train_samples.append(
            build_graph_window(
                graph=graph,
                pollutant_data=pollutant_data,
                context_sensor_mask=train_context,
                target_sensor_mask=train_target,
                forbidden_context_mask=forbidden_train,
                extra_features=window_data.get("static_input_prefix"),
            )
        )
        train_context_counts.append(int(train_context.sum()))
        train_target_counts.append(int(train_target.sum()))

    validation_samples = []
    for window_end in validation_end_times:
        window_data = collect_window(window_end)
        validation_samples.append(
            build_graph_window(
                graph=graph,
                pollutant_data=np.asarray(window_data["pollutant_data"], dtype=np.float32),
                context_sensor_mask=split_data["train_mask"],
                target_sensor_mask=validation_mask,
                forbidden_context_mask=forbidden_train,
                extra_features=window_data.get("static_input_prefix"),
            )
        )

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_samples,
        batch_size=args.batch_size,
        shuffle=False,
    )
    train_sample = train_samples[0]
    model = SpatioTemporalGNN(
        input_channels=int(train_sample.x.shape[-1]),
        output_channels=len(config.pollutants),
        edge_channels=int(graph.edge_attr.shape[-1]),
        temporal_hidden_channels=16,
        graph_hidden_channels=16,
        graph_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    run_config = {
        "dataset": config.aq_dataset,
        "pollutants": list(config.pollutants),
        "year": end_time.year,
        "window_index": args.window_index,
        "hours": config.hours,
        "batch_size": args.batch_size,
        "train_windows": len(train_samples),
        "validation_windows": len(validation_samples),
        "epochs": args.epochs,
        "seed": args.seed,
        "temporal_hidden_channels": 16,
        "graph_hidden_channels": 16,
        "graph_layers": 1,
        "attention_heads": 2,
        "learning_rate": 1e-3,
        "optimization_loss": "mae",
        "test_sensor_count": len(test_sensors),
    }
    logger = (
        WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            tags=["smoke-test", "real-data"],
            config=run_config,
        )
        if args.wandb
        else None
    )
    try:
        trainer = GNNTrainer(
            model=model,
            learning_rate=1e-3,
            optimization_loss="mae",
            patience=1,
            device="cpu",
            logger=logger,
        )
        result = trainer.fit(
            train_loader,
            validation_loader,
            epochs=args.epochs,
        )
    finally:
        if logger is not None:
            logger.finish()

    summary = {
        "first_window": end_time.isoformat(),
        "grid_shape": list(static_data["grid_ctx"]["grid"].shape),
        "nodes": int(graph.num_nodes),
        "edges": int(graph.edge_index.shape[1]),
        "input_shape": list(train_sample.x.shape),
        "train_windows": len(train_samples),
        "validation_windows": len(validation_samples),
        "batch_size": args.batch_size,
        "train_batches_per_epoch": len(train_loader),
        "validation_batches_per_epoch": len(validation_loader),
        "train_context_cells_min": min(train_context_counts),
        "train_context_cells_max": max(train_context_counts),
        "train_target_cells_min": min(train_target_counts),
        "train_target_cells_max": max(train_target_counts),
        "validation_cells": int(validation_mask.sum()),
        "test_cells": int(test_mask.sum()),
        "epochs_completed": result.epochs_completed,
        "train_mae": result.history[-1]["train_mae"],
        "validation_mae": result.history[-1]["validation_mae"],
        "test_sensors": test_sensors,
        "wandb_url": logger.url if logger is not None else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
