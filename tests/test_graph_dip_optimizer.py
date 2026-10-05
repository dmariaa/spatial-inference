from __future__ import annotations

import numpy as np
import pytest
import torch

from metraq_dip.trainer.graph_dip_optimizer import GraphDipOptimizer


def graph_split() -> dict:
    height = width = 3
    timesteps = 2
    train_mask = np.zeros((1, timesteps, height, width), dtype=bool)
    val_mask = np.zeros_like(train_mask)
    train_mask[:, :, 0, 0] = True
    train_mask[:, :, 2, 2] = True
    val_mask[:, :, 0, 2] = True
    observations = np.ones((1, timesteps, height, width), dtype=bool)
    train_data = np.zeros((1, timesteps, height, width), dtype=np.float32)
    val_data = np.zeros_like(train_data)
    train_data[:, :, 0, 0] = 1.0
    train_data[:, :, 2, 2] = 2.0
    val_data[:, :, 0, 2] = 1.5
    input_data = np.zeros((3, timesteps, height, width), dtype=np.float32)
    input_data[0] = 0.25
    input_data[1, :, 0, 0] = 1.0
    input_data[1, :, 2, 2] = 2.0
    input_data[2, :, 0, 0] = 1.0
    input_data[2, :, 2, 2] = 1.0
    return {
        "input_data": input_data,
        "train_data": train_data,
        "val_data": val_data,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "observation_mask": observations,
        "pollutants": [7],
    }


def graph_config(**overrides) -> dict:
    config = {
        "epochs": 2,
        "lr": 1e-3,
        "k_best_n": 1,
        "pollutants": [7],
        "normalize": False,
        "graph_dip": {
            "nearest_sensors": 1,
            "hidden_channels": 4,
            "attention_heads": 1,
            "patience": 2,
            "spatial_smoothness": 0.0,
        },
    }
    config.update(overrides)
    return config


def test_graph_dip_returns_dip_compatible_surface_and_artifacts():
    torch.manual_seed(0)
    optimizer = GraphDipOptimizer(
        configuration=graph_config(),
        split_data=graph_split(),
        device="cpu",
        disable_tqdm=True,
    )

    surface = optimizer.optimize()
    artifacts = optimizer.get_artifacts()

    assert surface.shape == (1, 3, 3)
    assert artifacts["member_surfaces"].shape == (1, 1, 3, 3)
    assert artifacts["member_artifacts"][0]["output_history"].shape == (2, 1, 3, 3)


def test_graph_dip_uses_only_train_nodes_as_sensor_edge_sources():
    optimizer = GraphDipOptimizer(
        configuration=graph_config(),
        split_data=graph_split(),
        device="cpu",
        disable_tqdm=True,
    )
    optimizer._prepare_split_tensors()

    sources = set(optimizer.graph_data.sensor_edge_index[0].cpu().tolist())
    assert sources == {0, 8}
    assert 2 not in sources


def test_graph_dip_requires_explicit_observation_mask():
    split = graph_split()
    split.pop("observation_mask")
    optimizer = GraphDipOptimizer(
        configuration=graph_config(),
        split_data=split,
        device="cpu",
        disable_tqdm=True,
    )

    with pytest.raises(ValueError, match="observation_mask"):
        optimizer.optimize()
