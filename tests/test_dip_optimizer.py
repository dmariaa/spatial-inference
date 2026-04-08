from __future__ import annotations

import numpy as np
import pytest
import torch

from metraq_dip.trainer.dip_optimizer import DipOptimizer, select_surface_from_validation


def test_select_surface_from_validation_uses_all_available_steps_when_k_best_matches_epoch_count():
    output_history = np.array(
        [
            [[[1.0]]],
            [[[3.0]]],
            [[[5.0]]],
        ],
        dtype=np.float32,
    )
    val_loss_history = np.array([2.0, 1.0, 3.0], dtype=np.float32)

    surface, best_indices = select_surface_from_validation(
        output_history=output_history,
        val_loss_history=val_loss_history,
        k_best_n=3,
    )

    assert surface.shape == (1, 1, 1)
    assert surface.item() == 3.0
    assert best_indices.tolist() == [1, 0, 2]


def test_get_artifacts_requires_optimization():
    optimizer = DipOptimizer(
        configuration={
            "epochs": 1,
            "lr": 1e-3,
            "model": {
                "base_channels": 1,
                "levels": 1,
                "preserve_time": True,
                "learned_upsampling": False,
            },
        },
        split_data={
            "input_data": np.zeros((1, 2, 4, 4), dtype=np.float32),
            "train_data": np.zeros((1, 2, 4, 4), dtype=np.float32),
            "val_data": np.zeros((1, 2, 4, 4), dtype=np.float32),
            "train_mask": np.ones((1, 2, 4, 4), dtype=bool),
            "val_mask": np.ones((1, 2, 4, 4), dtype=bool),
        },
        disable_tqdm=True,
    )

    with pytest.raises(RuntimeError):
        optimizer.get_artifacts()


def test_dip_optimizer_returns_surface_and_artifacts():
    torch.manual_seed(0)

    split_data = {
        "input_data": np.ones((1, 2, 4, 4), dtype=np.float32),
        "train_data": np.ones((1, 2, 4, 4), dtype=np.float32),
        "val_data": np.ones((1, 2, 4, 4), dtype=np.float32),
        "train_mask": np.ones((1, 2, 4, 4), dtype=bool),
        "val_mask": np.ones((1, 2, 4, 4), dtype=bool),
        "normalization_stats": {7: (12.5, 3.5)},
        "pollutants": [7],
    }
    configuration = {
        "epochs": 2,
        "lr": 1e-3,
        "k_best_n": 1,
        "normalize": True,
        "pollutants": [7],
        "model": {
            "architecture": "autoencoder",
            "base_channels": 1,
            "levels": 1,
            "preserve_time": True,
            "learned_upsampling": False,
            "skip_connections": False,
        },
    }

    optimizer = DipOptimizer(
        configuration=configuration,
        split_data=split_data,
        disable_tqdm=True,
        device="cpu",
    )

    surface = optimizer.optimize()
    artifacts = optimizer.get_artifacts()

    assert surface.shape == (1, 4, 4)
    assert artifacts["surface"].shape == (1, 4, 4)
    assert np.allclose(surface, artifacts["surface"])
    assert artifacts["surface_model_space"].shape == (1, 4, 4)
    assert artifacts["output_history"].shape == (2, 1, 4, 4)
    assert artifacts["train_l1_history"].shape == (2,)
    assert artifacts["val_l1_history"].shape == (2,)
    assert artifacts["selected_epoch_indices"].shape == (1,)
    assert artifacts["normalization_stats"] == {7: (12.5, 3.5)}


def test_dip_optimizer_returns_surface_in_real_values_when_normalized(monkeypatch):
    optimizer = DipOptimizer(
        configuration={
            "epochs": 1,
            "lr": 1e-3,
            "k_best_n": 1,
            "normalize": True,
            "pollutants": [7],
            "model": {
                "base_channels": 1,
                "levels": 1,
                "preserve_time": True,
                "learned_upsampling": False,
            },
        },
        split_data={
            "input_data": np.zeros((1, 1, 2, 2), dtype=np.float32),
            "train_data": np.zeros((1, 1, 2, 2), dtype=np.float32),
            "val_data": np.zeros((1, 1, 2, 2), dtype=np.float32),
            "train_mask": np.ones((1, 1, 2, 2), dtype=bool),
            "val_mask": np.ones((1, 1, 2, 2), dtype=bool),
            "pollutants": [7],
            "normalization_stats": {7: (10.0, 2.0)},
        },
        disable_tqdm=True,
        device="cpu",
    )

    monkeypatch.setattr(optimizer, "_get_model", lambda: object())
    monkeypatch.setattr(optimizer, "_get_optimizer", lambda: object())

    def fake_run_epoch(*, step: int):
        optimizer.artifacts["output_history"][step] = torch.zeros((1, 2, 2), dtype=torch.float32)
        optimizer.artifacts["train_l1_history"][step] = 0.0
        optimizer.artifacts["train_mse_history"][step] = 0.0
        optimizer.artifacts["val_l1_history"][step] = 0.0
        optimizer.artifacts["val_mse_history"][step] = 0.0
        return {"train_mae": 0.0, "val_mae": 0.0}

    monkeypatch.setattr(optimizer, "_run_epoch", fake_run_epoch)

    surface = optimizer.optimize()
    artifacts = optimizer.get_artifacts()

    assert np.allclose(surface, np.full((1, 2, 2), 10.0, dtype=np.float32))
    assert np.allclose(artifacts["surface"], np.full((1, 2, 2), 10.0, dtype=np.float32))
    assert np.allclose(artifacts["surface_model_space"], np.zeros((1, 2, 2), dtype=np.float32))


def test_dip_optimizer_expands_broadcastable_masks(monkeypatch):
    optimizer = DipOptimizer(
        configuration={
            "epochs": 1,
            "lr": 1e-3,
            "k_best_n": 1,
            "normalize": False,
            "pollutants": [7],
            "model": {
                "base_channels": 1,
                "levels": 1,
                "preserve_time": True,
                "learned_upsampling": False,
            },
        },
        split_data={
            "input_data": np.zeros((1, 24, 2, 2), dtype=np.float32),
            "train_data": np.zeros((1, 24, 2, 2), dtype=np.float32),
            "val_data": np.zeros((1, 24, 2, 2), dtype=np.float32),
            "train_mask": np.ones((1, 1, 2, 2), dtype=bool),
            "val_mask": np.ones((1, 1, 2, 2), dtype=bool),
            "pollutants": [7],
        },
        disable_tqdm=True,
        device="cpu",
    )

    monkeypatch.setattr(optimizer, "_get_model", lambda: object())
    monkeypatch.setattr(optimizer, "_get_optimizer", lambda: object())

    def fake_run_epoch(*, step: int):
        optimizer.artifacts["output_history"][step] = torch.zeros((1, 2, 2), dtype=torch.float32)
        optimizer.artifacts["train_l1_history"][step] = 0.0
        optimizer.artifacts["train_mse_history"][step] = 0.0
        optimizer.artifacts["val_l1_history"][step] = 0.0
        optimizer.artifacts["val_mse_history"][step] = 0.0
        return {"train_mae": 0.0, "val_mae": 0.0}

    monkeypatch.setattr(optimizer, "_run_epoch", fake_run_epoch)

    optimizer.optimize()
    artifacts = optimizer.get_artifacts()

    assert artifacts["train_mask"].shape == (1, 24, 2, 2)
    assert artifacts["val_mask"].shape == (1, 24, 2, 2)
