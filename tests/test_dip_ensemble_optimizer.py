from __future__ import annotations

import numpy as np
import pytest

from metraq_dip.trainer.dip_ensemble_optimizer import DipEnsembleOptimizer, reduce_surface_ensemble
from metraq_dip.trainer.dip_optimizer import DipOptimizer
from metraq_dip.trainer.optimizer_protocol import SurfaceOptimizer


def test_reduce_surface_ensemble_supports_mean():
    surface = reduce_surface_ensemble(
        surfaces=[
            np.array([[[1.0]]], dtype=np.float32),
            np.array([[[3.0]]], dtype=np.float32),
            np.array([[[5.0]]], dtype=np.float32),
        ],
        reduction="mean",
    )

    assert surface.shape == (1, 1, 1)
    assert surface.item() == 3.0


def test_optimizers_follow_surface_optimizer_protocol():
    split_data = {
        "input_data": np.zeros((1, 2, 4, 4), dtype=np.float32),
        "train_data": np.zeros((1, 2, 4, 4), dtype=np.float32),
        "val_data": np.zeros((1, 2, 4, 4), dtype=np.float32),
        "train_mask": np.ones((1, 2, 4, 4), dtype=bool),
        "val_mask": np.ones((1, 2, 4, 4), dtype=bool),
    }
    configuration = {
        "aq_dataset": "metraq",
        "aq_backend": "files",
        "epochs": 1,
        "lr": 1e-3,
        "validation_sensors": 1,
        "model": {
            "base_channels": 1,
            "levels": 1,
            "preserve_time": True,
            "learned_upsampling": False,
        },
    }

    optimizer = DipOptimizer(
        configuration=configuration,
        split_data=split_data,
        disable_tqdm=True,
        device="cpu",
    )
    ensemble_optimizer = DipEnsembleOptimizer(
        configuration=configuration | {"ensemble_size": 1},
        static_data={"seed": "unused"},
        disable_tqdm=True,
        optimizer_factory=lambda **kwargs: optimizer,
    )

    assert isinstance(optimizer, SurfaceOptimizer)
    assert isinstance(ensemble_optimizer, SurfaceOptimizer)


def test_dip_ensemble_optimizer_returns_reduced_surface_and_artifacts(monkeypatch):
    collected_values = iter([1.0, 3.0, 5.0])

    def fake_collect_ensemble_data(**kwargs):
        return {"surface_value": next(collected_values), "normalization_stats": {7: (12.5, 3.5)}}

    class DummyOptimizer:
        def __init__(self, *, configuration, split_data, device=None, disable_tqdm=False):
            self.surface = np.full((1, 2, 2), split_data["surface_value"], dtype=np.float32)
            self.artifacts = {
                "surface": np.array(self.surface, copy=True),
                "selected_epoch_indices": np.array([0], dtype=np.int64),
                "normalization_stats": split_data["normalization_stats"],
            }

        def optimize(self) -> np.ndarray:
            return np.array(self.surface, copy=True)

        def get_artifacts(self) -> dict[str, object]:
            return dict(self.artifacts)

    monkeypatch.setattr(
        "metraq_dip.trainer.dip_ensemble_optimizer.collect_ensemble_data",
        fake_collect_ensemble_data,
    )

    optimizer = DipEnsembleOptimizer(
        configuration={
            "aq_dataset": "metraq",
            "aq_backend": "files",
            "ensemble_size": 3,
            "validation_sensors": 1,
            "add_distance_to_sensors": False,
            "normalize": False,
        },
        static_data={"unused": True},
        disable_tqdm=True,
        optimizer_factory=DummyOptimizer,
        surface_reducer="mean",
    )

    surface = optimizer.optimize()
    artifacts = optimizer.get_artifacts()

    assert surface.shape == (1, 2, 2)
    assert np.allclose(surface, np.full((1, 2, 2), 3.0, dtype=np.float32))
    assert artifacts["member_surfaces"].shape == (3, 1, 2, 2)
    assert len(artifacts["member_artifacts"]) == 3
    assert artifacts["surface_reducer"] == "mean"
    assert artifacts["ensemble_size"] == 3
    assert artifacts["normalization_stats"] == {7: (12.5, 3.5)}


def test_dip_ensemble_optimizer_requires_optimization_before_artifacts():
    optimizer = DipEnsembleOptimizer(
        configuration={
            "aq_dataset": "metraq",
            "aq_backend": "files",
            "ensemble_size": 1,
            "validation_sensors": 1,
        },
        static_data={"unused": True},
        disable_tqdm=True,
        optimizer_factory=lambda **kwargs: None,
    )

    with pytest.raises(RuntimeError):
        optimizer.get_artifacts()
