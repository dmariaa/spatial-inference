from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from tqdm import tqdm

from metraq_dip.data.data import collect_ensemble_data
from metraq_dip.trainer.dip_optimizer import DipOptimizer
from metraq_dip.trainer.optimizer_protocol import SurfaceOptimizer


def reduce_surface_ensemble(
    *,
    surfaces: list[np.ndarray],
    reduction: str = "mean",
) -> np.ndarray:
    if not surfaces:
        raise ValueError("surfaces must contain at least one surface")

    stacked = np.stack([np.asarray(surface, dtype=np.float32) for surface in surfaces], axis=0)
    if reduction == "mean":
        return stacked.mean(axis=0)
    if reduction == "median":
        return np.median(stacked, axis=0)

    raise ValueError(f"Unknown surface reduction '{reduction}'. Expected one of: mean, median")


class DipEnsembleOptimizer(SurfaceOptimizer):
    def __init__(
        self,
        *,
        configuration: dict[str, Any],
        static_data: dict[str, Any],
        device: str | None = None,
        disable_tqdm: bool = False,
        optimizer_factory: Callable[..., SurfaceOptimizer] = DipOptimizer,
        surface_reducer: str = "mean",
    ):
        self.config = configuration
        self.static_data = static_data
        self.device = device
        self.disable_tqdm = disable_tqdm
        self.optimizer_factory = optimizer_factory
        self.surface_reducer = surface_reducer
        self.ensemble_size = int(self.config.get("ensemble_size") or 1)

        if self.ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {self.ensemble_size}")

        self.member_surfaces: list[np.ndarray] = []
        self.member_artifacts: list[dict[str, Any]] = []
        self.selected_surface: np.ndarray | None = None

    def _collect_split_data(self) -> dict[str, Any]:
        validation_sensors = self.config.get("validation_sensors")
        if validation_sensors is None:
            raise ValueError("configuration must define validation_sensors for ensemble optimization")

        return collect_ensemble_data(
            data=self.static_data,
            number_of_noise_channels=int(self.config.get("number_of_noise_channels") or 8),
            number_of_val_sensors=validation_sensors,
            add_distance_to_sensors=bool(self.config.get("add_distance_to_sensors")),
            normalize=bool(self.config.get("normalize")),
        )

    def _create_optimizer(self, *, split_data: dict[str, Any]) -> SurfaceOptimizer:
        return self.optimizer_factory(
            configuration=self.config,
            split_data=split_data,
            device=self.device,
            disable_tqdm=self.disable_tqdm,
        )

    def optimize(self) -> np.ndarray:
        self.member_surfaces = []
        self.member_artifacts = []

        with tqdm(total=self.ensemble_size, leave=False, disable=self.disable_tqdm) as pbar:
            for _ in range(self.ensemble_size):
                split_data = self._collect_split_data()
                optimizer = self._create_optimizer(split_data=split_data)
                surface = np.asarray(optimizer.optimize(), dtype=np.float32)

                self.member_surfaces.append(surface)
                self.member_artifacts.append(optimizer.get_artifacts())
                pbar.update(1)

        self.selected_surface = reduce_surface_ensemble(
            surfaces=self.member_surfaces,
            reduction=self.surface_reducer,
        )
        return np.array(self.selected_surface, copy=True)

    def get_artifacts(self) -> dict[str, Any]:
        if self.selected_surface is None:
            raise RuntimeError("optimize() must be called before artifacts are available.")

        artifacts = {
            "surface": np.array(self.selected_surface, copy=True),
            "member_surfaces": np.stack(self.member_surfaces, axis=0),
            "member_artifacts": list(self.member_artifacts),
            "surface_reducer": self.surface_reducer,
            "ensemble_size": self.ensemble_size,
        }

        if self.member_artifacts:
            normalization_stats = self.member_artifacts[0].get("normalization_stats")
            if normalization_stats is not None:
                artifacts["normalization_stats"] = normalization_stats

        return artifacts
