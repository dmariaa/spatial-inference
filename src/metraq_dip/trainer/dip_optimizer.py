from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from metraq_dip.model.base_model_v2 import Autoencoder3D
from metraq_dip.model.unet_model import UNet3D
from metraq_dip.tools.tools import get_padding
from metraq_dip.trainer.loss import get_losses
from metraq_dip.trainer.optimizer_protocol import SurfaceOptimizer


def select_surface_from_validation(
    *,
    output_history: torch.Tensor | np.ndarray,
    val_loss_history: torch.Tensor | np.ndarray,
    k_best_n: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    history = torch.as_tensor(output_history, dtype=torch.float32)
    losses = torch.as_tensor(val_loss_history, dtype=torch.float32)

    if history.ndim != 4:
        raise ValueError(
            "output_history must have shape (epochs, channels, height, width)"
        )
    if losses.ndim != 1:
        raise ValueError("val_loss_history must have shape (epochs,)")
    if history.shape[0] != losses.shape[0]:
        raise ValueError("output_history and val_loss_history must have the same number of epochs")
    if k_best_n < 1 or k_best_n > history.shape[0]:
        raise ValueError(f"k_best_n={k_best_n}")

    best_indices = torch.topk(losses, k_best_n, largest=False, sorted=True).indices
    surface = history.index_select(dim=0, index=best_indices).mean(dim=0)
    return surface, best_indices


class DipOptimizer(SurfaceOptimizer):
    def __init__(
        self,
        *,
        configuration: dict[str, Any],
        split_data: dict[str, Any],
        device: str | torch.device | None = None,
        disable_tqdm: bool = False,
    ):
        self.config = configuration
        self.split_data = split_data
        self.disable_tqdm = disable_tqdm

        self.k_best_n = self.config.get("k_best_n") or 10
        self.optimization_loss = str(self.config.get("optimization_loss", "mae")).lower()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.optimization_loss_map = {
            "mae": "L1Loss",
            "mse": "MSELoss",
            "rmse": "RMSELoss",
        }
        if self.optimization_loss not in self.optimization_loss_map:
            valid_options = ", ".join(sorted(self.optimization_loss_map))
            raise ValueError(
                f"Unknown optimization loss '{self.optimization_loss}'. Expected one of: {valid_options}"
            )

        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.padding: tuple[int, ...] | None = None
        self.selected_surface: torch.Tensor | None = None
        self.selected_surface_model_space: torch.Tensor | None = None
        self.selected_epoch_indices: torch.Tensor | None = None
        self.artifacts: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _expand_mask_to_match_data(
        *,
        mask: torch.Tensor,
        data: torch.Tensor,
        mask_name: str,
    ) -> torch.Tensor:
        if mask.ndim != data.ndim:
            raise ValueError(f"{mask_name} and its data tensor must have the same number of dimensions")

        expanded_shape: list[int] = []
        for mask_dim, data_dim in zip(mask.shape, data.shape):
            if mask_dim == data_dim or mask_dim == 1:
                expanded_shape.append(data_dim)
                continue
            raise ValueError(f"{mask_name} is not broadcastable to the corresponding data shape")

        return mask.expand(*expanded_shape)

    def _prepare_split_tensors(self) -> None:
        required_keys = ("input_data", "train_data", "val_data", "train_mask", "val_mask")
        missing_keys = [key for key in required_keys if key not in self.split_data]
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"split_data is missing required keys: {missing}")

        self.x_data = torch.as_tensor(
            self.split_data["input_data"],
            dtype=torch.float32,
            device=self.device,
        )[None, ...]
        self.train_data = torch.as_tensor(
            self.split_data["train_data"],
            dtype=torch.float32,
            device=self.device,
        )[None, ...]
        self.val_data = torch.as_tensor(
            self.split_data["val_data"],
            dtype=torch.float32,
            device=self.device,
        )[None, ...]
        self.train_mask = torch.as_tensor(
            self.split_data["train_mask"],
            device=self.device,
        )[None, ...].bool()
        self.val_mask = torch.as_tensor(
            self.split_data["val_mask"],
            device=self.device,
        )[None, ...].bool()

        if self.train_data.shape != self.val_data.shape:
            raise ValueError("train_data and val_data must have the same shape")
        self.train_mask = self._expand_mask_to_match_data(
            mask=self.train_mask,
            data=self.train_data,
            mask_name="train_mask",
        )
        self.val_mask = self._expand_mask_to_match_data(
            mask=self.val_mask,
            data=self.val_data,
            mask_name="val_mask",
        )

    def _initialize_artifacts(self) -> None:
        _, channels, _, height, width = self.train_data.shape
        epochs = self.config.get("epochs", 100)

        self.artifacts = {
            "input_data": self.x_data[0].detach().cpu(),
            "train_data": self.train_data[0].detach().cpu(),
            "val_data": self.val_data[0].detach().cpu(),
            "train_mask": self.train_mask[0].detach().cpu(),
            "val_mask": self.val_mask[0].detach().cpu(),
            "output_history": torch.zeros((epochs, channels, height, width), dtype=torch.float32),
            "train_l1_history": torch.zeros(epochs, dtype=torch.float32),
            "train_mse_history": torch.zeros(epochs, dtype=torch.float32),
            "val_l1_history": torch.zeros(epochs, dtype=torch.float32),
            "val_mse_history": torch.zeros(epochs, dtype=torch.float32),
        }

    def _get_pollutants(self) -> list[int]:
        pollutants = self.split_data.get("pollutants")
        if pollutants is None:
            pollutants = self.config.get("pollutants")
        if pollutants is None:
            raise ValueError("Pollutants are required to denormalize the selected surface.")
        return list(pollutants)

    def _restore_surface_to_real_values(self, surface: torch.Tensor) -> torch.Tensor:
        restored_surface = surface.detach().cpu().clone()
        if not self.config.get("normalize", False):
            return restored_surface

        normalization_stats = self.split_data.get("normalization_stats")
        if normalization_stats is None:
            raise ValueError("normalization_stats are required when normalize=True.")

        pollutants = self._get_pollutants()
        if len(pollutants) != restored_surface.shape[0]:
            raise ValueError(
                "The number of pollutants must match the number of output channels when normalize=True."
            )

        for channel_idx, pollutant in enumerate(pollutants):
            if pollutant not in normalization_stats:
                raise ValueError(f"Missing normalization_stats for pollutant {pollutant}.")
            mean, std = normalization_stats[pollutant]
            restored_surface[channel_idx] = restored_surface[channel_idx] * (std + 1e-6) + mean

        return restored_surface

    def _get_model(self) -> torch.nn.Module:
        levels = self.config["model"]["levels"]
        base_channels = self.config["model"]["base_channels"]
        preserve_time = self.config["model"]["preserve_time"]
        skip_connections = self.config["model"].get("skip_connections", False)
        learned_upsampling = self.config["model"]["learned_upsampling"]
        architecture = self.config["model"].get("architecture", "autoencoder").lower()

        self.padding = get_padding(
            self.x_data.shape,
            levels=levels,
            preserve_time=preserve_time,
        )

        model_registry = {
            "autoencoder": Autoencoder3D,
            "unet": UNet3D,
        }
        if architecture not in model_registry:
            valid_options = ", ".join(sorted(model_registry))
            raise ValueError(f"Unknown model architecture '{architecture}'. Expected one of: {valid_options}")

        model_kwargs = {
            "in_channels": self.x_data.shape[1],
            "out_channels": self.train_data.shape[1],
            "base_channels": base_channels,
            "levels": levels,
            "preserve_time": preserve_time,
            "learned_upsampling": learned_upsampling,
        }
        if architecture == "autoencoder":
            model_kwargs["use_skip_connections"] = skip_connections

        model_class = model_registry[architecture]
        return model_class(**model_kwargs).to(self.device)

    def _call_model(self, x: torch.Tensor) -> torch.Tensor:
        _, out_channels, timesteps, height, width = self.train_data.shape
        y_hat = self.model(x)
        return y_hat[:, :out_channels, :timesteps, :height, :width]

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def _get_optimization_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return losses[self.optimization_loss_map[self.optimization_loss]]

    def _record_epoch(
        self,
        *,
        step: int,
        output: torch.Tensor,
        train_losses: dict[str, torch.Tensor],
        val_losses: dict[str, torch.Tensor],
    ) -> None:
        self.artifacts["output_history"][step] = output.detach().cpu()
        self.artifacts["train_l1_history"][step] = train_losses["L1Loss"].detach().cpu()
        self.artifacts["train_mse_history"][step] = train_losses["MSELoss"].detach().cpu()
        self.artifacts["val_l1_history"][step] = val_losses["L1Loss"].detach().cpu()
        self.artifacts["val_mse_history"][step] = val_losses["MSELoss"].detach().cpu()

    def _run_epoch(self, *, step: int) -> dict[str, float]:
        x = F.pad(self.x_data, self.padding)

        self.model.train()
        self.optimizer.zero_grad()
        y_hat = self._call_model(x)
        train_losses = get_losses(self.train_data, y_hat, self.train_mask)
        optimization_loss = self._get_optimization_loss(train_losses)
        optimization_loss.backward()
        self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            y_hat = self._call_model(x)
            val_losses = get_losses(self.val_data, y_hat, self.val_mask)

        output = y_hat[0, :, -1, ...]
        self._record_epoch(
            step=step,
            output=output,
            train_losses=train_losses,
            val_losses=val_losses,
        )
        return {
            "train_mae": train_losses["L1Loss"].item(),
            "val_mae": val_losses["L1Loss"].item(),
        }

    def optimize(self) -> np.ndarray:
        self._prepare_split_tensors()
        self._initialize_artifacts()
        self.model = self._get_model()
        self.optimizer = self._get_optimizer()

        epochs = self.config.get("epochs", 100)
        log: dict[str, float] = {}
        with tqdm(total=epochs, leave=False, disable=self.disable_tqdm) as pbar:
            for step in range(epochs):
                log.update(self._run_epoch(step=step))
                pbar.update(1)
                pbar.set_postfix(**log)

        self.selected_surface_model_space, self.selected_epoch_indices = select_surface_from_validation(
            output_history=self.artifacts["output_history"],
            val_loss_history=self.artifacts["val_l1_history"],
            k_best_n=self.k_best_n,
        )
        self.selected_surface = self._restore_surface_to_real_values(self.selected_surface_model_space)
        return self.selected_surface.detach().cpu().numpy()

    def get_selected_epoch_indices(self) -> np.ndarray:
        if self.selected_epoch_indices is None:
            raise RuntimeError("optimize() must be called before selected epoch indices are available.")
        return self.selected_epoch_indices.detach().cpu().numpy()

    def get_artifacts(self) -> dict[str, Any]:
        if self.artifacts is None:
            raise RuntimeError("optimize() must be called before artifacts are available.")

        artifacts = {
            key: value.detach().cpu().numpy()
            for key, value in self.artifacts.items()
        }
        if self.selected_surface is not None:
            artifacts["surface"] = self.selected_surface.detach().cpu().numpy()
        if self.selected_surface_model_space is not None:
            artifacts["surface_model_space"] = self.selected_surface_model_space.detach().cpu().numpy()
        if self.selected_epoch_indices is not None:
            artifacts["selected_epoch_indices"] = self.selected_epoch_indices.detach().cpu().numpy()
        normalization_stats = self.split_data.get("normalization_stats")
        if normalization_stats is not None:
            artifacts["normalization_stats"] = normalization_stats
        return artifacts
