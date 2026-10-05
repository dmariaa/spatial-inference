from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Protocol

import torch
from torch import nn
from torch_geometric.data import Data
from tqdm.auto import tqdm


LossName = Literal["mae", "mse", "rmse"]


class TrainingLogger(Protocol):
    def log_epoch(self, metrics: Mapping[str, float], *, step: int) -> None: ...

    def log_summary(self, summary: Mapping[str, float | int]) -> None: ...

    def log_checkpoint(self, path: str | Path) -> None: ...


def masked_regression_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Calculate node losses using only explicitly selected targets."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if mask.shape != target.shape:
        raise ValueError("mask and target must have the same shape")

    selected = mask.bool()
    if not selected.any():
        raise ValueError("mask must select at least one target value")

    difference = prediction[selected] - target[selected]
    mse = difference.square().mean()
    return {
        "mae": difference.abs().mean(),
        "mse": mse,
        "rmse": torch.sqrt(mse),
    }


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    history: tuple[dict[str, float], ...]


class GNNTrainer:
    """Train a node-regression model and retain the best validation checkpoint."""

    def __init__(
        self,
        *,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimization_loss: LossName = "mae",
        patience: int = 20,
        min_delta: float = 0.0,
        gradient_clip_norm: float | None = 1.0,
        device: str | torch.device | None = None,
        mixed_precision: bool = False,
        logger: TrainingLogger | None = None,
        show_progress: bool = False,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if optimization_loss not in {"mae", "mse", "rmse"}:
            raise ValueError("optimization_loss must be one of: mae, mse, rmse")
        if patience <= 0:
            raise ValueError("patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must not be negative")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        if mixed_precision and self.device.type != "cuda":
            raise ValueError("mixed_precision currently requires a CUDA device")
        self.mixed_precision = bool(mixed_precision)
        self.grad_scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.mixed_precision,
        )
        self.optimization_loss = optimization_loss
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_clip_norm = gradient_clip_norm
        self.logger = logger
        self.show_progress = show_progress
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _run_loader(
        self,
        loader: Iterable[Data],
        *,
        training: bool,
        description: str | None = None,
    ) -> dict[str, float]:
        self.model.train(training)
        totals = {"mae": 0.0, "mse": 0.0}
        selected_count = 0
        batch_count = 0

        batches = tqdm(
            loader,
            total=len(loader) if hasattr(loader, "__len__") else None,
            desc=description,
            leave=False,
            dynamic_ncols=True,
            mininterval=5.0,
            disable=not self.show_progress,
        )
        for batch in batches:
            batch_count += 1
            batch = batch.to(self.device)
            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.mixed_precision,
                ):
                    prediction = self.model(batch)
                    losses = masked_regression_losses(prediction, batch.y, batch.target_mask)
                    objective = losses[self.optimization_loss]
                if training:
                    self.grad_scaler.scale(objective).backward()
                    if self.gradient_clip_norm is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

            count = int(batch.target_mask.bool().sum().item())
            selected_count += count
            totals["mae"] += float(losses["mae"].detach().item()) * count
            totals["mse"] += float(losses["mse"].detach().item()) * count

        if batch_count == 0:
            raise ValueError("data loader must contain at least one batch")
        if selected_count == 0:
            raise ValueError("data loader must contain at least one target value")

        mae = totals["mae"] / selected_count
        mse = totals["mse"] / selected_count
        return {"mae": mae, "mse": mse, "rmse": mse**0.5}

    def fit(
        self,
        train_loader: Iterable[Data],
        validation_loader: Iterable[Data],
        *,
        epochs: int,
        checkpoint_path: str | Path | None = None,
    ) -> TrainingResult:
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        best_epoch = -1
        best_validation_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []

        epoch_progress = tqdm(
            range(epochs),
            desc="Epochs",
            unit="epoch",
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        for epoch in epoch_progress:
            self._set_dataset_epoch(train_loader, epoch)
            train_metrics = self._run_loader(
                train_loader,
                training=True,
                description=f"Epoch {epoch + 1}/{epochs} train",
            )
            validation_metrics = self._run_loader(
                validation_loader,
                training=False,
                description=f"Epoch {epoch + 1}/{epochs} validation",
            )
            validation_loss = validation_metrics[self.optimization_loss]
            history.append(
                {
                    "epoch": float(epoch),
                    **{f"train_{name}": value for name, value in train_metrics.items()},
                    **{f"validation_{name}": value for name, value in validation_metrics.items()},
                }
            )
            if self.logger is not None:
                self.logger.log_epoch(history[-1], step=epoch)

            epoch_progress.set_postfix(
                train_mae=f"{train_metrics['mae']:.4f}",
                validation_mae=f"{validation_metrics['mae']:.4f}",
                refresh=True,
            )

            if validation_loss < best_validation_loss - self.min_delta:
                best_epoch = epoch
                best_validation_loss = validation_loss
                best_state = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
                if checkpoint_path is not None:
                    destination = Path(checkpoint_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": best_epoch,
                            "validation_loss": best_validation_loss,
                            "optimization_loss": self.optimization_loss,
                            "model_state_dict": best_state,
                        },
                        destination,
                    )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is None:
            raise RuntimeError("training did not produce a finite validation checkpoint")
        self.model.load_state_dict(best_state)
        result = TrainingResult(
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            epochs_completed=len(history),
            history=tuple(history),
        )
        if self.logger is not None:
            self.logger.log_summary(
                {
                    "best_epoch": result.best_epoch,
                    "best_validation_loss": result.best_validation_loss,
                    "epochs_completed": result.epochs_completed,
                }
            )
            if checkpoint_path is not None:
                self.logger.log_checkpoint(checkpoint_path)
        return result

    @staticmethod
    def _set_dataset_epoch(loader: Iterable[Data], epoch: int) -> None:
        dataset = getattr(loader, "dataset", None)
        while dataset is not None:
            setter = getattr(dataset, "set_epoch", None)
            if setter is not None:
                setter(epoch)
                return
            dataset = getattr(dataset, "dataset", None)

    def evaluate(self, loader: Iterable[Data]) -> dict[str, float]:
        return self._run_loader(loader, training=False)


__all__ = ["GNNTrainer", "TrainingLogger", "TrainingResult", "masked_regression_losses"]
