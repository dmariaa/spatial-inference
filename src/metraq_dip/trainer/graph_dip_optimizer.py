from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from metraq_dip.trainer.dip_optimizer import DipOptimizer, select_surface_from_validation
from metraq_gnn.data.graph import build_grid_graph, build_knn_sensor_to_grid_edges
from metraq_gnn.model import SensorToGridGNN
from metraq_gnn.trainer import masked_regression_losses


class GraphDipOptimizer(DipOptimizer):
    """Optimize a freshly initialized SensorToGridGNN for one DIP window."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        graph_config = dict(self.config.get("graph_dip") or {})
        self.nearest_sensors = int(graph_config.get("nearest_sensors", 4))
        self.patience = int(graph_config.get("patience", 50))
        self.min_delta = float(graph_config.get("min_delta", 0.0))
        self.weight_decay = float(graph_config.get("weight_decay", 1e-5))
        self.spatial_smoothness = float(graph_config.get("spatial_smoothness", 1e-4))
        if self.nearest_sensors <= 0:
            raise ValueError("graph_dip.nearest_sensors must be positive")
        if self.patience <= 0:
            raise ValueError("graph_dip.patience must be positive")
        if self.min_delta < 0 or self.weight_decay < 0 or self.spatial_smoothness < 0:
            raise ValueError("Graph DIP regularization values must not be negative")

    def _prepare_split_tensors(self) -> None:
        super()._prepare_split_tensors()
        pollutant_count = len(self._get_pollutants())
        _, _, timesteps, height, width = self.x_data.shape
        self.grid_shape = (height, width)

        observation_mask = self.split_data.get("observation_mask")
        if observation_mask is None:
            raise ValueError("Graph DIP requires observation_mask in split_data")
        observations = torch.as_tensor(observation_mask, dtype=torch.bool, device=self.device)
        if observations.shape != (pollutant_count, timesteps, height, width):
            raise ValueError(
                "observation_mask must have shape (pollutants, time, height, width)"
            )

        train_observations = observations & self.train_mask[0]
        val_observations = observations & self.val_mask[0]
        source_sensor_mask = train_observations.any(dim=(0, 1))
        if not source_sensor_mask.any():
            raise ValueError("Graph DIP requires at least one visible TRAIN sensor")
        if not val_observations[:, -1].any():
            raise ValueError("Graph DIP requires a final-timestep validation target")

        base_graph = build_grid_graph({"grid": np.zeros(self.grid_shape, dtype=np.uint8)})
        sensor_edge_index, sensor_edge_attr = build_knn_sensor_to_grid_edges(
            source_sensor_mask.detach().cpu(),
            k=self.nearest_sensors,
        )
        node_features = self.x_data[0].permute(2, 3, 1, 0).reshape(
            height * width,
            timesteps,
            self.x_data.shape[1],
        )
        self.graph_data = Data(
            x=node_features,
            edge_index=base_graph.edge_index.to(self.device),
            edge_attr=base_graph.edge_attr.to(self.device),
            sensor_edge_index=sensor_edge_index.to(self.device),
            sensor_edge_attr=sensor_edge_attr.to(self.device),
            pos=base_graph.pos.to(self.device),
            num_nodes=height * width,
        )
        self.train_target = self.train_data[0, :, -1].reshape(pollutant_count, -1).T
        self.val_target = self.val_data[0, :, -1].reshape(pollutant_count, -1).T
        self.train_target_mask = train_observations[:, -1].reshape(pollutant_count, -1).T
        self.val_target_mask = val_observations[:, -1].reshape(pollutant_count, -1).T

    def _get_model(self) -> torch.nn.Module:
        graph_config = dict(self.config.get("graph_dip") or {})
        hidden_channels = int(graph_config.get("hidden_channels", 32))
        attention_heads = int(graph_config.get("attention_heads", 4))
        return SensorToGridGNN(
            input_channels=int(self.graph_data.x.shape[-1]),
            output_channels=int(self.train_target.shape[-1]),
            edge_channels=int(self.graph_data.edge_attr.shape[-1]),
            temporal_hidden_channels=hidden_channels,
            graph_hidden_channels=hidden_channels,
            attention_heads=attention_heads,
            local_refinement_layers=int(graph_config.get("local_refinement_layers", 1)),
            dropout=float(graph_config.get("dropout", 0.0)),
        ).to(self.device)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.weight_decay,
        )

    def _prediction_surface(self, prediction: torch.Tensor) -> torch.Tensor:
        height, width = self.grid_shape
        return prediction.T.reshape(prediction.shape[1], height, width)

    def _smoothness_loss(self, prediction: torch.Tensor) -> torch.Tensor:
        source, target = self.graph_data.edge_index
        selected = source != target
        if not selected.any():
            return prediction.new_zeros(())
        return (prediction[source[selected]] - prediction[target[selected]]).square().mean()

    def _run_epoch(self, *, step: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(self.graph_data)
        train_losses = masked_regression_losses(
            prediction, self.train_target, self.train_target_mask
        )
        objective = (
            train_losses[self.optimization_loss]
            + self.spatial_smoothness * self._smoothness_loss(prediction)
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(self.graph_data)
            val_losses = masked_regression_losses(
                prediction, self.val_target, self.val_target_mask
            )
        self._record_epoch(
            step=step,
            output=self._prediction_surface(prediction),
            train_losses={"L1Loss": train_losses["mae"], "MSELoss": train_losses["mse"]},
            val_losses={"L1Loss": val_losses["mae"], "MSELoss": val_losses["mse"]},
        )
        return {
            "train_mae": float(train_losses["mae"].detach()),
            "val_mae": float(val_losses["mae"].detach()),
        }

    def optimize(self) -> np.ndarray:
        self._prepare_split_tensors()
        self._initialize_artifacts()
        self.model = self._get_model()
        self.optimizer = self._get_optimizer()

        epochs = int(self.config.get("epochs", 100))
        best_loss = float("inf")
        epochs_without_improvement = 0
        completed_epochs = 0
        with tqdm(total=epochs, leave=False, disable=self.disable_tqdm) as pbar:
            for step in range(epochs):
                log = self._run_epoch(step=step)
                completed_epochs = step + 1
                current_loss = self._get_selection_loss_history()[step].item()
                if current_loss < best_loss - self.min_delta:
                    best_loss = current_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                pbar.update(1)
                pbar.set_postfix(**log)
                if epochs_without_improvement >= self.patience:
                    break

        self.artifacts["epochs_completed"] = torch.tensor(completed_epochs, dtype=torch.long)
        if completed_epochs < epochs:
            for key in (
                "output_history",
                "train_l1_history",
                "train_mse_history",
                "val_l1_history",
                "val_mse_history",
            ):
                self.artifacts[key][completed_epochs:] = self.artifacts[key][completed_epochs - 1]
        k_best_n = min(self.k_best_n, completed_epochs)
        if self.surface_selection == "last":
            self.selected_surface_model_space = self.artifacts["output_history"][-1]
            self.selected_epoch_indices = torch.tensor([completed_epochs - 1], dtype=torch.long)
        else:
            reduction = {
                "validation": "mean",
                "validation_weighted": "weighted_mean",
                "validation_median": "median",
            }[self.surface_selection]
            self.selected_surface_model_space, self.selected_epoch_indices = select_surface_from_validation(
                output_history=self.artifacts["output_history"][:completed_epochs],
                val_loss_history=self._get_selection_loss_history()[:completed_epochs],
                k_best_n=k_best_n,
                reduction=reduction,
            )
        self.selected_surface = self._restore_surface_to_real_values(
            self.selected_surface_model_space
        )
        return self.selected_surface.detach().cpu().numpy()
