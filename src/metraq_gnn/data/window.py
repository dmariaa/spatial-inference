from __future__ import annotations

import math

import numpy as np
import torch
from torch_geometric.data import Data

from metraq_gnn.data.graph import build_sensor_to_grid_edges, grid_to_nodes


ArrayLike = np.ndarray | torch.Tensor


def _as_tensor(array: ArrayLike, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype)


def _validate_spatial_mask(
    mask: ArrayLike,
    *,
    name: str,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    tensor = _as_tensor(mask, dtype=torch.bool)
    if tuple(tensor.shape) != grid_shape:
        raise ValueError(f"{name} must have shape {grid_shape}, got {tuple(tensor.shape)}")
    return tensor


def split_training_sensor_masks(
    train_sensor_mask: ArrayLike,
    final_observation_mask: ArrayLike,
    *,
    target_fraction: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split TRAIN sensor cells into visible context and hidden target cells.

    Target cells are sampled only where at least one pollutant has a valid
    final-timestep observation. At least one eligible cell is kept as context.
    """
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be between 0 and 1")

    train = _as_tensor(train_sensor_mask, dtype=torch.bool)
    final_observations = _as_tensor(final_observation_mask, dtype=torch.bool)
    if final_observations.ndim == train.ndim + 1:
        final_observations = final_observations.any(dim=0)
    if final_observations.shape != train.shape:
        raise ValueError("final_observation_mask must match train_sensor_mask spatially")

    eligible = train & final_observations
    eligible_indices = eligible.flatten().nonzero(as_tuple=False).flatten()
    if eligible_indices.numel() < 2:
        raise ValueError("at least two TRAIN sensor cells with final observations are required")

    target_count = max(1, math.ceil(eligible_indices.numel() * target_fraction))
    target_count = min(target_count, eligible_indices.numel() - 1)
    permutation = torch.randperm(eligible_indices.numel(), generator=generator)
    target_indices = eligible_indices[permutation[:target_count]]

    target = torch.zeros_like(train, dtype=torch.bool).flatten()
    target[target_indices] = True
    target = target.reshape_as(train)
    context = train & ~target
    return context, target


def build_graph_window(
    *,
    graph: Data,
    pollutant_data: ArrayLike,
    context_sensor_mask: ArrayLike,
    target_sensor_mask: ArrayLike,
    forbidden_context_mask: ArrayLike | None = None,
    extra_features: ArrayLike | None = None,
) -> Data:
    """Create one graph-learning sample from an interleaved pollutant window.

    ``pollutant_data`` follows the existing METRAQ convention
    ``(2 * pollutants, time, height, width)`` with interleaved value and
    availability channels. Only context-sensor AQ is exposed in ``x``. Targets
    contain final-timestep values and are selected exclusively by
    ``target_mask``.
    """
    pollution = _as_tensor(pollutant_data, dtype=torch.float32)
    if pollution.ndim != 4:
        raise ValueError(
            "pollutant_data must have shape (2 * pollutants, time, height, width)"
        )
    if pollution.shape[0] == 0 or pollution.shape[0] % 2:
        raise ValueError("pollutant_data must contain interleaved value/mask channels")

    grid_shape = (int(pollution.shape[-2]), int(pollution.shape[-1]))
    if int(graph.num_nodes) != grid_shape[0] * grid_shape[1]:
        raise ValueError("graph node count does not match pollutant_data grid")

    context_spatial = _validate_spatial_mask(
        context_sensor_mask,
        name="context_sensor_mask",
        grid_shape=grid_shape,
    )
    target_spatial = _validate_spatial_mask(
        target_sensor_mask,
        name="target_sensor_mask",
        grid_shape=grid_shape,
    )
    if torch.any(context_spatial & target_spatial):
        raise ValueError("context and target sensor masks must be disjoint")
    if not target_spatial.any():
        raise ValueError("target_sensor_mask must select at least one cell")

    forbidden_context = torch.zeros_like(context_spatial)
    if forbidden_context_mask is not None:
        forbidden_context = _validate_spatial_mask(
            forbidden_context_mask,
            name="forbidden_context_mask",
            grid_shape=grid_shape,
        )
        if torch.any(context_spatial & forbidden_context):
            raise ValueError("forbidden sensor cells must not appear in context")

    values = pollution[0::2]
    availability = pollution[1::2].bool()
    context_observations = availability & context_spatial[None, None, ...]
    source_sensor_mask = context_observations.any(dim=(0, 1))
    if not source_sensor_mask.any():
        raise ValueError("the window must contain at least one visible sensor observation")
    sensor_edge_index, sensor_edge_attr = build_sensor_to_grid_edges(source_sensor_mask)
    visible_values = torch.where(context_observations, values, 0.0)

    # Convert (pollutants, time, height, width) to (nodes, time, pollutants).
    value_features = grid_to_nodes(visible_values).permute(2, 1, 0)
    availability_features = grid_to_nodes(context_observations).permute(2, 1, 0).float()
    feature_parts = [value_features, availability_features]

    if extra_features is not None:
        extras = _as_tensor(extra_features, dtype=torch.float32)
        if extras.ndim != 4 or tuple(extras.shape[1:]) != (
            pollution.shape[1],
            *grid_shape,
        ):
            raise ValueError(
                "extra_features must have shape (features, time, height, width) "
                "matching pollutant_data"
            )
        feature_parts.append(grid_to_nodes(extras).permute(2, 1, 0))

    final_values = grid_to_nodes(values[:, -1]).T.contiguous()
    final_availability = grid_to_nodes(availability[:, -1]).T.contiguous()
    target_nodes = grid_to_nodes(target_spatial)
    target_mask = final_availability & target_nodes[:, None]
    if not target_mask.any():
        raise ValueError("target sensors have no valid final-timestep observations")

    return Data(
        x=torch.cat(feature_parts, dim=-1).contiguous(),
        edge_index=graph.edge_index.clone(),
        edge_attr=graph.edge_attr.clone(),
        sensor_edge_index=sensor_edge_index,
        sensor_edge_attr=sensor_edge_attr,
        pos=graph.pos.clone(),
        y=final_values,
        context_mask=grid_to_nodes(context_observations).permute(2, 1, 0).contiguous(),
        target_mask=target_mask,
        context_sensor_mask=grid_to_nodes(context_spatial),
        source_sensor_mask=grid_to_nodes(source_sensor_mask),
        target_sensor_mask=target_nodes,
        forbidden_context_mask=grid_to_nodes(forbidden_context),
        num_nodes=int(graph.num_nodes),
        grid_shape=grid_shape,
    )


__all__ = ["build_graph_window", "split_training_sensor_masks"]
