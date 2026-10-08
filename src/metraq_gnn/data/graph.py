from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch_geometric.data import Data


def grid_to_nodes(array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Flatten the final ``(height, width)`` axes into row-major graph nodes."""
    if array.ndim < 2:
        raise ValueError("array must have at least height and width dimensions")
    return array.reshape(*array.shape[:-2], -1)


def nodes_to_grid(
    array: np.ndarray | torch.Tensor,
    grid_shape: Sequence[int],
) -> np.ndarray | torch.Tensor:
    """Restore a row-major node axis to the final ``(height, width)`` axes."""
    if array.ndim < 1:
        raise ValueError("array must have a node dimension")
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain exactly height and width")

    height, width = (int(value) for value in grid_shape)
    if height <= 0 or width <= 0:
        raise ValueError("grid_shape values must be positive")
    if array.shape[-1] != height * width:
        raise ValueError(
            f"node dimension has size {array.shape[-1]}, expected {height * width}"
        )
    return array.reshape(*array.shape[:-1], height, width)


def build_grid_graph(grid_ctx: dict) -> Data:
    """Build a directed eight-neighbour graph with explicit self-loops.

    Nodes follow the row-major mapping ``node_id = row * width + column``.
    Edge attributes describe message travel from source to target as normalized
    ``(dx, dy, distance, geometric_weight)``. Grid rows grow downwards, hence
    physical ``dy`` is the negated row displacement.
    """
    if "grid" not in grid_ctx:
        raise ValueError("grid_ctx must contain 'grid'")

    grid = np.asarray(grid_ctx["grid"])
    if grid.ndim != 2:
        raise ValueError("grid_ctx['grid'] must be two-dimensional")

    height, width = grid.shape
    if height == 0 or width == 0:
        raise ValueError("grid_ctx['grid'] must not be empty")

    sources: list[int] = []
    targets: list[int] = []
    attributes: list[tuple[float, float, float, float]] = []

    for target_row in range(height):
        for target_col in range(width):
            target = target_row * width + target_col
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    source_row = target_row + row_offset
                    source_col = target_col + col_offset
                    if not (0 <= source_row < height and 0 <= source_col < width):
                        continue

                    source = source_row * width + source_col
                    dx = float(target_col - source_col)
                    dy = float(source_row - target_row)
                    distance = float(np.hypot(dx, dy))
                    geometric_weight = float(np.exp(-0.5 * distance**2))

                    sources.append(source)
                    targets.append(target)
                    attributes.append((dx, dy, distance, geometric_weight))

    rows, cols = np.indices((height, width))
    positions = np.stack((cols, -rows), axis=-1).reshape(-1, 2)

    return Data(
        pos=torch.as_tensor(positions, dtype=torch.float32),
        edge_index=torch.tensor([sources, targets], dtype=torch.long),
        edge_attr=torch.tensor(attributes, dtype=torch.float32),
        num_nodes=height * width,
        grid_shape=(height, width),
    )


def build_sensor_to_grid_edges(
    source_sensor_mask: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Connect every visible sensor node to every grid node."""
    mask = torch.as_tensor(source_sensor_mask, dtype=torch.bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("source_sensor_mask must be a non-empty 2D mask")
    height, width = (int(value) for value in mask.shape)
    sources = mask.flatten().nonzero(as_tuple=False).flatten()
    destinations = torch.arange(height * width, dtype=torch.long)
    source_index = sources.repeat_interleave(destinations.numel())
    destination_index = destinations.repeat(sources.numel())
    source_row = torch.div(source_index, width, rounding_mode="floor")
    source_col = source_index % width
    destination_row = torch.div(destination_index, width, rounding_mode="floor")
    destination_col = destination_index % width
    dx = (destination_col - source_col).float() / max(width - 1, 1)
    dy = (source_row - destination_row).float() / max(height - 1, 1)
    distance = torch.sqrt(dx.square() + dy.square())
    geometric_weight = torch.exp(-distance)
    return (
        torch.stack((source_index, destination_index)),
        torch.stack((dx, dy, distance, geometric_weight), dim=1),
    )


def build_knn_sensor_to_grid_edges(
    source_sensor_mask: np.ndarray | torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Connect every grid node to its ``k`` nearest visible sensor nodes."""
    mask = torch.as_tensor(source_sensor_mask, dtype=torch.bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("source_sensor_mask must be a non-empty 2D mask")
    if k <= 0:
        raise ValueError("k must be positive")

    height, width = (int(value) for value in mask.shape)
    sources = mask.flatten().nonzero(as_tuple=False).flatten()
    destinations = torch.arange(height * width, dtype=torch.long)
    source_rows = torch.div(sources, width, rounding_mode="floor")
    source_cols = sources % width
    destination_rows = torch.div(destinations, width, rounding_mode="floor")
    destination_cols = destinations % width
    dx_all = (destination_cols[:, None] - source_cols[None, :]).float() / max(width - 1, 1)
    dy_all = (source_rows[None, :] - destination_rows[:, None]).float() / max(height - 1, 1)
    distance_all = torch.sqrt(dx_all.square() + dy_all.square())
    candidate_count = min(int(k), int(sources.numel()))
    candidates = torch.topk(
        distance_all,
        k=candidate_count,
        dim=1,
        largest=False,
        sorted=True,
    ).indices

    destination_index = destinations.repeat_interleave(candidate_count)
    source_index = sources[candidates.reshape(-1)]
    dx = dx_all.gather(1, candidates).reshape(-1)
    dy = dy_all.gather(1, candidates).reshape(-1)
    distance = distance_all.gather(1, candidates).reshape(-1)
    return (
        torch.stack((source_index, destination_index)),
        torch.stack((dx, dy, distance, torch.exp(-distance)), dim=1),
    )


__all__ = [
    "build_grid_graph",
    "build_knn_sensor_to_grid_edges",
    "build_sensor_to_grid_edges",
    "grid_to_nodes",
    "nodes_to_grid",
]
