from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from metraq_gnn.data.cache import AQSensorCache


def split_sensor_nodes(
    cache: AQSensorCache,
    *,
    validation_nodes: int,
    test_nodes: int,
    seed: int,
    test_sensor_ids: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible, disjoint TRAIN/VALIDATION/TEST masks over observed nodes."""
    nodes = np.unique(cache.sensor_node_indices)
    if validation_nodes <= 0 or (test_sensor_ids is None and test_nodes <= 0):
        raise ValueError("validation_nodes and test_nodes must be positive")
    if test_sensor_ids is None:
        explicit_test_nodes = None
        effective_test_nodes = test_nodes
    else:
        requested_ids = np.asarray(test_sensor_ids, dtype=np.int64)
        if requested_ids.ndim != 1 or requested_ids.size == 0:
            raise ValueError("test_sensor_ids must contain at least one sensor id")
        if np.unique(requested_ids).size != requested_ids.size:
            raise ValueError("test_sensor_ids must not contain duplicates")
        sensor_lookup = {
            int(sensor_id): int(node)
            for sensor_id, node in zip(cache.sensor_ids, cache.sensor_node_indices)
        }
        missing = [
            int(sensor_id)
            for sensor_id in requested_ids
            if int(sensor_id) not in sensor_lookup
        ]
        if missing:
            raise ValueError(f"unknown test sensor ids: {missing}")
        explicit_test_nodes = np.unique(
            [sensor_lookup[int(sensor_id)] for sensor_id in requested_ids]
        )
        if explicit_test_nodes.size != requested_ids.size:
            raise ValueError("test_sensor_ids must map to distinct grid nodes")
        effective_test_nodes = int(explicit_test_nodes.size)

    if validation_nodes + effective_test_nodes >= len(nodes):
        raise ValueError("at least one observed node must remain for training")
    candidate_nodes = (
        nodes
        if explicit_test_nodes is None
        else nodes[~np.isin(nodes, explicit_test_nodes)]
    )
    shuffled = candidate_nodes[np.random.default_rng(seed).permutation(len(candidate_nodes))]
    validation = shuffled[:validation_nodes]
    if explicit_test_nodes is None:
        test = shuffled[validation_nodes : validation_nodes + effective_test_nodes]
        train = shuffled[validation_nodes + effective_test_nodes :]
    else:
        test = explicit_test_nodes
        train = shuffled[validation_nodes:]
    shape = cache.grid_shape

    def mask(selected: np.ndarray) -> np.ndarray:
        result = np.zeros(int(np.prod(shape)), dtype=bool)
        result[selected] = True
        return result.reshape(shape)

    return mask(train), mask(validation), mask(test)


def compute_training_normalization(
    cache: AQSensorCache,
    *,
    train_node_mask: np.ndarray,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate per-pollutant mean/std from TRAIN dates and TRAIN nodes only."""
    spatial = np.asarray(train_node_mask, dtype=bool)
    if spatial.shape != cache.grid_shape:
        raise ValueError(f"train_node_mask must have shape {cache.grid_shape}")
    timestamps = cache.time_index
    first = int(timestamps.searchsorted(pd.Timestamp(start)))
    last = int(timestamps.searchsorted(pd.Timestamp(end), side="right"))
    if first >= last:
        raise ValueError("normalization interval contains no cached timestamps")
    allowed_sensors = spatial.reshape(-1)[cache.sensor_node_indices]
    values = np.asarray(cache.values[first:last, allowed_sensors, :], dtype=np.float64)
    available = np.asarray(cache.availability[first:last, allowed_sensors, :], dtype=bool)
    means = np.empty(values.shape[-1], dtype=np.float32)
    stds = np.empty_like(means)
    for pollutant in range(values.shape[-1]):
        selected = values[..., pollutant][available[..., pollutant]]
        if selected.size == 0:
            raise ValueError("no TRAIN observations available for normalization")
        means[pollutant] = selected.mean()
        stds[pollutant] = selected.std()
        if not np.isfinite(stds[pollutant]) or stds[pollutant] < 1e-6:
            stds[pollutant] = 1.0
    return means, stds


__all__ = ["compute_training_normalization", "split_sensor_nodes"]
