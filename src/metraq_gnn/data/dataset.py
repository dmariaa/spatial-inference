from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from metraq_gnn.data.window import build_graph_window, split_training_sensor_masks
from metraq_gnn.data.cache import AQSensorCache


ArrayLike = np.ndarray | torch.Tensor


class GraphWindowDataset(Dataset[Data]):
    """Expose complete hourly windows from one preloaded graph time series.

    Training datasets use ``target_fraction`` to hide a deterministic random
    subset of TRAIN sensor cells. Validation and test datasets instead provide a
    fixed ``target_sensor_mask``. Call ``set_epoch`` before each training epoch
    to obtain a new, reproducible masking pattern.
    """

    def __init__(
        self,
        *,
        graph: Data,
        pollutant_data: ArrayLike,
        time_index: Sequence[pd.Timestamp] | pd.DatetimeIndex,
        context_sensor_mask: ArrayLike,
        forbidden_context_mask: ArrayLike,
        hours: int,
        split_start: str | pd.Timestamp,
        split_end: str | pd.Timestamp,
        extra_features: ArrayLike | None = None,
        target_sensor_mask: ArrayLike | None = None,
        target_fraction: float | None = None,
        seed: int = 42,
    ) -> None:
        if hours <= 0:
            raise ValueError("hours must be positive")
        if (target_sensor_mask is None) == (target_fraction is None):
            raise ValueError("provide exactly one of target_sensor_mask or target_fraction")

        pollution = torch.as_tensor(pollutant_data, dtype=torch.float32)
        if pollution.ndim != 4 or pollution.shape[0] == 0 or pollution.shape[0] % 2:
            raise ValueError(
                "pollutant_data must have shape (2 * pollutants, time, height, width)"
            )
        timestamps = pd.DatetimeIndex(time_index)
        if len(timestamps) != pollution.shape[1]:
            raise ValueError("time_index length must match pollutant_data time dimension")
        if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
            raise ValueError("time_index must be increasing and contain no duplicates")

        grid_shape = tuple(int(value) for value in pollution.shape[-2:])
        if int(graph.num_nodes) != grid_shape[0] * grid_shape[1]:
            raise ValueError("graph node count does not match pollutant_data grid")
        context = self._spatial_mask(context_sensor_mask, "context_sensor_mask", grid_shape)
        forbidden = self._spatial_mask(
            forbidden_context_mask,
            "forbidden_context_mask",
            grid_shape,
        )
        if torch.any(context & forbidden):
            raise ValueError("forbidden sensor cells must not appear in context")

        fixed_target = None
        if target_sensor_mask is not None:
            fixed_target = self._spatial_mask(
                target_sensor_mask,
                "target_sensor_mask",
                grid_shape,
            )
            if torch.any(context & fixed_target):
                raise ValueError("context and target sensor masks must be disjoint")
        elif target_fraction is not None and not 0.0 < target_fraction < 1.0:
            raise ValueError("target_fraction must be between 0 and 1")

        extras = None
        if extra_features is not None:
            extras = torch.as_tensor(extra_features, dtype=torch.float32)
            if extras.ndim != 4 or tuple(extras.shape[1:]) != (
                pollution.shape[1],
                *grid_shape,
            ):
                raise ValueError(
                    "extra_features must have shape (features, time, height, width)"
                )

        start = pd.Timestamp(split_start)
        end = pd.Timestamp(split_end)
        if start > end:
            raise ValueError("split_start must not be after split_end")

        self.graph = graph
        self.pollutant_data = pollution
        self.extra_features = extras
        self.time_index = timestamps
        self.context_sensor_mask = context
        self.forbidden_context_mask = forbidden
        self.target_sensor_mask = fixed_target
        self.target_fraction = target_fraction
        self.hours = hours
        self.seed = int(seed)
        self.epoch = 0
        self.end_indices = self._select_end_indices(split_start=start, split_end=end)
        if not self.end_indices:
            raise ValueError("split contains no complete windows with valid targets")

    @staticmethod
    def _spatial_mask(
        mask: ArrayLike,
        name: str,
        grid_shape: tuple[int, int],
    ) -> torch.Tensor:
        tensor = torch.as_tensor(mask, dtype=torch.bool)
        if tuple(tensor.shape) != grid_shape:
            raise ValueError(f"{name} must have shape {grid_shape}, got {tuple(tensor.shape)}")
        return tensor

    def _has_usable_target(self, end_index: int) -> bool:
        final_availability = self.pollutant_data[1::2, end_index].bool()
        if self.target_sensor_mask is not None:
            return bool(torch.any(final_availability & self.target_sensor_mask[None, ...]))
        eligible = final_availability.any(dim=0) & self.context_sensor_mask
        return int(eligible.sum()) >= 2

    def _select_end_indices(
        self,
        *,
        split_start: pd.Timestamp,
        split_end: pd.Timestamp,
    ) -> list[int]:
        one_hour_ns = pd.Timedelta(hours=1).value
        result = []
        for end_index in range(self.hours - 1, len(self.time_index)):
            start_index = end_index - self.hours + 1
            window_times = self.time_index[start_index : end_index + 1]
            if window_times[0] < split_start or window_times[-1] > split_end:
                continue
            if len(window_times) > 1 and not np.all(np.diff(window_times.asi8) == one_hour_ns):
                continue
            if self._has_usable_target(end_index):
                result.append(end_index)
        return result

    @property
    def window_end_times(self) -> pd.DatetimeIndex:
        return self.time_index[self.end_indices]

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must not be negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, index: int) -> Data:
        end_index = self.end_indices[index]
        start_index = end_index - self.hours + 1
        pollution_window = self.pollutant_data[:, start_index : end_index + 1]
        extra_window = (
            None
            if self.extra_features is None
            else self.extra_features[:, start_index : end_index + 1]
        )

        if self.target_sensor_mask is None:
            final_availability = pollution_window[1::2, -1].bool()
            generator = torch.Generator().manual_seed(
                self.seed + self.epoch * len(self) + index
            )
            context, target = split_training_sensor_masks(
                self.context_sensor_mask,
                final_availability,
                target_fraction=float(self.target_fraction),
                generator=generator,
            )
        else:
            context = self.context_sensor_mask
            target = self.target_sensor_mask

        sample = build_graph_window(
            graph=self.graph,
            pollutant_data=pollution_window,
            context_sensor_mask=context,
            target_sensor_mask=target,
            forbidden_context_mask=self.forbidden_context_mask,
            extra_features=extra_window,
        )
        sample.window_start_ns = torch.tensor(
            self.time_index[start_index].value,
            dtype=torch.int64,
        )
        sample.window_end_ns = torch.tensor(
            self.time_index[end_index].value,
            dtype=torch.int64,
        )
        sample.window_index = torch.tensor(index, dtype=torch.int64)
        return sample


class SensorGraphWindowDataset(Dataset[Data]):
    """Graph windows backed by a memory-mapped sensor-space AQ cache."""

    def __init__(
        self,
        *,
        cache: AQSensorCache,
        context_sensor_mask: ArrayLike,
        forbidden_context_mask: ArrayLike,
        hours: int,
        split_start: str | pd.Timestamp,
        split_end: str | pd.Timestamp,
        target_sensor_mask: ArrayLike | None = None,
        target_fraction: float | None = None,
        normalization_mean: ArrayLike | None = None,
        normalization_std: ArrayLike | None = None,
        include_time_features: bool = False,
        seed: int = 42,
    ) -> None:
        if hours <= 0:
            raise ValueError("hours must be positive")
        if (target_sensor_mask is None) == (target_fraction is None):
            raise ValueError("provide exactly one of target_sensor_mask or target_fraction")
        grid_shape = cache.grid_shape
        context = GraphWindowDataset._spatial_mask(
            context_sensor_mask, "context_sensor_mask", grid_shape
        )
        forbidden = GraphWindowDataset._spatial_mask(
            forbidden_context_mask, "forbidden_context_mask", grid_shape
        )
        if torch.any(context & forbidden):
            raise ValueError("forbidden sensor cells must not appear in context")
        fixed_target = None
        if target_sensor_mask is not None:
            fixed_target = GraphWindowDataset._spatial_mask(
                target_sensor_mask, "target_sensor_mask", grid_shape
            )
            if torch.any(context & fixed_target):
                raise ValueError("context and target sensor masks must be disjoint")
        elif not 0.0 < float(target_fraction) < 1.0:
            raise ValueError("target_fraction must be between 0 and 1")

        self.sensor_cache = cache
        self.graph = cache.graph
        self.time_index = cache.time_index
        self.context_sensor_mask = context
        self.forbidden_context_mask = forbidden
        self.target_sensor_mask = fixed_target
        self.target_fraction = target_fraction
        pollutant_count = len(cache.magnitude_ids)
        if (normalization_mean is None) != (normalization_std is None):
            raise ValueError("normalization_mean and normalization_std must be provided together")
        self.normalization_mean = _normalization_vector(
            normalization_mean, pollutant_count, "normalization_mean"
        )
        self.normalization_std = _normalization_vector(
            normalization_std, pollutant_count, "normalization_std"
        )
        if self.normalization_std is not None and np.any(self.normalization_std <= 0):
            raise ValueError("normalization_std values must be positive")
        self.include_time_features = bool(include_time_features)
        self.hours = int(hours)
        self.seed = int(seed)
        self.epoch = 0
        start, end = pd.Timestamp(split_start), pd.Timestamp(split_end)
        if start > end:
            raise ValueError("split_start must not be after split_end")
        self.end_indices = self._select_end_indices(start, end)
        if not self.end_indices:
            raise ValueError("split contains no complete windows with valid targets")

    def _final_node_availability(self, end_index: int) -> torch.Tensor:
        result = np.zeros((len(self.sensor_cache.magnitude_ids), int(self.graph.num_nodes)), dtype=bool)
        present = self.sensor_cache.availability[end_index]
        for sensor_position, node in enumerate(self.sensor_cache.sensor_node_indices):
            result[:, int(node)] |= present[sensor_position]
        return torch.from_numpy(result.reshape(len(result), *self.sensor_cache.grid_shape))

    def _has_usable_target(self, end_index: int) -> bool:
        availability = self._final_node_availability(end_index)
        if self.target_sensor_mask is not None:
            return bool(torch.any(availability & self.target_sensor_mask[None, ...]))
        eligible = availability.any(dim=0) & self.context_sensor_mask
        return int(eligible.sum()) >= 2

    def _select_end_indices(self, start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
        one_hour_ns = pd.Timedelta(hours=1).value
        result = []
        for end_index in range(self.hours - 1, len(self.time_index)):
            first = end_index - self.hours + 1
            times = self.time_index[first : end_index + 1]
            if times[0] < start or times[-1] > end:
                continue
            if len(times) > 1 and not np.all(np.diff(times.asi8) == one_hour_ns):
                continue
            if self._has_usable_target(end_index):
                result.append(end_index)
        return result

    @property
    def window_end_times(self) -> pd.DatetimeIndex:
        return self.time_index[self.end_indices]

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must not be negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, index: int) -> Data:
        end_index = self.end_indices[index]
        start_index = end_index - self.hours + 1
        pollution = _cache_to_node_window(
            self.sensor_cache,
            start_index,
            end_index + 1,
            normalization_mean=self.normalization_mean,
            normalization_std=self.normalization_std,
        )
        extra_features = None
        if self.include_time_features:
            extra_features = _cyclical_time_features(
                self.time_index[start_index : end_index + 1],
                self.sensor_cache.grid_shape,
            )
        if self.target_sensor_mask is None:
            generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + index)
            context, target = split_training_sensor_masks(
                self.context_sensor_mask,
                pollution[1::2, -1].astype(bool),
                target_fraction=float(self.target_fraction),
                generator=generator,
            )
        else:
            context, target = self.context_sensor_mask, self.target_sensor_mask
        sample = build_graph_window(
            graph=self.graph,
            pollutant_data=pollution,
            context_sensor_mask=context,
            target_sensor_mask=target,
            forbidden_context_mask=self.forbidden_context_mask,
            extra_features=extra_features,
        )
        sample.window_start_ns = torch.tensor(self.time_index[start_index].value, dtype=torch.int64)
        sample.window_end_ns = torch.tensor(self.time_index[end_index].value, dtype=torch.int64)
        sample.window_index = torch.tensor(index, dtype=torch.int64)
        return sample


def _normalization_vector(
    value: ArrayLike | None, pollutant_count: int, name: str
) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (pollutant_count,):
        raise ValueError(f"{name} must have shape ({pollutant_count},)")
    return result


def _cyclical_time_features(
    timestamps: pd.DatetimeIndex,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Return hour/day-of-year sine and cosine channels over the grid."""
    hour_fraction = (
        timestamps.hour.to_numpy(dtype=np.float32)
        + timestamps.minute.to_numpy(dtype=np.float32) / 60.0
    ) / 24.0
    days_in_year = np.where(timestamps.is_leap_year, 366.0, 365.0).astype(np.float32)
    year_fraction = (
        timestamps.dayofyear.to_numpy(dtype=np.float32) - 1.0 + hour_fraction
    ) / days_in_year
    angles = 2.0 * np.pi * np.stack((hour_fraction, year_fraction))
    temporal = np.stack(
        (np.sin(angles[0]), np.cos(angles[0]), np.sin(angles[1]), np.cos(angles[1]))
    ).astype(np.float32)
    return np.broadcast_to(
        temporal[:, :, None, None],
        (4, len(timestamps), *grid_shape),
    ).copy()


def _cache_to_node_window(
    cache: AQSensorCache,
    start: int,
    end: int,
    *,
    normalization_mean: np.ndarray | None = None,
    normalization_std: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate colocated sensors by mean and return interleaved grid channels."""
    window_values = np.asarray(cache.values[start:end], dtype=np.float32)
    window_availability = cache.availability[start:end]
    if normalization_mean is not None:
        window_values = (window_values - normalization_mean[None, None, :]) / normalization_std[
            None, None, :
        ]
    time_count, _, pollutant_count = window_values.shape
    height, width = cache.grid_shape
    node_count = height * width
    values = np.zeros((pollutant_count, time_count, node_count), dtype=np.float32)
    availability = np.zeros_like(values, dtype=np.float32)
    for node in np.unique(cache.sensor_node_indices):
        sensor_positions = np.flatnonzero(cache.sensor_node_indices == node)
        present = window_availability[:, sensor_positions, :]
        count = present.sum(axis=1)
        summed = np.where(present, window_values[:, sensor_positions, :], 0.0).sum(axis=1)
        valid = count > 0
        aggregated = np.divide(summed, count, out=np.zeros_like(summed), where=valid)
        values[:, :, int(node)] = aggregated.T
        availability[:, :, int(node)] = valid.T
    interleaved = np.empty((2 * pollutant_count, time_count, height, width), dtype=np.float32)
    interleaved[0::2] = values.reshape(pollutant_count, time_count, height, width)
    interleaved[1::2] = availability.reshape(pollutant_count, time_count, height, width)
    return interleaved


__all__ = ["GraphWindowDataset", "SensorGraphWindowDataset"]
