"""Current-hour AQ snapshots and lightweight batches for convex interpolation."""
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from metraq_gnn.data.window import split_training_sensor_masks


def adapt_kernel_window(sample):
    """Compatibility adapter for the original graph-window implementation."""
    result = sample.clone()
    channels = sample.context_mask.shape[-1]
    result.kernel_values = sample.x[:, -1, :channels].clone()
    result.kernel_available = sample.context_mask[:, -1, :].clone()
    if torch.any(result.kernel_available & sample.target_mask):
        raise ValueError("target AQ cannot be visible")
    if torch.any(result.kernel_available & sample.forbidden_context_mask[:, None]):
        raise ValueError("forbidden AQ cannot be visible")
    if not result.kernel_available.any(dim=0).all():
        raise ValueError("no current visible AQ sources for a sample/channel")
    return result


class KernelWindowDataset(Dataset):
    """Original graph adapter, retained for compatibility and equivalence checks."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return adapt_kernel_window(self.dataset[index])

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)


@dataclass
class KernelBatch:
    kernel_values: torch.Tensor
    kernel_available: torch.Tensor
    y: torch.Tensor
    target_mask: torch.Tensor
    window_end_ns: torch.Tensor

    def to(self, device, non_blocking=False):
        return type(self)(**{
            field.name: getattr(self, field.name).to(device, non_blocking=non_blocking)
            for field in fields(self)
        })

    def pin_memory(self):
        return type(self)(**{
            field.name: getattr(self, field.name).pin_memory() for field in fields(self)
        })


def collate_kernel_snapshots(samples):
    return KernelBatch(**{
        field.name: torch.stack([getattr(sample, field.name) for sample in samples])
        for field in fields(KernelBatch)
    })


class CachedKernelDataset(Dataset):
    """Preload permitted current-hour observations, without histories or edges.

    Window eligibility and TRAIN masking match SensorGraphWindowDataset. Values
    are averaged per cell *after* per-sensor normalization, as in the baseline.
    No values or availability from forbidden, non-target nodes are loaded.
    """
    def __init__(
        self, *, cache, context_sensor_mask, forbidden_context_mask, hours,
        split_start, split_end, target_sensor_mask=None, target_fraction=None,
        normalization_mean=None, normalization_std=None, seed=42,
    ):
        if hours <= 0:
            raise ValueError("hours must be positive")
        if (target_sensor_mask is None) == (target_fraction is None):
            raise ValueError("provide exactly one target mask or target fraction")
        shape = cache.grid_shape
        def spatial(value):
            result = np.asarray(value, dtype=bool)
            if result.shape != shape:
                raise ValueError("sensor mask must match grid shape")
            return result
        context = spatial(context_sensor_mask)
        forbidden = spatial(forbidden_context_mask)
        target = None if target_sensor_mask is None else spatial(target_sensor_mask)
        if (context & forbidden).any() or (target is not None and (context & target).any()):
            raise ValueError("context overlaps forbidden or target cells")
        if target_fraction is not None and not 0 < target_fraction < 1:
            raise ValueError("target_fraction must be between zero and one")
        if (normalization_mean is None) != (normalization_std is None):
            raise ValueError("provide both normalization mean and std")
        channels = len(cache.magnitude_ids)
        if normalization_mean is not None:
            mean = np.asarray(normalization_mean, dtype=np.float32)
            std = np.asarray(normalization_std, dtype=np.float32)
            if mean.shape != (channels,) or std.shape != (channels,):
                raise ValueError("normalization must have one entry per pollutant")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError("normalization statistics must be finite with positive std")

        times = cache.time_index
        start, end = pd.Timestamp(split_start), pd.Timestamp(split_end)
        if start > end:
            raise ValueError("split_start must not be after split_end")
        first = int(times.searchsorted(start))
        stop = int(times.searchsorted(end, side="right"))
        split_times = times[first:stop]
        # Read only sensors eligible as context or labels in this split.
        permitted = context if target is None else context | target
        sensor_nodes = np.asarray(cache.sensor_node_indices)
        sensor_columns = np.flatnonzero(permitted.reshape(-1)[sensor_nodes])
        compact_nodes = np.unique(sensor_nodes[sensor_columns])
        node_count = int(np.prod(shape))
        values = np.asarray(cache.values[first:stop, sensor_columns, :], dtype=np.float32)
        present = np.asarray(cache.availability[first:stop, sensor_columns, :], dtype=bool)
        if normalization_mean is not None:
            values = (values - mean[None, None, :]) / std[None, None, :]
        aggregated = np.zeros((stop-first, len(compact_nodes), channels), dtype=np.float32)
        available = np.zeros_like(aggregated, dtype=bool)
        for compact_index, node in enumerate(compact_nodes):
            columns = np.flatnonzero(sensor_nodes[sensor_columns] == node)
            valid = present[:, columns, :]
            counts = valid.sum(axis=1)
            sums = np.where(valid, values[:, columns, :], 0.).sum(axis=1)
            aggregated[:, compact_index] = np.divide(
                sums, counts, out=np.zeros_like(sums), where=counts > 0)
            available[:, compact_index] = counts > 0

        ends = np.arange(hours - 1, len(split_times))
        if len(ends):
            gaps = np.r_[0, np.cumsum(np.diff(split_times.asi8) != pd.Timedelta(hours=1).value)]
            continuous = gaps[ends] == gaps[ends - hours + 1]
            if target is None:
                eligible = available.any(axis=-1) & context.reshape(-1)[compact_nodes]
                usable = eligible.sum(axis=1) >= 2
            else:
                usable = (available & target.reshape(-1)[compact_nodes][None, :, None]).any(axis=(1, 2))
            ends = ends[continuous & usable[ends]]
        if not len(ends):
            raise ValueError("split contains no complete windows with valid targets")

        self.end_indices = (ends + first).tolist()
        self.window_end_times = times[self.end_indices]
        # Keep only eligible endpoints; sample access performs no cache I/O.
        self.values = torch.from_numpy(aggregated[ends])
        self.available = torch.from_numpy(available[ends])
        self.compact_nodes = torch.as_tensor(compact_nodes, dtype=torch.long)
        self.node_count, self.channels, self.grid_shape = node_count, channels, shape
        self.context = torch.from_numpy(context.copy())
        self.target = None if target is None else torch.from_numpy(target.copy())
        self.target_fraction, self.seed, self.epoch = target_fraction, int(seed), 0

    def __len__(self):
        return len(self.end_indices)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must not be negative")
        self.epoch = int(epoch)

    def __getitem__(self, index):
        values = torch.zeros((self.node_count, self.channels))
        available = torch.zeros_like(values, dtype=torch.bool)
        values[self.compact_nodes] = self.values[index]
        available[self.compact_nodes] = self.available[index]
        if self.target is None:
            generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + index)
            context, target = split_training_sensor_masks(
                self.context, available.T.reshape(self.channels, *self.grid_shape),
                target_fraction=self.target_fraction, generator=generator)
        else:
            context, target = self.context, self.target
        source_mask = available & context.reshape(-1, 1)
        if not source_mask.any(dim=0).all():
            raise ValueError("no current visible AQ sources for a sample/channel")
        target_mask = available & target.reshape(-1, 1)
        return KernelBatch(
            kernel_values=torch.where(source_mask, values, 0.),
            kernel_available=source_mask,
            y=values,
            target_mask=target_mask,
            window_end_ns=torch.tensor(self.window_end_times[index].value, dtype=torch.int64),
        )
