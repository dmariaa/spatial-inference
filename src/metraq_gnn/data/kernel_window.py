"""Preloaded current-hour AQ samples in the standard PyG Data format."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from metraq_gnn.data.window import split_training_sensor_masks


class CachedKernelDataset(Dataset):
    """Preload permitted current-hour observations, without histories or edges.

    Window eligibility and TRAIN masking match SensorGraphWindowDataset. Values
    are averaged per cell *after* per-sensor normalization, as in the baseline.
    No values or availability from forbidden, non-target nodes are loaded.
    """
    def __init__(
        self, *, cache, context_sensor_mask, forbidden_context_mask, hours,
        split_start, split_end, target_sensor_mask=None, target_fraction=None,
        normalization_mean=None, normalization_std=None, seed=42, wind_cache=None,
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

        self.positions = cache.graph.pos
        self.forbidden = torch.from_numpy(forbidden.copy()).reshape(-1)
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
        self.wind_nodes = None
        self.wind_coverage = None
        if wind_cache is not None:
            wind_values, wind_available = wind_cache.align(cache, self.window_end_times)
            self.wind_values = torch.from_numpy(wind_values)
            self.wind_available = torch.from_numpy(wind_available)
            self.wind_nodes = torch.as_tensor(np.asarray(wind_cache.node_indices).copy(), dtype=torch.long)
            context_columns = context.reshape(-1)[np.asarray(wind_cache.node_indices)]
            self.wind_coverage = float(wind_available[:, context_columns].mean())

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
        visible_values = torch.where(source_mask, values, 0.)
        data = Data(
            x=torch.cat((visible_values, source_mask.float()), dim=-1).unsqueeze(1),
            context_mask=source_mask.unsqueeze(1),
            pos=self.positions,
            y=values,
            target_mask=target_mask,
            context_sensor_mask=context.reshape(-1),
            target_sensor_mask=target.reshape(-1),
            forbidden_context_mask=self.forbidden,
            window_end_ns=torch.tensor(self.window_end_times[index].value, dtype=torch.int64),
            window_index=torch.tensor(index, dtype=torch.int64),
            num_nodes=self.node_count,
            grid_shape=self.grid_shape,
        )
        if self.wind_nodes is not None:
            data.wind_uv = torch.zeros((self.node_count, 2))
            data.wind_available = torch.zeros(self.node_count, dtype=torch.bool)
            # Only context AQ sources can use wind, independent of target labels.
            wind_mask = self.wind_available[index] & context.reshape(-1)[self.wind_nodes]
            data.wind_uv[self.wind_nodes] = torch.where(wind_mask[:, None], self.wind_values[index], 0.)
            data.wind_available[self.wind_nodes] = wind_mask
        return data
