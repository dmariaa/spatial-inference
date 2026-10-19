"""Frozen-checkpoint sensitivity to losing AQ context stations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from metraq_dip.tools.interpolator import IdwInterpolator, KrigingInterpolator

LEVELS = (20, 12, 8, 4)
METHODS = ("KERNEL_WIND", "KRG", "IDW")
KEYS = ["sensor_group", "window_end", "sensor_id"]


def subset_plans(context, *, seed, group_index, repeats=5, levels=LEVELS):
    """Choose geometry-only nested sets, fixed across all timestamps and methods."""
    context = np.asarray(context, dtype=bool)
    nodes = np.flatnonzero(context)
    if len(nodes) != levels[0] or repeats < 1 or min(levels) < 2:
        raise ValueError("source levels must start with the full context and retain at least two")
    if tuple(sorted(set(levels), reverse=True)) != tuple(levels):
        raise ValueError("source levels must be distinct and decreasing")
    plans = [dict(repeat=-1, source_count=len(nodes), nodes=nodes.tolist())]
    for repeat in range(repeats):
        rng = np.random.default_rng(np.random.SeedSequence([seed, group_index, repeat]))
        order = rng.permutation(nodes)
        for count in levels[1:]:
            plans.append(dict(repeat=repeat, source_count=count,
                              nodes=sorted(order[:count].tolist())))
    return plans


class RestrictedContext(Dataset):
    """Mask AQ and wind at removed sources without changing TEST labels or dates."""
    def __init__(self, samples, nodes):
        self.samples = samples
        self.nodes = tuple(nodes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self.samples[index].clone()
        keep = torch.zeros(data.num_nodes, dtype=torch.bool)
        keep[list(self.nodes)] = True
        original = data.context_sensor_mask.bool()
        if (keep & ~original).any():
            raise ValueError("selected sources include a forbidden or target node")
        removed = original & ~keep
        data.x[removed] = 0.
        data.context_mask[removed] = False
        data.context_sensor_mask = keep
        data.forbidden_context_mask = data.forbidden_context_mask | removed
        data.y[removed] = 0.
        if hasattr(data, "wind_available"):
            data.wind_available[removed] = False
            data.wind_uv[removed] = 0.
        if (data.context_mask[:, -1].bool() & data.target_mask.bool()).any():
            raise ValueError("TEST targets are visible")
        if int(data.context_mask[:, -1].any(dim=-1).sum()) < 2:
            raise ValueError("fewer than two actual AQ sources; do not silently drop this case")
        return data


def classical_predictions(data, grid_shape, mean, std):
    """Both classical methods consume exactly the kernel's visible AQ observations."""
    if data.context_mask.shape[-1] != 1:
        raise ValueError("sparsity experiment requires a single pollutant")
    source = data.context_mask[:, -1, 0].numpy().astype(bool)
    if np.count_nonzero(source) < 2:
        raise ValueError("at least two actual sources are required")
    physical = data.x[:, -1, 0].numpy().astype(np.float64) * float(std[0]) + float(mean[0])
    values = np.where(source, physical, 0.).reshape(grid_shape)
    mask = source.reshape(grid_shape)
    target_nodes = data.target_mask[:, 0].nonzero().flatten().numpy()
    rows, cols = np.unravel_index(target_nodes, grid_shape)
    result = {}
    for label, method in (("KRG", KrigingInterpolator), ("IDW", IdwInterpolator)):
        prediction = np.asarray(method(values[None], mask[None])(
            cols.astype(float), rows.astype(float), mode="points"), dtype=float).reshape(-1)
        if not np.isfinite(prediction).all():
            raise ValueError(f"{label} returned non-finite predictions; all targets must be retained")
        result[label] = prediction
    return target_nodes, result, int(source.sum())


def canonicalize(frame, truth):
    """Check one scenario/method against every original held-out observation."""
    if frame.duplicated(KEYS).any() or truth.duplicated(KEYS).any():
        raise ValueError("duplicate target keys")
    merged = frame.merge(truth[KEYS + ["target"]].rename(columns={"target": "canonical_target"}),
                         on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("target keys differ")
    if not np.isfinite(merged[["target", "canonical_target", "prediction"]]).all().all():
        raise ValueError("non-finite targets or predictions")
    tolerance = 1e-5 + np.finfo(np.float32).eps * merged.canonical_target.abs()
    if ((merged.target - merged.canonical_target).abs() > tolerance).any():
        raise ValueError("target values differ")
    merged["target"] = merged.pop("canonical_target")
    merged["error"] = merged.prediction - merged.target
    merged["absolute_error"] = merged.error.abs()
    merged["squared_error"] = merged.error ** 2
    return merged.drop(columns="_merge")


def summarize_sparse(frame, groups):
    records = []
    for keys, rows in frame.groupby(groups, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        denominator = rows.target.abs().sum()
        records.append(dict(zip(groups, keys), count=len(rows),
            mae=float(rows.absolute_error.mean()), rmse=float(np.sqrt(rows.squared_error.mean())),
            wape=float(rows.absolute_error.sum()/denominator) if denominator else np.nan))
    return pd.DataFrame(records)
