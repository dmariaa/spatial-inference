from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from metraq_dip.tools.interpolator import Interpolator


def is_truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def calculate_interpolations(
    x_data: np.ndarray,
    x_mask: np.ndarray,
    interpolator_class: type[Interpolator],
) -> np.ndarray:
    x_data_array = np.asarray(x_data, dtype=np.float32)
    x_mask_array = np.asarray(x_mask, dtype=bool)
    if x_data_array.shape != x_mask_array.shape:
        raise ValueError("x_data and x_mask must have the same shape")

    channels, _, height, width = x_data_array.shape
    y_hat_grid = np.zeros((channels, 1, height, width), dtype=np.float32)

    for c in range(channels):
        interpolator = interpolator_class(x_data_array[c], x_mask_array[c])
        surface = interpolator(
            np.arange(width, dtype=np.float64),
            np.arange(height, dtype=np.float64),
            mode="grid",
        )
        target_data = x_data_array[c, -1]
        target_mask = x_mask_array[c, -1] & np.isfinite(target_data)
        surface[target_mask] = target_data[target_mask]
        y_hat_grid[c, 0] = surface

    return y_hat_grid


def _get_numpy_metrics(
    y: np.ndarray,
    y_hat: np.ndarray,
    mask: np.ndarray,
    pollutants: dict[int, str] | list[int],
) -> list[dict[str, Any]]:
    y_array = np.asarray(y, dtype=np.float32)
    y_hat_array = np.asarray(y_hat, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=bool)
    if y_array.shape != y_hat_array.shape or y_array.shape != mask_array.shape:
        raise ValueError("y, y_hat, and mask must have the same shape")

    result: list[dict[str, Any]] = []
    for channel_idx, pollutant in enumerate(pollutants):
        channel_mask = mask_array[channel_idx]
        count = int(channel_mask.sum())
        if count == 0:
            raise ValueError("mask must contain at least one observed cell per pollutant")

        diff = y_hat_array[channel_idx][channel_mask] - y_array[channel_idx][channel_mask]
        losses = {
            "L1Loss": float(np.abs(diff).sum() / count),
            "MSELoss": float(np.square(diff).sum() / count),
        }

        for loss_name, loss_value in losses.items():
            result.append(
                {
                    "pollutant": pollutant,
                    "criterion": loss_name,
                    "loss": loss_value,
                }
            )

    return result


def get_interpolation_loss(
    x: np.ndarray,
    x_mask: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    pollutants: dict[int, str] | list[int],
    *,
    include_gp: bool = False,
) -> list[Any]:
    losses = []

    from metraq_dip.tools.interpolator import (
        IdwInterpolator,
        KrigingInterpolator,
        SpatioTemporalGPInterpolator,
    )

    interpolators = [KrigingInterpolator, IdwInterpolator]
    if include_gp:
        interpolators.insert(0, SpatioTemporalGPInterpolator)

    for interpolator in interpolators:
        y_hat = calculate_interpolations(x, x_mask, interpolator)
        loss = _get_numpy_metrics(y, y_hat, y_mask, pollutants)

        for l in loss:
            l['model'] = interpolator.__name__
            losses.append(l)

    return losses

def get_loss(x: torch.Tensor, y_hat: torch.Tensor, mask: torch.Tensor,
             pollutants: dict[int, str],
             criteria: torch.nn.Module | list[torch.nn.Module]) -> list[Any]:
    criteria = criteria if isinstance(criteria, list) else [criteria]

    losses = []

    for i, pollutant in enumerate(pollutants):
        for criterion in criteria:
            loss = criterion(y_hat[0, i] * mask[0, i], x[0, i] * mask[0, i])
            losses.append({
                'pollutant': pollutants[pollutant],
                'criterion': criterion.__class__.__name__,
                'loss': loss.item()
            })

    return losses


def get_padding(shape, levels=3, preserve_time=True):
    # shape: (..., D, H, W)
    d, h, w = shape[-3:]

    time_stride = 1 if preserve_time else 2
    spatial_stride = 2

    d_div = time_stride ** levels
    h_div = spatial_stride ** levels
    w_div = spatial_stride ** levels

    d_pad = (d_div - d % d_div) % d_div
    h_pad = (h_div - h % h_div) % h_div
    w_pad = (w_div - w % w_div) % w_div

    return (0, w_pad, 0, h_pad, 0, d_pad)
