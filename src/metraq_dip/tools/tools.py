from __future__ import annotations

from typing import Any

import numpy as np
import torch

from metraq_dip.tools.interpolator import Interpolator
from metraq_dip.trainer.loss import get_metrics


def calculate_interpolations(x_data: torch.Tensor, x_mask: torch.Tensor, interpolator_class: Interpolator):
    y_hat_grid = np.zeros(x_data.shape)
    channels, timestamps, height, width = x_data.shape

    for c in range(channels):
        for t in range(timestamps):
            f = x_data[c, t].numpy()
            m = x_mask[c, t].numpy()

            # Get known data locations and values
            data = np.array([[i, j, f[i, j]] for i in range(f.shape[0]) for j in range(f.shape[1]) if
                    not np.isnan(f[i, j]) and m[i, j]], dtype=np.float64)
            y, x, z = data[:, 0], data[:, 1], data[:, 2]

            # Generate interpolator
            interpolator = interpolator_class(x, y, z)

            # Get unknown data locations
            data_val = np.array([[i, j] for i in range(f.shape[0]) for j in range(f.shape[1]) if f[i, j] == 0.0],
                            dtype=np.float64)
            y_val, x_val = data_val[:, 0], data_val[:, 1]

            # interpolate
            z_val = interpolator(x_val, y_val, mode="points")

            # Back to the grid
            y_hat_grid[c, t, y.astype(int), x.astype(int)] = z
            y_hat_grid[c, t, y_val.astype(int), x_val.astype(int)] = z_val

    return y_hat_grid


# def calculate_interpolation_loss(x: torch.Tensor,
#                            x_val: torch.Tensor,
#                            val_mask: torch.Tensor,
#                            interpolator_class: Interpolator,
#                            criterion: torch.nn.Module | list[torch.nn.Module]) -> float | list[float]:
#
#     y_hat_interp = calculate_interpolations(x, interpolator_class)
#
#     if isinstance(criterion, list):
#         krig_loss = []
#         for cr in criterion:
#             krig_loss.append(cr(torch.Tensor(y_hat_interp) * val_mask, x_val * val_mask))
#     else:
#         krig_loss = criterion(torch.Tensor(y_hat_interp) * val_mask, x_val * val_mask)
#
#     return krig_loss

def get_interpolation_loss(x: torch.Tensor, x_mask: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor,
                           pollutants: dict[int, str]) -> list[Any]:
    losses = []

    from metraq_dip.tools.interpolator import KrigingInterpolator, IdwInterpolator

    for interpolator in [KrigingInterpolator, IdwInterpolator]:
        y_hat = torch.Tensor(calculate_interpolations(x, x_mask, interpolator))
        loss = get_metrics(y, y_hat, y_mask, pollutants)

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

