from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


PREDICTION_COLUMNS = (
    "window_end",
    "sensor_id",
    "node_index",
    "grid_x",
    "grid_y",
    "magnitude_id",
    "target",
    "prediction",
    "error",
    "absolute_error",
    "squared_error",
)


def collect_sensor_predictions(
    model: torch.nn.Module,
    loader: Iterable[Data],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device | str,
    sensor_ids: np.ndarray,
    sensor_node_indices: np.ndarray,
    node_positions: np.ndarray,
    magnitude_ids: np.ndarray,
) -> pd.DataFrame:
    """Collect physical-unit predictions while retaining sensor identity."""
    model.eval()
    device = torch.device(device)
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
    node_count = int(len(node_positions))
    node_to_sensor = {
        int(node): int(sensor_id)
        for sensor_id, node in zip(sensor_ids, sensor_node_indices, strict=True)
    }
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch) * std_tensor + mean_tensor
            target = batch.y * std_tensor + mean_tensor
            selected = batch.target_mask.nonzero(as_tuple=False)
            window_ends = batch.window_end_ns.reshape(-1).detach().cpu().numpy()

            for flat_node, channel in selected.detach().cpu().numpy():
                graph_index, node_index = divmod(int(flat_node), node_count)
                sensor_id = node_to_sensor.get(node_index)
                if sensor_id is None:
                    raise ValueError(f"target node {node_index} has no sensor id")
                actual = float(target[flat_node, channel].item())
                estimate = float(prediction[flat_node, channel].item())
                error = estimate - actual
                grid_x, grid_y = node_positions[node_index]
                rows.append(
                    {
                        "window_end": pd.Timestamp(int(window_ends[graph_index])),
                        "sensor_id": sensor_id,
                        "node_index": node_index,
                        "grid_x": float(grid_x),
                        "grid_y": float(grid_y),
                        "magnitude_id": int(magnitude_ids[channel]),
                        "target": actual,
                        "prediction": estimate,
                        "error": error,
                        "absolute_error": abs(error),
                        "squared_error": error * error,
                    }
                )

    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def summarize_sensor_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prediction rows into physical-unit metrics per sensor."""
    if predictions.empty:
        return pd.DataFrame(
            columns=("sensor_id", "magnitude_id", "count", "mae", "rmse", "wape", "bias")
        )

    def summarize(group: pd.DataFrame) -> pd.Series:
        absolute_target = group["target"].abs().sum()
        return pd.Series(
            {
                "count": len(group),
                "mae": group["absolute_error"].mean(),
                "rmse": np.sqrt(group["squared_error"].mean()),
                "wape": (
                    group["absolute_error"].sum() / absolute_target
                    if absolute_target
                    else np.nan
                ),
                "bias": group["error"].mean(),
            }
        )

    records = []
    for (sensor_id, magnitude_id), group in predictions.groupby(
        ["sensor_id", "magnitude_id"], sort=True
    ):
        records.append(
            {
                "sensor_id": sensor_id,
                "magnitude_id": magnitude_id,
                **summarize(group).to_dict(),
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_all_predictions(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate prediction rows using the same definitions as the runner."""
    count = len(predictions)
    absolute_target = predictions["target"].abs().sum()
    return {
        "test_mae_physical": float(predictions["absolute_error"].sum() / count),
        "test_rmse_physical": float(np.sqrt(predictions["squared_error"].sum() / count)),
        "test_wape": float(
            predictions["absolute_error"].sum() / absolute_target
            if absolute_target
            else np.nan
        ),
        "test_target_count": count,
    }


__all__ = [
    "collect_sensor_predictions",
    "summarize_all_predictions",
    "summarize_sensor_predictions",
]
