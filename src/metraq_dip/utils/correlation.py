from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from metraq_dip.tools.tools import calculate_interpolations
from metraq_dip.tools.interpolator import (
    IdwInterpolator,
    KrigingInterpolator,
    NearestNeighborInterpolator,
)

SESSION = Path(r"output/experiments/airparif/spatial")  # change this
UNNORMALIZE = True

def load_pollutants(session):
    config_file = session / "config.yaml"
    with config_file.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return [int(pollutant) for pollutant in config.get("pollutants", [])]

def denormalize_masked(data, mask, normalization_stats, pollutants):
    restored = np.asarray(data, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)

    if restored.shape != mask.shape:
        raise ValueError("data and mask must have the same shape")
    if restored.shape[0] != len(pollutants):
        raise ValueError("number of pollutants must match data channel count")

    for channel_idx, pollutant in enumerate(pollutants):
        mean, std = normalization_stats[int(pollutant)]
        channel_mask = mask[channel_idx]
        restored[channel_idx][channel_mask] = restored[channel_idx][channel_mask] * (std + 1e-6) + mean
        restored[channel_idx][~channel_mask] = 0.0

    return restored

def denormalize_full(data, normalization_stats, pollutants):
    restored = np.asarray(data, dtype=np.float32).copy()

    if restored.ndim == 2:
        if len(pollutants) != 1:
            raise ValueError("2D prediction surfaces only support one pollutant")
        mean, std = normalization_stats[int(pollutants[0])]
        return restored * (std + 1e-6) + mean

    if restored.shape[0] != len(pollutants):
        raise ValueError("number of pollutants must match prediction channel count")

    for channel_idx, pollutant in enumerate(pollutants):
        mean, std = normalization_stats[int(pollutant)]
        restored[channel_idx] = restored[channel_idx] * (std + 1e-6) + mean

    return restored

def align_prediction_to_target(prediction, target_shape):
    prediction = np.asarray(prediction, dtype=np.float32)

    if prediction.shape == target_shape:
        return prediction

    if prediction.ndim == 2 and prediction.shape == target_shape[-2:]:
        if target_shape[0] != 1 or target_shape[1] != 1:
            raise ValueError("2D predictions can only align to one pollutant and one timestamp")
        return prediction[None, None, ...]

    if prediction.ndim == 3 and prediction.shape == (target_shape[0], *target_shape[-2:]):
        if target_shape[1] != 1:
            raise ValueError("3D predictions can only align to one timestamp")
        return prediction[:, None, ...]

    raise ValueError(f"Cannot align prediction shape {prediction.shape} to target shape {target_shape}")

def metric_row(file_name, method, target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    ok = np.isfinite(target) & np.isfinite(prediction)

    n_valid = int(ok.sum())
    if n_valid == 0:
        return {
            "file": file_name,
            "method": method,
            "pearson": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "n_valid": 0,
        }

    y = target[ok]
    y_hat = prediction[ok]
    error = y_hat - y

    pearson = np.nan
    if n_valid >= 2 and np.std(y) > 0 and np.std(y_hat) > 0:
        pearson = float(np.corrcoef(y, y_hat)[0, 1])

    return {
        "file": file_name,
        "method": method,
        "pearson": pearson,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "n_valid": n_valid,
    }

rows = []
all_values = {}
pollutants = load_pollutants(SESSION)

for file in sorted(SESSION.glob("exp_*.npz")):
    with np.load(file, allow_pickle=True) as z:
        # First ensemble member matches the current baseline-loss code path.
        train_data = z["train_data"][0]
        val_data = z["val_data"][0]
        test_data = z["test_data"][0]

        train_mask = z["train_mask"][0]
        val_mask = z["val_mask"][0]
        test_mask = z["test_mask"][0]

        normalization_stats = None
        if UNNORMALIZE and "normalization_stats" in z:
            normalization_stats = z["normalization_stats"].item()
            train_data = denormalize_masked(train_data, train_mask, normalization_stats, pollutants)
            val_data = denormalize_masked(val_data, val_mask, normalization_stats, pollutants)
            test_data = denormalize_masked(test_data, test_mask, normalization_stats, pollutants)

        # Literal nearest train sensor value.
        nearest_grid = calculate_interpolations(
            train_data,
            train_mask,
            NearestNeighborInterpolator,
        )

        # Same observed set used by KRG/IDW losses in experiments.py: train + validation.
        observed_mask = train_mask | val_mask
        # observed_data = train_data + val_data
        observed_data = np.where(train_mask, train_data, val_data)

        nearest_train_val_grid = calculate_interpolations(
            observed_data,
            observed_mask,
            NearestNeighborInterpolator,
        )
        idw_grid = calculate_interpolations(
            observed_data,
            observed_mask,
            IdwInterpolator,
        )
        krg_grid = calculate_interpolations(
            observed_data,
            observed_mask,
            KrigingInterpolator,
        )

        if UNNORMALIZE and "train_output_real" in z:
            dip_grid = z["train_output_real"]
        else:
            dip_grid = z["train_output"]
            if UNNORMALIZE and normalization_stats is not None:
                dip_grid = denormalize_full(dip_grid, normalization_stats, pollutants)
        dip_grid = align_prediction_to_target(dip_grid, test_data.shape)

        y = test_data[test_mask]
        predictions = {
            "nearest_train": nearest_grid[test_mask],
            "nearest_train_val": nearest_train_val_grid[test_mask],
            "idw_train_val": idw_grid[test_mask],
            "kriging_train_val": krg_grid[test_mask],
            "dip_final": dip_grid[test_mask],
        }

    for method, prediction in predictions.items():
        rows.append(metric_row(file.name, method, y, prediction))
        bucket = all_values.setdefault(method, {"target": [], "prediction": []})
        bucket["target"].append(y)
        bucket["prediction"].append(prediction)

per_file = pd.DataFrame(rows)
per_file_path = SESSION / "metrics_per_file.csv"
per_file.to_csv(per_file_path, index=False)

overall_rows = []
for method, values in all_values.items():
    overall_rows.append(
        metric_row(
            "ALL",
            method,
            np.concatenate(values["target"]),
            np.concatenate(values["prediction"]),
        )
    )
overall = pd.DataFrame(overall_rows)
overall_path = SESSION / "metrics_overall.csv"
overall.to_csv(overall_path, index=False)

print("Overall metrics")
print(overall.to_string(index=False))
print()
print("Per-file metrics written to:", per_file_path)
print("Overall metrics written to:", overall_path)
