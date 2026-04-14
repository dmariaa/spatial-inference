from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DATA_STAT_COLUMNS = (
    "data_mean",
    "data_median",
    "data_max",
    "data_std",
    "data_p90_p10",
)


def get_experiment_name(sensor_group_key: str, time_window: datetime) -> str:
    return f"exp_{sensor_group_key}_{time_window.strftime('%Y%m%dT%H%M%S')}"


def compute_results_data_stats(
    *,
    train_data: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> dict[str, float]:
    train_data_array = np.asarray(train_data, dtype=float)
    observed_mask = np.asarray(train_mask, dtype=bool) | np.asarray(val_mask, dtype=bool)
    observed_data = np.max(train_data_array, axis=0)[observed_mask.any(axis=0)]

    if observed_data.size == 0:
        raise ValueError("Cannot compute results data stats from an empty observed mask.")

    return {
        "data_mean": float(observed_data.mean()),
        "data_median": float(np.median(observed_data)),
        "data_max": float(observed_data.max()),
        "data_std": float(observed_data.std()),
        "data_p90_p10": float(np.percentile(observed_data, 90) - np.percentile(observed_data, 10)),
    }


def ensure_results_stat_columns(df: pd.DataFrame) -> bool:
    updated = False
    for column in RESULTS_DATA_STAT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
            updated = True
    return updated


def load_experiment_result_stats(experiment_file: str | Path) -> dict[str, float]:
    experiment_path = Path(experiment_file)
    with np.load(experiment_path, allow_pickle=True) as experiment_data:
        train_data_key = "x_data" if "x_data" in experiment_data.files else "train_data"
        train_mask = experiment_data["train_mask"]
        val_mask = (
            experiment_data["val_mask"]
            if "val_mask" in experiment_data.files
            else np.zeros_like(train_mask, dtype=bool)
        )
        return compute_results_data_stats(
            train_data=experiment_data[train_data_key],
            train_mask=train_mask,
            val_mask=val_mask,
        )


def backfill_results_stat_columns(
    *,
    df: pd.DataFrame,
    experiment_folder: str | Path,
    row_selector: pd.Series | None = None,
) -> bool:
    if "time_window" not in df.columns or "sensor_group" not in df.columns:
        return False

    updated = False
    session_path = Path(experiment_folder)
    missing_mask = df.loc[:, RESULTS_DATA_STAT_COLUMNS].isna().any(axis=1)
    eligible_mask = missing_mask
    if row_selector is not None:
        eligible_mask = eligible_mask & row_selector.reindex(df.index, fill_value=False).fillna(False).astype(bool)

    for row_index in df.index[eligible_mask]:
        time_window = pd.to_datetime(df.at[row_index, "time_window"], errors="coerce")
        if pd.isna(time_window):
            continue

        sensor_group_key = str(df.at[row_index, "sensor_group"])
        experiment_file = session_path / f"{get_experiment_name(sensor_group_key, time_window.to_pydatetime())}.npz"
        if not experiment_file.exists():
            continue

        try:
            row_stats = load_experiment_result_stats(experiment_file)
        except Exception:
            continue

        for column, value in row_stats.items():
            df.at[row_index, column] = value
        updated = True

    return updated


def validate_results_stat_columns(
    *,
    df: pd.DataFrame,
    row_selector: pd.Series | None = None,
    error_message: str | None = None,
) -> None:
    target_df = (
        df
        if row_selector is None
        else df.loc[row_selector.reindex(df.index, fill_value=False).fillna(False).astype(bool)]
    )
    if target_df.empty:
        return

    missing_mask = target_df.loc[:, RESULTS_DATA_STAT_COLUMNS].isna().any(axis=1)
    missing_count = int(missing_mask.sum())
    if missing_count == 0:
        return

    if error_message is None:
        error_message = (
            f"Results file is missing data stats for {missing_count} row(s). "
            "Ensure the corresponding exp_*.npz files exist or rerun the experiments."
        )

    raise ValueError(error_message)
