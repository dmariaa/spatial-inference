"""Add a fixed-parameter spatiotemporal GP baseline to experiment result copies.

The script never reads ``exp_*.npz`` files and never modifies ``results.csv`` or
``config.yaml``.  It reads only ``config.yaml``, ``data.npz`` (sensor groups and
time windows), and ``results.csv``, then writes a resumable
``results_with_gp.csv`` plus a small aggregate ``gp_summary.csv``.
"""

from __future__ import annotations

import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from metraq_dip.data.aq_backends import get_aq_backend_for_config
from metraq_dip.data.data import collect_data
from metraq_dip.tools.interpolator import SpatioTemporalGPInterpolator
from metraq_dip.tools.random_tools import sensor_group_hash


GP_COLUMNS = ("GP_L1Loss", "GP_MSELoss")


def _load_experiment(experiment_dir: Path):
    config_path = experiment_dir / "config.yaml"
    results_path = experiment_dir / "results.csv"
    data_path = experiment_dir / "data.npz"
    for path in (config_path, results_path, data_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if int(config["hours"]) != 24:
        raise ValueError(f"{config_path} is not a 24-hour experiment")
    # Historical METRAQ configurations predate the explicit backend fields.
    # Apply the former defaults in memory without changing the YAML file.
    config.setdefault("aq_dataset", "metraq")
    config.setdefault("aq_backend", "files")

    # data.npz contains only the experiment-wide sensor groups and timestamps.
    # Per-case exp_*.npz artifacts are deliberately never inspected.
    with np.load(data_path, allow_pickle=True) as index:
        test_groups = [list(map(int, group)) for group in index["test_sensors"]]
    groups_by_key = {sensor_group_hash(group): group for group in test_groups}

    output_path = experiment_dir / "results_with_gp.csv"
    source_path = output_path if output_path.exists() else results_path
    frame = pd.read_csv(
        source_path,
        dtype={"sensor_group": "string"},
        parse_dates=["time_window"],
    )
    for column in GP_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return config, groups_by_key, frame, output_path


def _gp_losses(
    *,
    config: dict,
    test_sensors: list[int],
    target_time: pd.Timestamp,
    spatial_length_scale: float,
    temporal_length_scale: float,
    noise_fraction: float,
) -> tuple[float, float]:
    backend = get_aq_backend_for_config(config)
    end = target_time.to_pydatetime()
    static_data = collect_data(
        start_date=end - timedelta(hours=int(config["hours"]) - 1),
        end_date=end,
        add_meteo=False,
        add_time_channels=False,
        add_coordinates=False,
        add_traffic_data=False,
        pollutants=list(config["pollutants"]),
        test_sensors=test_sensors,
        normalize=False,
        aq_backend=backend,
    )

    pollutant_data = np.asarray(static_data["pollutant_data"])
    values = np.asarray(pollutant_data[::2], dtype=np.float64)
    availability = np.asarray(pollutant_data[1::2], dtype=bool)
    test_locations = np.asarray(static_data["test_mask"], dtype=bool)
    train_mask = availability & ~test_locations[None, None, :, :]

    absolute_errors: list[np.ndarray] = []
    squared_errors: list[np.ndarray] = []
    for channel in range(values.shape[0]):
        target_mask = test_locations & availability[channel, -1] & np.isfinite(values[channel, -1])
        y, x = np.nonzero(target_mask)
        if x.size == 0:
            raise ValueError("no finite test observations at the target hour")

        gp = SpatioTemporalGPInterpolator(
            values[channel],
            train_mask[channel],
            spatial_length_scale=spatial_length_scale,
            temporal_length_scale=temporal_length_scale,
            noise_fraction=noise_fraction,
        )
        prediction = gp(x.astype(float), y.astype(float), mode="points")
        target = values[channel, -1, y, x]
        error = prediction - target
        absolute_errors.append(np.abs(error))
        squared_errors.append(np.square(error))

    return (
        float(np.concatenate(absolute_errors).mean()),
        float(np.concatenate(squared_errors).mean()),
    )


def _write_summary(frame: pd.DataFrame, path: Path) -> None:
    def statistic(column: str, name: str) -> float:
        values = frame[column].dropna()
        return float(getattr(values, name)()) if not values.empty else np.nan

    summary = pd.DataFrame(
        {
            "metric": ["GP_L1Loss", "GP_MSELoss"],
            "count": [frame[column].notna().sum() for column in GP_COLUMNS],
            "mean": [statistic(column, "mean") for column in GP_COLUMNS],
            "median": [statistic(column, "median") for column in GP_COLUMNS],
            "std": [statistic(column, "std") for column in GP_COLUMNS],
        }
    )
    summary.to_csv(path, index=False)


def _run_row(job: dict) -> tuple[int, float | None, float | None, str | None]:
    try:
        l1_loss, mse_loss = _gp_losses(
            config=job["config"],
            test_sensors=job["test_sensors"],
            target_time=job["target_time"],
            spatial_length_scale=job["spatial_length_scale"],
            temporal_length_scale=job["temporal_length_scale"],
            noise_fraction=job["noise_fraction"],
        )
        return job["index"], l1_loss, mse_loss, None
    except Exception:
        return job["index"], None, None, traceback.format_exc()


def process_experiment(
    experiment_dir: Path,
    *,
    limit: int | None,
    max_workers: int,
    spatial_length_scale: float,
    temporal_length_scale: float,
    noise_fraction: float,
) -> tuple[int, int]:
    config, groups_by_key, frame, output_path = _load_experiment(experiment_dir)
    pending = frame[
        frame["processed"].fillna(False).astype(bool)
        & frame.loc[:, list(GP_COLUMNS)].isna().any(axis=1)
    ]
    if limit is not None:
        pending = pending.head(limit)

    completed = 0
    failures = 0
    failure_path = experiment_dir / "gp_failures.log"
    jobs = [
        {
            "index": int(index),
            "config": config,
            "test_sensors": groups_by_key[str(row["sensor_group"])],
            "target_time": pd.Timestamp(row["time_window"]),
            "spatial_length_scale": spatial_length_scale,
            "temporal_length_scale": temporal_length_scale,
            "noise_fraction": noise_fraction,
        }
        for index, row in pending.iterrows()
    ]

    def accept(result: tuple[int, float | None, float | None, str | None]) -> None:
        nonlocal completed, failures
        index, l1_loss, mse_loss, error = result
        if error is None:
            frame.at[index, "GP_L1Loss"] = l1_loss
            frame.at[index, "GP_MSELoss"] = mse_loss
            completed += 1
        else:
            failures += 1
            row = frame.loc[index]
            with failure_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"\n=== {row['sensor_group']} {row['time_window']} ===\n{error}"
                )

        if (completed + failures) % 10 == 0:
            temporary = output_path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, output_path)

    if max_workers == 1:
        iterator = map(_run_row, jobs)
        for result in tqdm(iterator, total=len(jobs), desc=experiment_dir.name):
            accept(result)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_row, job) for job in jobs]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=experiment_dir.name
            ):
                accept(future.result())

    # Always write the final partial or complete state.
    temporary = output_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output_path)

    _write_summary(frame, experiment_dir / "gp_summary.csv")
    if failures == 0 and failure_path.exists():
        failure_path.unlink()
    return completed, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--limit", type=int, help="Process at most N pending rows per experiment.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--spatial-length-scale", type=float, default=3.0)
    parser.add_argument("--temporal-length-scale", type=float, default=6.0)
    parser.add_argument("--noise-fraction", type=float, default=0.1)
    args = parser.parse_args()

    total_completed = 0
    total_failures = 0
    for experiment_dir in args.experiment_dirs:
        completed, failures = process_experiment(
            experiment_dir.resolve(),
            limit=args.limit,
            max_workers=max(1, args.max_workers),
            spatial_length_scale=args.spatial_length_scale,
            temporal_length_scale=args.temporal_length_scale,
            noise_fraction=args.noise_fraction,
        )
        total_completed += completed
        total_failures += failures
        print(f"{experiment_dir}: completed={completed}, failures={failures}")

    print(f"TOTAL: completed={total_completed}, failures={total_failures}")
    if total_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
