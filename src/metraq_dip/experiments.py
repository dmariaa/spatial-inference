from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Optional

import click
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from metraq_dip.tools.random_tools import (
    get_random_time_windows,
    get_spread_test_groups,
    sensor_group_hash,
)
from metraq_dip.tools.tools import get_interpolation_loss
from metraq_dip.trainer.trainer_dip import DipTrainer


def get_experiment_name(sensor_group_key: str, time_window: datetime) -> str:
    return f"exp_{sensor_group_key}_{time_window.strftime('%Y%m%dT%H%M%S')}"


def _load_config_for_training(config_file: Path) -> dict[str, Any]:
    with config_file.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if loaded is None:
        raise ValueError("Configuration file is empty. Please provide a YAML mapping with the experiment settings.")
    if not isinstance(loaded, dict):
        raise ValueError("Configuration file must contain a YAML mapping at the top level.")
    if "pollutants" not in loaded:
        raise ValueError("Configuration file must define a top-level 'pollutants' field.")
    return loaded


def _validate_config(
    config_base: Optional[dict[str, Any]]
) -> tuple[dict[str, int], dict[str, Any]]:
    loaded = config_base

    if "spread_test_groups" not in loaded:
        raise ValueError(
            "Configuration file must define a top-level 'spread_test_groups' mapping "
            "with n_groups, group_size and max_uses_per_sensor."
        )
    if "random_time_windows" not in loaded:
        raise ValueError(
            "Configuration file must define a top-level 'random_time_windows' mapping "
            "with year, windows_per_month and start_hours."
        )

    spread_test_groups_cfg = loaded["spread_test_groups"]
    random_time_windows_cfg = loaded["random_time_windows"]
    if not isinstance(spread_test_groups_cfg, dict):
        raise ValueError("'spread_test_groups' must be a mapping.")
    if not isinstance(random_time_windows_cfg, dict):
        raise ValueError("'random_time_windows' must be a mapping.")

    spread_required = ("n_groups", "group_size", "max_uses_per_sensor")
    missing_spread = [k for k in spread_required if k not in spread_test_groups_cfg]
    if missing_spread:
        raise ValueError(
            f"'spread_test_groups' is missing required field(s): {', '.join(missing_spread)}."
        )

    time_required = ("year", "windows_per_month", "start_hours")
    missing_time = [k for k in time_required if k not in random_time_windows_cfg]
    if missing_time:
        raise ValueError(
            f"'random_time_windows' is missing required field(s): {', '.join(missing_time)}."
        )

    spread_test_groups_params = {
        "n_groups": int(spread_test_groups_cfg["n_groups"]),
        "group_size": int(spread_test_groups_cfg["group_size"]),
        "max_uses_per_sensor": int(spread_test_groups_cfg["max_uses_per_sensor"]),
    }

    start_hours_cfg = random_time_windows_cfg["start_hours"]
    if not isinstance(start_hours_cfg, list) or not start_hours_cfg:
        raise ValueError("'random_time_windows.start_hours' must be a non-empty list of hours.")

    random_time_windows_params: dict[str, Any] = {
        "year": int(random_time_windows_cfg["year"]),
        "windows_per_month": int(random_time_windows_cfg["windows_per_month"]),
        "start_hours": [int(hour) for hour in start_hours_cfg],
    }
    if "weekend_fraction" in random_time_windows_cfg:
        random_time_windows_params["weekend_fraction"] = float(random_time_windows_cfg["weekend_fraction"])

    return spread_test_groups_params, random_time_windows_params


def _ensure_base_files(
    *,
    config_file: Path,
) -> tuple[dict[str, Any], str, np.ndarray, np.ndarray, pd.DataFrame]:
    experiment_output_folder = str(config_file.parent)
    config_base = _load_config_for_training(config_file)

    data_file = os.path.join(experiment_output_folder, "data.npz")
    if os.path.exists(data_file):
        click.echo(f"Session at {experiment_output_folder} already has data, skipping data generation")
        data = np.load(data_file, allow_pickle=True)
        test_sensors = data["test_sensors"]
        time_windows = data["time_windows"]
    else:
        spread_test_groups_params, random_time_windows_params = _validate_config(config_base)
        click.echo(f"Generating data for session at {experiment_output_folder}")
        pollutants = config_base["pollutants"]
        test_sensors, _ = get_spread_test_groups(
            n_groups=spread_test_groups_params["n_groups"],
            group_size=spread_test_groups_params["group_size"],
            max_uses_per_sensor=spread_test_groups_params["max_uses_per_sensor"],
            magnitudes=pollutants,
        )
        time_windows = get_random_time_windows(**random_time_windows_params)
        np.savez(data_file, test_sensors=test_sensors, time_windows=time_windows)

    results_file = os.path.join(experiment_output_folder, "results.csv")
    if os.path.exists(results_file):
        df = pd.read_csv(results_file, dtype={"sensor_group": "string"}, parse_dates=["time_window"])
    else:
        time_window_series = pd.to_datetime(time_windows)
        sensor_group_keys = [sensor_group_hash(group) for group in test_sensors]
        rows = [(time_window, sensor_group) for sensor_group in sensor_group_keys for time_window in time_window_series]
        df = pd.DataFrame(rows, columns=["time_window", "sensor_group"])
        df["sensor_group"] = df["sensor_group"].astype("string")
        df["processed"] = False
        df["DIP_L1Loss"] = 0.0
        df["DIP_MSELoss"] = 0.0
        df["KRG_L1Loss"] = 0.0
        df["KRG_MSELoss"] = 0.0
        df["IDW_L1Loss"] = 0.0
        df["IDW_MSELoss"] = 0.0
        df.to_csv(results_file, index=False)

    return config_base, experiment_output_folder, test_sensors, time_windows, df


def _run_single_experiment(
    *,
    config_base: dict[str, Any],
    experiment_output_folder: str,
    test_sensor_group: list[int],
    sensor_group_key: str,
    time_window_iso: str,
) -> dict[str, Any]:
    # Silence nested trainer progress bars in worker processes.
    import metraq_dip.trainer.trainer_dip as trainer_dip_module

    trainer_dip_module.tqdm = partial(tqdm, disable=True)
    time_window_dt = pd.to_datetime(time_window_iso).to_pydatetime()

    config = config_base.copy()
    config["date"] = time_window_dt.isoformat()
    config["validation_sensors"] = 4
    config["test_sensors"] = test_sensor_group

    trainer = DipTrainer(configuration=config)
    trainer()
    result, output, min_idx_s = trainer.get_best_result()

    x_data = (trainer.dip_logger["train_data"] + trainer.dip_logger["val_data"])[0, :, -1:, ...].detach().cpu()
    y_data = trainer.dip_logger["test_data"][0, :, -1:, ...].detach().cpu()
    train_mask = (trainer.dip_logger["train_mask"] + trainer.dip_logger["val_mask"])[0, :, -1:, ...].detach().cpu()
    test_mask = trainer.dip_logger["test_mask"][0, :, -1:, ...].detach().cpu()

    interpolation_results = get_interpolation_loss(x_data, train_mask, y_data, test_mask, trainer.pollutants)

    experiment_data = {
        "train_data": trainer.dip_logger["train_data"].detach().cpu().numpy(),
        "val_data": trainer.dip_logger["val_data"].detach().cpu().numpy(),
        "test_data": trainer.dip_logger["test_data"].detach().cpu().numpy(),
        "train_mask": trainer.dip_logger["train_mask"].detach().cpu().numpy(),
        "val_mask": trainer.dip_logger["val_mask"].detach().cpu().numpy(),
        "test_mask": trainer.dip_logger["test_mask"].detach().cpu().numpy(),
        "train_output": output,
        "val_min_idx": min_idx_s,
        "train_k_output": trainer.dip_logger["train_output"].detach().cpu().numpy(),
        "train_k_loss": trainer.dip_logger["train_loss"].detach().cpu().numpy(),
        "val_k_loss": trainer.dip_logger["val_loss"].detach().cpu().numpy(),
        "test_k_loss": trainer.dip_logger["test_loss"].detach().cpu().numpy(),
    }
    experiment_file_name = f"{get_experiment_name(sensor_group_key, time_window_dt)}.npz"
    np.savez_compressed(os.path.join(experiment_output_folder, experiment_file_name), **experiment_data)

    return {
        "sensor_group": sensor_group_key,
        "time_window": pd.Timestamp(time_window_iso),
        "DIP_L1Loss": result[0]["loss"],
        "DIP_MSELoss": result[1]["loss"],
        "KRG_L1Loss": interpolation_results[0]["loss"],
        "KRG_MSELoss": interpolation_results[1]["loss"],
        "IDW_L1Loss": interpolation_results[2]["loss"],
        "IDW_MSELoss": interpolation_results[3]["loss"],
        "processed": True,
    }


def _apply_row_result(df: pd.DataFrame, row_result: dict[str, Any]) -> None:
    mask = (df["time_window"] == row_result["time_window"]) & (df["sensor_group"] == row_result["sensor_group"])
    df.loc[
        mask,
        [
            "sensor_group",
            "processed",
            "DIP_L1Loss",
            "DIP_MSELoss",
            "KRG_L1Loss",
            "KRG_MSELoss",
            "IDW_L1Loss",
            "IDW_MSELoss",
        ],
    ] = [
        row_result["sensor_group"],
        row_result["processed"],
        row_result["DIP_L1Loss"],
        row_result["DIP_MSELoss"],
        row_result["KRG_L1Loss"],
        row_result["KRG_MSELoss"],
        row_result["IDW_L1Loss"],
        row_result["IDW_MSELoss"],
    ]


def run_experiments(
    *,
    config_file: Path,
    max_workers: Optional[int] = None,
) -> None:
    config_base, experiment_output_folder, test_sensors, time_windows, df = _ensure_base_files(
        config_file=config_file,
    )

    jobs: list[dict[str, Any]] = []
    for test_sensor_group in test_sensors:
        sensor_group_key = sensor_group_hash(test_sensor_group)
        for time_window in time_windows:
            time_window_ts = pd.to_datetime(time_window)
            row_mask = (df["time_window"] == time_window_ts) & (df["sensor_group"] == sensor_group_key)
            if not df.loc[row_mask].empty and df.loc[row_mask, "processed"].any():
                continue

            jobs.append(
                {
                    "config_base": config_base,
                    "experiment_output_folder": experiment_output_folder,
                    "test_sensor_group": list(test_sensor_group),
                    "sensor_group_key": sensor_group_key,
                    "time_window_iso": time_window_ts.isoformat(),
                }
            )

    if not jobs:
        click.echo("All experiments are already processed.")
        return

    workers = max_workers if max_workers is not None else os.cpu_count() or 1
    workers = max(1, workers)
    run_in_parallel = max_workers is not None and workers > 1

    if run_in_parallel:
        click.echo(f"Running {len(jobs)} experiments in parallel with {workers} worker(s).")
    else:
        mode = "sequentially" if max_workers is None else "sequentially (max_workers=1)"
        click.echo(f"Running {len(jobs)} experiments {mode}.")

    results_file = os.path.join(experiment_output_folder, "results.csv")
    failures: list[tuple[str, str, str]] = []

    with tqdm(total=len(jobs), position=0, desc="Experiments") as pbar:
        if run_in_parallel:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_single_experiment, **job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    sensor_group_key = job["sensor_group_key"]
                    time_window_iso = job["time_window_iso"]

                    try:
                        row_result = future.result()
                    except Exception as exc:
                        failures.append((sensor_group_key, time_window_iso, str(exc)))
                    else:
                        _apply_row_result(df, row_result)
                        df.to_csv(results_file, index=False)
                    finally:
                        pbar.set_postfix({"failed": len(failures)})
                        pbar.update(1)
        else:
            for job in jobs:
                sensor_group_key = job["sensor_group_key"]
                time_window_iso = job["time_window_iso"]
                try:
                    row_result = _run_single_experiment(**job)
                except Exception as exc:
                    failures.append((sensor_group_key, time_window_iso, str(exc)))
                else:
                    _apply_row_result(df, row_result)
                    df.to_csv(results_file, index=False)
                finally:
                    pbar.set_postfix({"failed": len(failures)})
                    pbar.update(1)

    if failures:
        click.echo(f"{len(failures)} experiment(s) failed. Their rows remain with processed=False.")
        for sensor_group_key, time_window_iso, message in failures[:10]:
            click.echo(f"- {sensor_group_key} @ {time_window_iso}: {message}")


if __name__ == "__main__":
    @click.command(no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})
    @click.option(
        "--config-file",
        required=True,
        type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
        help="Path to a YAML config file.",
    )
    @click.option(
        "--max-workers",
        default=None,
        type=click.IntRange(min=1),
        help="Number of parallel workers to use. Defaults to CPU count.",
    )
    def cli(
            config_file: Path,
            max_workers: Optional[int],
    ) -> None:
        try:
            run_experiments(
                config_file=config_file,
                max_workers=max_workers,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc


    cli()
