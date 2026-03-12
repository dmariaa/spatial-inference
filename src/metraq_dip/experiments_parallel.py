from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
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


def _ensure_base_files(
    *,
    config_base: Optional[dict],
    output_folder: str,
    experiment_name: Optional[str],
) -> tuple[dict[str, Any], str, np.ndarray, np.ndarray, pd.DataFrame]:
    experiment_name = experiment_name if experiment_name else f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_output_folder = os.path.join(output_folder, experiment_name)

    if os.path.exists(experiment_output_folder) and os.path.exists(os.path.join(experiment_output_folder, "config.yaml")):
        click.echo(f"Experiment {experiment_name} already exists, loading configuration")
        with open(os.path.join(experiment_output_folder, "config.yaml"), "r", encoding="utf-8") as file:
            saved_config = yaml.safe_load(file)
            if config_base:
                saved_config.update(config_base)
            config_base = saved_config
    else:
        if config_base is None:
            raise ValueError("config_base cannot be None when creating a new experiment session.")
        click.echo(f"Experiment {experiment_name} not found, generating a new one")
        os.makedirs(experiment_output_folder, exist_ok=True)
        with open(os.path.join(experiment_output_folder, "config.yaml"), "w", encoding="utf-8") as file:
            yaml.dump(config_base, file)

    data_file = os.path.join(experiment_output_folder, "data.npz")
    if os.path.exists(data_file):
        click.echo(f"Experiment {experiment_name} already has data, skipping data generation")
        data = np.load(data_file, allow_pickle=True)
        test_sensors = data["test_sensors"]
        time_windows = data["time_windows"]
    else:
        click.echo(f"Generating data for experiment {experiment_name}")
        pollutants = config_base["pollutants"]
        test_sensors, _ = get_spread_test_groups(
            n_groups=10,
            group_size=4,
            max_uses_per_sensor=2,
            magnitudes=pollutants,
        )
        time_windows = get_random_time_windows(year=2024, windows_per_month=20, start_hours=(8, 17))
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


def run_experiments(
    *,
    config_base: Optional[dict] = None,
    output_folder: str = "output",
    experiment_name: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> None:
    config_base, experiment_output_folder, test_sensors, time_windows, df = _ensure_base_files(
        config_base=config_base,
        output_folder=output_folder,
        experiment_name=experiment_name,
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

    click.echo(f"Running {len(jobs)} experiments with {workers} worker(s).")
    results_file = os.path.join(experiment_output_folder, "results.csv")
    failures: list[tuple[str, str, str]] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single_experiment, **job): job for job in jobs}
        with tqdm(total=len(jobs), position=0, desc="Experiments") as pbar:
            for future in as_completed(futures):
                job = futures[future]
                sensor_group_key = job["sensor_group_key"]
                time_window_iso = job["time_window_iso"]

                try:
                    row_result = future.result()
                except Exception as exc:
                    failures.append((sensor_group_key, time_window_iso, str(exc)))
                    pbar.set_postfix({"failed": len(failures)})
                else:
                    mask = (df["time_window"] == row_result["time_window"]) & (
                        df["sensor_group"] == row_result["sensor_group"]
                    )
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
                    df.to_csv(results_file, index=False)
                    pbar.set_postfix({"failed": len(failures)})
                finally:
                    pbar.update(1)

    if failures:
        click.echo(f"{len(failures)} experiment(s) failed. Their rows remain with processed=False.")
        for sensor_group_key, time_window_iso, message in failures[:10]:
            click.echo(f"- {sensor_group_key} @ {time_window_iso}: {message}")


if __name__ == "__main__":
    from pathlib import Path


    @click.command()
    @click.option(
        "--config-file",
        default=None,
        type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
        help="Path to a YAML config file.",
    )
    @click.option(
        "--output-folder",
        default="output/experiments",
        show_default=True,
        type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
        help="Base folder where experiment sessions are stored.",
    )
    @click.option(
        "--experiment-name",
        default=None,
        help="Experiment session name. If omitted, a timestamped name is generated.",
    )
    @click.option(
        "--max-workers",
        default=None,
        type=click.IntRange(min=1),
        help="Number of parallel workers to use. Defaults to CPU count.",
    )
    def cli(
            config_file: Optional[Path],
            output_folder: Path,
            experiment_name: Optional[str],
            max_workers: Optional[int],
    ) -> None:
        config_base: Optional[dict[str, Any]] = None
        if config_file is not None:
            with config_file.open("r", encoding="utf-8") as file:
                loaded = yaml.safe_load(file)

            if loaded is None:
                config_base = {}
            elif isinstance(loaded, dict):
                config_base = loaded
            else:
                raise click.ClickException("Configuration file must contain a YAML mapping at the top level.")

        try:
            run_experiments(
                config_base=config_base,
                output_folder=str(output_folder),
                experiment_name=experiment_name,
                max_workers=max_workers,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc


    cli()
