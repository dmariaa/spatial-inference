from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Traffic to_grid: discarding .*",
    category=UserWarning,
)

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from metraq_dip.data.data import collect_data
from metraq_dip.tools.config_tools import SessionConfig, load_session_config
from metraq_dip.tools.random_tools import (
    get_all_time_windows,
    get_random_time_windows,
    get_spread_test_groups,
    sensor_group_hash,
)
from metraq_dip.tools.tools import get_interpolation_loss
from metraq_dip.trainer.dip_ensemble_optimizer import DipEnsembleOptimizer, reduce_surface_ensemble


def get_experiment_name(sensor_group_key: str, time_window: datetime) -> str:
    return f"exp_{sensor_group_key}_{time_window.strftime('%Y%m%dT%H%M%S')}"


def _get_time_windows(session_config: SessionConfig) -> list[pd.Timestamp]:
    if session_config.random_time_windows is not None:
        random_time_windows_params = session_config.random_time_windows.model_dump()
        return get_random_time_windows(**random_time_windows_params)

    all_time_windows_config = session_config.all_time_windows
    if all_time_windows_config is None:
        raise ValueError("No time windows strategy configured.")

    all_time_windows_params = all_time_windows_config.model_dump()
    return get_all_time_windows(**all_time_windows_params)


def _ensure_base_files(
    *,
    config_file: Path,
) -> tuple[dict[str, Any], str, np.ndarray, np.ndarray, pd.DataFrame]:
    experiment_output_folder = str(config_file.parent)
    session_config = load_session_config(config_file)
    config_base = session_config.model_dump(
        exclude={"spread_test_groups", "random_time_windows", "all_time_windows"},
    )

    data_file = os.path.join(experiment_output_folder, "data.npz")
    if os.path.exists(data_file):
        click.echo(f"Session at {experiment_output_folder} already has data, skipping data generation")
        data = np.load(data_file, allow_pickle=True)
        test_sensors = data["test_sensors"]
        time_windows = data["time_windows"]
    else:
        spread_test_groups_params = session_config.spread_test_groups
        click.echo(f"Generating data for session at {experiment_output_folder}")
        test_sensors, _ = get_spread_test_groups(
            n_groups=spread_test_groups_params.n_groups,
            group_size=spread_test_groups_params.group_size,
            max_uses_per_sensor=spread_test_groups_params.max_uses_per_sensor,
            magnitudes=session_config.pollutants,
        )
        time_windows = _get_time_windows(session_config)
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


def _denormalize_masked_channels(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    pollutants: list[int],
    normalization_stats: dict[int, tuple[float, float]],
) -> np.ndarray:
    restored = np.array(data, copy=True, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=bool)
    mask_array = np.broadcast_to(mask_array, restored.shape)

    for channel_idx, pollutant in enumerate(pollutants):
        mean, std = normalization_stats[pollutant]
        channel_mask = mask_array[channel_idx]
        restored[channel_idx][channel_mask] = restored[channel_idx][channel_mask] * (std + 1e-6) + mean
        restored[channel_idx][~channel_mask] = 0.0

    return restored


def _compute_masked_losses(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    y_true_array = np.asarray(y_true, dtype=np.float32)
    y_pred_array = np.asarray(y_pred, dtype=np.float32)
    mask_array = np.broadcast_to(np.asarray(mask, dtype=bool), y_true_array.shape)
    count = int(mask_array.sum())
    if count == 0:
        raise ValueError("mask must contain at least one observed cell")

    diff = y_pred_array[mask_array] - y_true_array[mask_array]
    l1_loss = float(np.abs(diff).sum() / count)
    mse_loss = float(np.square(diff).sum() / count)
    return l1_loss, mse_loss


def _build_loss_history_cube(
    *,
    l1_history: np.ndarray,
    mse_history: np.ndarray,
    channels: int,
) -> np.ndarray:
    pair_history = np.stack([l1_history, mse_history], axis=-1).astype(np.float32)
    return np.broadcast_to(pair_history[None, ...], (channels, pair_history.shape[0], 2)).copy()


def _build_test_loss_history_cube(
    *,
    output_history: np.ndarray,
    test_data_last: np.ndarray,
    test_mask_last: np.ndarray,
) -> np.ndarray:
    epochs, channels, _, _ = output_history.shape
    l1_history = np.zeros(epochs, dtype=np.float32)
    mse_history = np.zeros(epochs, dtype=np.float32)

    for epoch_idx, epoch_output in enumerate(output_history):
        l1_loss, mse_loss = _compute_masked_losses(test_data_last, epoch_output, test_mask_last)
        l1_history[epoch_idx] = l1_loss
        mse_history[epoch_idx] = mse_loss

    return _build_loss_history_cube(
        l1_history=l1_history,
        mse_history=mse_history,
        channels=channels,
    )


def _format_surface_for_storage(surface: np.ndarray) -> np.ndarray:
    surface_array = np.asarray(surface, dtype=np.float32)
    if surface_array.ndim == 3 and surface_array.shape[0] == 1:
        return surface_array[0]
    return surface_array


def _build_experiment_artifacts(
    *,
    static_data: dict[str, Any],
    optimizer_artifacts: dict[str, Any],
) -> dict[str, Any]:
    member_artifacts = optimizer_artifacts["member_artifacts"]
    ensemble_size = len(member_artifacts)
    if ensemble_size == 0:
        raise ValueError("optimizer did not produce any ensemble member artifacts")

    channels = int(member_artifacts[0]["train_data"].shape[0])
    test_data = np.repeat(
        np.asarray(static_data["test_data"][:, -1:, ...], dtype=np.float32)[None, ...],
        repeats=ensemble_size,
        axis=0,
    )
    test_mask = np.repeat(
        np.asarray(static_data["test_mask"], dtype=bool)[None, ...],
        repeats=ensemble_size,
        axis=0,
    )

    model_space_member_surfaces = [
        np.asarray(member_artifact["surface_model_space"], dtype=np.float32)
        for member_artifact in member_artifacts
    ]
    final_model_space_surface = reduce_surface_ensemble(
        surfaces=model_space_member_surfaces,
        reduction=optimizer_artifacts["surface_reducer"],
    )

    train_data = np.stack(
        [np.asarray(member_artifact["train_data"][:, -1:, ...], dtype=np.float32) for member_artifact in member_artifacts],
        axis=0,
    )
    val_data = np.stack(
        [np.asarray(member_artifact["val_data"][:, -1:, ...], dtype=np.float32) for member_artifact in member_artifacts],
        axis=0,
    )
    train_mask = np.stack(
        [np.asarray(member_artifact["train_mask"][:, -1:, ...], dtype=bool) for member_artifact in member_artifacts],
        axis=0,
    )
    val_mask = np.stack(
        [np.asarray(member_artifact["val_mask"][:, -1:, ...], dtype=bool) for member_artifact in member_artifacts],
        axis=0,
    )
    train_k_output = np.stack(
        [np.moveaxis(np.asarray(member_artifact["output_history"], dtype=np.float32), 0, 1) for member_artifact in member_artifacts],
        axis=0,
    )
    val_min_idx = np.stack(
        [np.asarray(member_artifact["selected_epoch_indices"], dtype=np.int64) for member_artifact in member_artifacts],
        axis=0,
    )
    train_k_loss = np.stack(
        [
            _build_loss_history_cube(
                l1_history=np.asarray(member_artifact["train_l1_history"], dtype=np.float32),
                mse_history=np.asarray(member_artifact["train_mse_history"], dtype=np.float32),
                channels=channels,
            )
            for member_artifact in member_artifacts
        ],
        axis=0,
    )
    val_k_loss = np.stack(
        [
            _build_loss_history_cube(
                l1_history=np.asarray(member_artifact["val_l1_history"], dtype=np.float32),
                mse_history=np.asarray(member_artifact["val_mse_history"], dtype=np.float32),
                channels=channels,
            )
            for member_artifact in member_artifacts
        ],
        axis=0,
    )

    test_data_last = np.asarray(static_data["test_data"][:, -1, ...], dtype=np.float32)
    test_mask_last = np.asarray(static_data["test_mask"][:, 0, ...], dtype=bool)
    test_k_loss = np.stack(
        [
            _build_test_loss_history_cube(
                output_history=np.asarray(member_artifact["output_history"], dtype=np.float32),
                test_data_last=test_data_last,
                test_mask_last=test_mask_last,
            )
            for member_artifact in member_artifacts
        ],
        axis=0,
    )

    experiment_data = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "train_output": _format_surface_for_storage(final_model_space_surface),
        "train_output_real": _format_surface_for_storage(optimizer_artifacts["surface"]),
        "val_min_idx": val_min_idx,
        "train_k_output": train_k_output,
        "train_k_loss": train_k_loss,
        "val_k_loss": val_k_loss,
        "test_k_loss": test_k_loss,
    }

    normalization_stats = optimizer_artifacts.get("normalization_stats")
    if normalization_stats is not None:
        experiment_data["normalization_stats"] = normalization_stats

    return experiment_data


def _run_single_experiment(
    *,
    config_base: dict[str, Any],
    experiment_output_folder: str,
    test_sensor_group: list[int],
    sensor_group_key: str,
    time_window_iso: str,
    disable_nested_tqdm: bool = False,
) -> dict[str, Any]:
    time_window_dt = pd.to_datetime(time_window_iso).to_pydatetime()

    config = config_base.copy()
    config["date"] = time_window_dt.isoformat()
    config["validation_sensors"] = 4
    config["test_sensors"] = test_sensor_group

    date_window = pd.to_timedelta(config["hours"] - 1, unit="h")
    static_data = collect_data(
        start_date=time_window_dt - date_window,
        end_date=time_window_dt,
        add_meteo=bool(config.get("add_meteo")),
        add_time_channels=bool(config.get("add_time_channels")),
        add_coordinates=bool(config.get("add_coordinates")),
        add_traffic_data=bool(config.get("add_traffic_data")),
        pollutants=list(config["pollutants"]),
        test_sensors=list(test_sensor_group),
        normalize=bool(config.get("normalize")),
    )

    optimizer = DipEnsembleOptimizer(
        configuration=config,
        static_data=static_data,
        disable_tqdm=disable_nested_tqdm,
    )
    surface_real = np.asarray(optimizer.optimize(), dtype=np.float32)
    optimizer_artifacts = optimizer.get_artifacts()
    experiment_data = _build_experiment_artifacts(
        static_data=static_data,
        optimizer_artifacts=optimizer_artifacts,
    )

    pollutants = list(config["pollutants"])
    normalization_stats = optimizer_artifacts.get("normalization_stats")
    train_data_first = np.asarray(experiment_data["train_data"][0], dtype=np.float32)
    val_data_first = np.asarray(experiment_data["val_data"][0], dtype=np.float32)
    test_data_first = np.asarray(experiment_data["test_data"][0], dtype=np.float32)
    train_mask_first = np.asarray(experiment_data["train_mask"][0], dtype=bool)
    val_mask_first = np.asarray(experiment_data["val_mask"][0], dtype=bool)
    test_mask_first = np.asarray(experiment_data["test_mask"][0], dtype=bool)

    if bool(config.get("normalize")):
        if normalization_stats is None:
            raise ValueError("normalization_stats are required when normalize=True.")
        x_data = _denormalize_masked_channels(
            train_data_first + val_data_first,
            train_mask_first | val_mask_first,
            pollutants=pollutants,
            normalization_stats=normalization_stats,
        )
        y_data = _denormalize_masked_channels(
            test_data_first,
            test_mask_first,
            pollutants=pollutants,
            normalization_stats=normalization_stats,
        )
    else:
        x_data = train_data_first + val_data_first
        y_data = test_data_first

    test_target = y_data[:, -1, ...]
    test_mask_last = np.asarray(test_mask_first[:, 0, ...], dtype=bool)
    dip_l1_loss, dip_mse_loss = _compute_masked_losses(test_target, surface_real, test_mask_last)

    interpolation_results = get_interpolation_loss(
        x_data,
        train_mask_first | val_mask_first,
        y_data,
        test_mask_first,
        pollutants,
    )

    experiment_file_name = f"{get_experiment_name(sensor_group_key, time_window_dt)}.npz"
    np.savez_compressed(os.path.join(experiment_output_folder, experiment_file_name), **experiment_data)

    return {
        "sensor_group": sensor_group_key,
        "time_window": pd.Timestamp(time_window_iso),
        "DIP_L1Loss": dip_l1_loss,
        "DIP_MSELoss": dip_mse_loss,
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


def _build_failure_record(
    *,
    sensor_group_key: str,
    time_window_iso: str,
    exc: BaseException,
) -> dict[str, str]:
    message = f"{exc.__class__.__name__}: {exc}"
    traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "sensor_group": sensor_group_key,
        "time_window": time_window_iso,
        "message": message,
        "traceback": traceback_text,
    }


def _append_failure_log(
    *,
    failure_log_file: str,
    failure: dict[str, str],
    failure_index: int,
) -> None:
    with open(failure_log_file, "a", encoding="utf-8") as handle:
        handle.write(f"[{failure_index}] {failure['sensor_group']} @ {failure['time_window']}\n")
        handle.write(f"{failure['message']}\n")
        handle.write(failure["traceback"])
        if not failure["traceback"].endswith("\n"):
            handle.write("\n")
        handle.write("\n")


def run_experiments(
    *,
    config_file: Path,
    max_workers: Optional[int] = None,
) -> None:
    config_base, experiment_output_folder, test_sensors, time_windows, df = _ensure_base_files(
        config_file=config_file,
    )
    workers = max_workers if max_workers is not None else os.cpu_count() or 1
    workers = max(1, workers)
    run_in_parallel = max_workers is not None and workers > 1

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
                    "disable_nested_tqdm": run_in_parallel,
                }
            )

    if not jobs:
        click.echo("All experiments are already processed.")
        return

    if run_in_parallel:
        click.echo(f"Running {len(jobs)} experiments in parallel with {workers} worker(s).")
    else:
        mode = "sequentially" if max_workers is None else "sequentially (max_workers=1)"
        click.echo(f"Running {len(jobs)} experiments {mode}.")

    results_file = os.path.join(experiment_output_folder, "results.csv")
    failure_log_file = os.path.join(experiment_output_folder, "failures.log")
    if os.path.exists(failure_log_file):
        os.remove(failure_log_file)
    failures: list[dict[str, str]] = []

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
                        failure = _build_failure_record(
                            sensor_group_key=sensor_group_key,
                            time_window_iso=time_window_iso,
                            exc=exc,
                        )
                        failures.append(failure)
                        _append_failure_log(
                            failure_log_file=failure_log_file,
                            failure=failure,
                            failure_index=len(failures),
                        )
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
                    failure = _build_failure_record(
                        sensor_group_key=sensor_group_key,
                        time_window_iso=time_window_iso,
                        exc=exc,
                    )
                    failures.append(failure)
                    _append_failure_log(
                        failure_log_file=failure_log_file,
                        failure=failure,
                        failure_index=len(failures),
                    )
                else:
                    _apply_row_result(df, row_result)
                    df.to_csv(results_file, index=False)
                finally:
                    pbar.set_postfix({"failed": len(failures)})
                    pbar.update(1)

    if failures:
        click.echo(f"{len(failures)} experiment(s) failed. Their rows remain with processed=False.")
        click.echo(f"Full tracebacks written to {failure_log_file}")


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
