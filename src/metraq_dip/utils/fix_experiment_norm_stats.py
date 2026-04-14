from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from metraq_dip.data.aq_backends import AQBackend, get_aq_backend_for_config
from metraq_dip.data import data as data_module
from metraq_dip.tools.config_tools import load_session_config
from metraq_dip.tools.random_tools import sensor_group_hash

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"


@dataclass(frozen=True)
class ExperimentArtifactInfo:
    experiment_file: Path
    sensor_group_key: str
    test_sensors: tuple[int, ...]
    time_window: pd.Timestamp


@dataclass
class RepairSummary:
    total_files: int = 0
    updated: int = 0
    skipped_existing: int = 0
    skipped_not_normalized: int = 0
    failed: list[str] = field(default_factory=list)


def parse_experiment_artifact(experiment_file: Path) -> ExperimentArtifactInfo:
    stem = experiment_file.stem
    if not stem.startswith("exp_"):
        raise ValueError(f"Invalid experiment artifact name: {experiment_file.name}")

    payload = stem[4:]
    sensor_group_key, sep, time_token = payload.rpartition("_")
    if not sep or not sensor_group_key or not time_token:
        raise ValueError(f"Invalid experiment artifact name: {experiment_file.name}")

    try:
        time_window = pd.Timestamp(datetime.strptime(time_token, TIMESTAMP_FORMAT))
    except ValueError as exc:
        raise ValueError(f"Invalid experiment timestamp in {experiment_file.name}") from exc

    try:
        test_sensors = tuple(int(sensor_id) for sensor_id in sensor_group_key.split("-"))
    except ValueError as exc:
        raise ValueError(f"Invalid sensor group in {experiment_file.name}") from exc

    if sensor_group_hash(test_sensors) != sensor_group_key:
        raise ValueError(f"Sensor group key mismatch in {experiment_file.name}")

    return ExperimentArtifactInfo(
        experiment_file=experiment_file,
        sensor_group_key=sensor_group_key,
        test_sensors=test_sensors,
        time_window=time_window,
    )


def load_session_inventory(session_folder: Path) -> tuple[set[str], set[str]]:
    data_file = session_folder / "data.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Session folder {session_folder} does not contain data.npz")

    with np.load(data_file, allow_pickle=True) as data:
        sensor_groups = {
            sensor_group_hash(np.asarray(group, dtype=int).tolist())
            for group in data["test_sensors"]
        }
        time_windows = {
            pd.Timestamp(window).isoformat()
            for window in data["time_windows"].tolist()
        }

    return sensor_groups, time_windows


def read_saved_minmax_map(experiment_file: Path) -> dict[int, tuple[float, float]] | None:
    with np.load(experiment_file, allow_pickle=True) as archive:
        if "minmax_map" not in archive.files:
            return None
        return archive["minmax_map"].item()


def recover_experiment_minmax_map(
    *,
    pollutants: list[int],
    hours: int,
    time_window: pd.Timestamp,
    test_sensors: tuple[int, ...],
    aq_backend: AQBackend,
) -> dict[int, tuple[float, float]]:
    end_date = pd.Timestamp(time_window)
    start_date = end_date - pd.Timedelta(hours=hours - 1)
    grid_ctx, sensor_ids = data_module.get_grid(pollutants=pollutants, aq_backend=aq_backend)
    pollutant_data, _, _, _ = data_module.generate_pollutant_magnitudes(
        start_date=start_date.to_pydatetime(),
        end_date=end_date.to_pydatetime(),
        pollutants=pollutants,
        grid_ctx=grid_ctx,
        sensor_ids=sensor_ids,
        normalize=False,
        aq_backend=aq_backend,
    )
    test_mask = data_module._build_sensor_mask(grid_ctx=grid_ctx, sensors=list(test_sensors), aq_backend=aq_backend)
    return data_module._compute_pollutant_normalization_stats(
        pollutant_data=pollutant_data,
        pollutants=pollutants,
        test_mask=test_mask,
    )


def write_experiment_minmax_map(
    experiment_file: Path,
    *,
    minmax_map: dict[int, tuple[float, float]],
) -> None:
    with np.load(experiment_file, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}

    payload["minmax_map"] = minmax_map

    temp_file: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".npz",
            dir=experiment_file.parent,
            delete=False,
        ) as handle:
            temp_file = Path(handle.name)

        np.savez_compressed(temp_file, **payload)
        os.replace(temp_file, experiment_file)
    finally:
        if temp_file is not None and temp_file.exists():
            temp_file.unlink(missing_ok=True)


def repair_experiment_folder(
    session_folder: Path,
    *,
    force: bool = False,
    show_progress: bool = False,
    reporter: Callable[[str], None] | None = None,
) -> RepairSummary:
    summary = RepairSummary()
    experiment_files = sorted(session_folder.glob("exp_*.npz"))
    summary.total_files = len(experiment_files)
    if not experiment_files:
        if reporter is not None:
            reporter("No experiment artifacts found.")
        return summary

    config = load_session_config(session_folder / "config.yaml")
    if not config.normalize:
        summary.skipped_not_normalized = len(experiment_files)
        if reporter is not None:
            reporter("Session is not normalized; skipping all experiment artifacts.")
        return summary

    if reporter is not None:
        action = "Recomputing" if force else "Backfilling"
        reporter(f"{action} normalization stats for {len(experiment_files)} experiment artifact(s).")

    valid_sensor_groups, valid_time_windows = load_session_inventory(session_folder)
    cached_stats: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
    aq_backend = get_aq_backend_for_config(config)
    progress_bar = tqdm(
        experiment_files,
        desc="Repairing experiments",
        unit="file",
        disable=not show_progress,
    )

    try:
        for experiment_file in progress_bar:
            try:
                saved_minmax_map = read_saved_minmax_map(experiment_file)
                if saved_minmax_map is not None and not force:
                    summary.skipped_existing += 1
                    continue

                artifact = parse_experiment_artifact(experiment_file)
                time_window_key = artifact.time_window.isoformat()
                if artifact.sensor_group_key not in valid_sensor_groups:
                    raise ValueError("sensor group is not present in session data.npz")
                if time_window_key not in valid_time_windows:
                    raise ValueError("time window is not present in session data.npz")

                cache_key = (artifact.sensor_group_key, time_window_key)
                if cache_key not in cached_stats:
                    cached_stats[cache_key] = recover_experiment_minmax_map(
                        pollutants=list(config.pollutants),
                        hours=int(config.hours),
                        time_window=artifact.time_window,
                        test_sensors=artifact.test_sensors,
                        aq_backend=aq_backend,
                    )

                write_experiment_minmax_map(
                    experiment_file,
                    minmax_map=cached_stats[cache_key],
                )
                summary.updated += 1
            except Exception as exc:
                message = f"{experiment_file.name}: {exc}"
                summary.failed.append(message)
                if reporter is not None:
                    reporter(f"Failed to repair {message}")
            finally:
                progress_bar.set_postfix(
                    updated=summary.updated,
                    skipped=summary.skipped_existing + summary.skipped_not_normalized,
                    failed=len(summary.failed),
                )
    finally:
        progress_bar.close()

    return summary


@click.command()
@click.argument(
    "session_folder",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True, path_type=Path),
)
@click.option(
    "--force",
    is_flag=True,
    help="Recompute and overwrite minmax_map even when it already exists.",
)
def cli(session_folder: Path, force: bool) -> None:
    summary = repair_experiment_folder(
        session_folder,
        force=force,
        show_progress=True,
        reporter=click.echo,
    )

    click.echo(f"Session: {session_folder.resolve()}")
    click.echo(f"{'total_files':>22}: {summary.total_files}")
    click.echo(f"{'updated':>22}: {summary.updated}")
    click.echo(f"{'skipped_existing':>22}: {summary.skipped_existing}")
    click.echo(f"{'skipped_not_normalized':>22}: {summary.skipped_not_normalized}")
    click.echo(f"{'failed':>22}: {len(summary.failed)}")

    for failure in summary.failed:
        click.echo(f"  - {failure}", err=True)

    if summary.failed:
        raise click.ClickException(f"Failed to update {len(summary.failed)} experiment file(s).")


if __name__ == "__main__":
    cli()
