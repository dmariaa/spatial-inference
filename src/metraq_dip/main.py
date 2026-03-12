from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import yaml

from metraq_dip.trainer.tools import load_training_session, get_session_results
import metraq_dip.experiments as ex

def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _build_session_summary(session_data: dict) -> dict[str, str]:
    configuration = session_data.get("configuration", {})
    data = session_data.get("data", {})
    experiment_files = session_data.get("experiment_files", [])
    results = session_data.get("results")

    test_sensors = data.get("test_sensors", [])
    time_windows = data.get("time_windows", [])

    sensor_groups_count = len(test_sensors)
    time_windows_count = len(time_windows)
    expected_experiments = sensor_groups_count * time_windows_count

    summary = {
        "pollutants": str(configuration.get("pollutants", "-")),
        "epochs": str(configuration.get("epochs", "-")),
        "hours": str(configuration.get("hours", "-")),
        "sensor_groups": str(sensor_groups_count),
        "time_windows": str(time_windows_count),
        "experiments_loaded": str(len(experiment_files)),
        "expected_experiments": str(expected_experiments),
    }

    if results is not None:
        results_rows = len(results.index)
        summary["results_rows"] = str(results_rows)
        if "processed" in results.columns:
            processed_rows = sum(_truthy(value) for value in results["processed"].tolist())
            summary["processed_rows"] = f"{processed_rows}/{results_rows}"

    return summary


def _format_metric(value: float) -> str:
    return f"{value:.6g}"


@click.group()
def cli() -> None:
    """metraq command line client."""


@cli.command()
@click.argument("session_folder",
                type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path))
@click.option("--config-file", default=None,
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
              help="Path to a configuration file.")
def run_experiments(session_folder: Path, config_file: Path) -> None:
    if config_file is None and not session_folder.exists():
        raise click.ClickException(f"Session folder {session_folder} does not exist and no configuration provided.")
    config = yaml.safe_load(config_file.read_text()) if config_file is not None else None
    ex.run_experiments(config_base=config, output_folder=str(session_folder))


@cli.command()
@click.argument(
    "session_folder",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
)
def session(session_folder: Path) -> None:
    """Show session information from an experiment folder."""
    try:
        session_data = load_training_session(str(session_folder))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Session: {session_folder.resolve()}")
    click.echo("-------------------")
    for key, value in _build_session_summary(session_data).items():
        click.echo(f"{key:>15}: {value}")


@cli.command()
@click.argument("session_folder", type=click.Path(exists=True, file_okay=False, dir_okay=True,
                                                  readable=True, path_type=Path))
def results(session_folder: Path) -> None:
    """Show statistical summary for a session results.csv file."""
    try:
        session_data = load_training_session(str(session_folder), load_experiments=False)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    session_results = get_session_results(session_data)
    friedman_stat, p_value = session_results["friedman"]
    mean_ranks = session_results["mean_ranks"]
    model_names = ("DIP", "KRG", "IDW")

    click.echo(f"Results: {session_folder.resolve()}")
    click.echo("-------------------")
    click.echo(f"{'friedman_stat':>15}: {_format_metric(friedman_stat)}")
    click.echo(f"{'p_value':>15}: {_format_metric(p_value)}")
    click.echo(f"{'significant':>15}: {'yes' if p_value < 0.05 else 'no'}")
    click.echo("")
    click.echo("Mean ranks (lower is better):")

    for model_name, mean_rank in zip(model_names, mean_ranks):
        click.echo(f"{model_name:>15}: {_format_metric(mean_rank)}")


@cli.command()
@click.argument("session_folder", type=click.Path(exists=True, file_okay=False, dir_okay=True,
                                                  readable=True, path_type=Path))
@click.argument("output_folder", type=click.Path(file_okay=False, dir_okay=True, writable=True,
                                                 path_type=Path), required=False)
def plot(session_folder: Path, output_folder: Optional[Path, None] = None):
    pass

if __name__ == "__main__":
    cli()
