from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import friedmanchisquare, rankdata


TRUE_VALUES = {"1", "true", "t", "yes", "y"}
CONFIG_DIFF_HEADER_LABELS: dict[str, str] = {
    "pollutants": "pollut.",
    "hours": "hrs",
    "epochs": "ep",
    "ensemble_size": "ens",
    "lr": "lr",
    "optimization_loss": "opt_loss",
    "normalize": "norm",
    "add_meteo": "meteo",
    "add_time_channels": "time_ch",
    "add_coordinates": "coords",
    "add_distance_to_sensors": "dist2sens",
    "model.architecture": "m.arch",
    "model.base_channels": "m.base_ch",
    "model.levels": "m.levels",
    "model.preserve_time": "m.keep_t",
    "model.learned_upsampling": "m.upscale",
    "model.skip_connections": "m.skip",
    "spread_test_groups.n_groups": "grp_n",
    "spread_test_groups.group_size": "grp_sz",
    "spread_test_groups.max_uses_per_sensor": "grp_max_use",
    "time_windows.strategy": "tw_mode",
    "time_windows.year": "tw_year",
    "time_windows.windows_per_month": "tw_win_mo",
    "time_windows.start_hours": "tw_start_h",
    "time_windows.weekend_fraction": "tw_weekend",
}


@dataclass
class SessionSummary:
    name: str
    pollutants: str
    epochs: str
    ensemble_size: str
    hours: str
    lr: str
    normalize: str
    add_meteo: str
    add_time_channels: str
    add_coordinates: str
    add_distance_to_sensors: str
    sensor_groups: int
    time_windows: int
    expected_runs: int
    rows: int
    processed_rows: int
    completion_pct: float
    dip_l1_mean: float
    krg_l1_mean: float
    idw_l1_mean: float
    dip_mse_mean: float
    krg_mse_mean: float
    idw_mse_mean: float
    dip_vs_krg_pct: float
    dip_vs_idw_pct: float
    best_l1_model: str
    friedman_n: int
    friedman_stat: float
    friedman_p_value: float
    dip_mean_rank: float
    krg_mean_rank: float
    idw_mean_rank: float
    time_window_start: str
    time_window_end: str
    config_values: dict[str, str]


def _to_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def _safe_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return float("nan")
    return float(df[column].astype(float).mean())


def _safe_pct_delta(reference: float, value: float) -> float:
    if np.isnan(reference) or reference == 0 or np.isnan(value):
        return float("nan")
    return float((1.0 - (value / reference)) * 100.0)


def _format_float(value: float, decimals: int = 4) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def _format_pct(value: float, decimals: int = 1) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value:.{decimals}f}%"


def _format_sci(value: float, decimals: int = 3) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value:.{decimals}e}"


def _format_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return _format_bool(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def _extract_config_values(config: dict[str, Any]) -> dict[str, str]:
    model = config.get("model", {}) or {}
    spread = config.get("spread_test_groups", {}) or {}
    random_windows = config.get("random_time_windows")
    all_windows = config.get("all_time_windows")

    if random_windows is not None:
        window_strategy = "random_time_windows"
        window_year = random_windows.get("year")
        windows_per_month = random_windows.get("windows_per_month")
        start_hours = random_windows.get("start_hours")
        weekend_fraction = random_windows.get("weekend_fraction")
    elif all_windows is not None:
        window_strategy = "all_time_windows"
        window_year = all_windows.get("year")
        windows_per_month = None
        start_hours = all_windows.get("start_hours")
        weekend_fraction = None
    else:
        window_strategy = "n/a"
        window_year = None
        windows_per_month = None
        start_hours = None
        weekend_fraction = None

    values: dict[str, str] = {
        "pollutants": _format_value(config.get("pollutants")),
        "hours": _format_value(config.get("hours")),
        "epochs": _format_value(config.get("epochs")),
        "ensemble_size": _format_value(config.get("ensemble_size")),
        "lr": _format_value(config.get("lr")),
        "optimization_loss": _format_value(config.get("optimization_loss")),
        "normalize": _format_value(config.get("normalize")),
        "add_meteo": _format_value(config.get("add_meteo")),
        "add_time_channels": _format_value(config.get("add_time_channels")),
        "add_coordinates": _format_value(config.get("add_coordinates")),
        "add_distance_to_sensors": _format_value(config.get("add_distance_to_sensors")),
        "model.architecture": _format_value(model.get("architecture")),
        "model.base_channels": _format_value(model.get("base_channels")),
        "model.levels": _format_value(model.get("levels")),
        "model.preserve_time": _format_value(model.get("preserve_time")),
        "model.learned_upsampling": _format_value(model.get("learned_upsampling")),
        "model.skip_connections": _format_value(model.get("skip_connections")),
        "spread_test_groups.n_groups": _format_value(spread.get("n_groups")),
        "spread_test_groups.group_size": _format_value(spread.get("group_size")),
        "spread_test_groups.max_uses_per_sensor": _format_value(spread.get("max_uses_per_sensor")),
        "time_windows.strategy": _format_value(window_strategy),
        "time_windows.year": _format_value(window_year),
        "time_windows.windows_per_month": _format_value(windows_per_month),
        "time_windows.start_hours": _format_value(start_hours),
        "time_windows.weekend_fraction": _format_value(weekend_fraction),
    }
    return values


def _compute_friedman(processed: pd.DataFrame) -> tuple[int, float, float, float, float, float]:
    required_columns = ["DIP_L1Loss", "KRG_L1Loss", "IDW_L1Loss"]
    missing = [column for column in required_columns if column not in processed.columns]
    if missing:
        return 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    data = processed[required_columns].apply(pd.to_numeric, errors="coerce").dropna()
    n = int(len(data))
    if n < 2:
        return n, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    stat, p_value = friedmanchisquare(data["DIP_L1Loss"], data["KRG_L1Loss"], data["IDW_L1Loss"])
    ranks = np.apply_along_axis(rankdata, 1, data.to_numpy(dtype=float))
    mean_ranks = np.mean(ranks, axis=0)
    return n, float(stat), float(p_value), float(mean_ranks[0]), float(mean_ranks[1]), float(mean_ranks[2])


def _load_session_summary(session_dir: Path) -> SessionSummary | None:
    config_file = session_dir / "config.yaml"
    data_file = session_dir / "data.npz"
    results_file = session_dir / "results.csv"

    if not (config_file.exists() and data_file.exists() and results_file.exists()):
        return None

    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    config_values = _extract_config_values(config)
    data = np.load(data_file, allow_pickle=True)
    results = pd.read_csv(results_file)

    test_sensors = data["test_sensors"]
    time_windows = pd.to_datetime(data["time_windows"])
    expected_runs = int(len(test_sensors) * len(time_windows))

    if "processed" in results.columns:
        processed_mask = _to_bool_series(results["processed"])
    else:
        processed_mask = pd.Series([True] * len(results))

    processed = results.loc[processed_mask].copy()
    friedman_n, friedman_stat, friedman_p_value, dip_mean_rank, krg_mean_rank, idw_mean_rank = _compute_friedman(
        processed
    )

    dip_l1_mean = _safe_mean(processed, "DIP_L1Loss")
    krg_l1_mean = _safe_mean(processed, "KRG_L1Loss")
    idw_l1_mean = _safe_mean(processed, "IDW_L1Loss")
    dip_mse_mean = _safe_mean(processed, "DIP_MSELoss")
    krg_mse_mean = _safe_mean(processed, "KRG_MSELoss")
    idw_mse_mean = _safe_mean(processed, "IDW_MSELoss")

    l1_scores = {
        "DIP": dip_l1_mean,
        "KRG": krg_l1_mean,
        "IDW": idw_l1_mean,
    }
    best_l1_model = min(
        (k for k, v in l1_scores.items() if not np.isnan(v)),
        key=lambda k: l1_scores[k],
        default="n/a",
    )

    return SessionSummary(
        name=session_dir.name,
        pollutants=str(config.get("pollutants", "n/a")),
        epochs=str(config.get("epochs", "n/a")),
        ensemble_size=str(config.get("ensemble_size", "n/a")),
        hours=str(config.get("hours", "n/a")),
        lr=str(config.get("lr", "n/a")),
        normalize=_format_bool(config.get("normalize", False)),
        add_meteo=_format_bool(config.get("add_meteo", False)),
        add_time_channels=_format_bool(config.get("add_time_channels", False)),
        add_coordinates=_format_bool(config.get("add_coordinates", False)),
        add_distance_to_sensors=_format_bool(config.get("add_distance_to_sensors", False)),
        sensor_groups=int(len(test_sensors)),
        time_windows=int(len(time_windows)),
        expected_runs=expected_runs,
        rows=int(len(results)),
        processed_rows=int(processed_mask.sum()),
        completion_pct=float((processed_mask.mean() * 100.0) if len(results) else 0.0),
        dip_l1_mean=dip_l1_mean,
        krg_l1_mean=krg_l1_mean,
        idw_l1_mean=idw_l1_mean,
        dip_mse_mean=dip_mse_mean,
        krg_mse_mean=krg_mse_mean,
        idw_mse_mean=idw_mse_mean,
        dip_vs_krg_pct=_safe_pct_delta(krg_l1_mean, dip_l1_mean),
        dip_vs_idw_pct=_safe_pct_delta(idw_l1_mean, dip_l1_mean),
        best_l1_model=best_l1_model,
        friedman_n=friedman_n,
        friedman_stat=friedman_stat,
        friedman_p_value=friedman_p_value,
        dip_mean_rank=dip_mean_rank,
        krg_mean_rank=krg_mean_rank,
        idw_mean_rank=idw_mean_rank,
        time_window_start=str(time_windows.min()),
        time_window_end=str(time_windows.max()),
        config_values=config_values,
    )


def _is_session_folder(path: Path) -> bool:
    return (
        (path / "config.yaml").exists()
        and (path / "data.npz").exists()
        and (path / "results.csv").exists()
    )


def _collect_session_dirs(source_folder: Path) -> list[Path]:
    if _is_session_folder(source_folder):
        return [source_folder]

    session_dirs: set[Path] = set()
    for config_file in source_folder.rglob("config.yaml"):
        session_dir = config_file.parent
        if _is_session_folder(session_dir):
            session_dirs.add(session_dir)

    return sorted(session_dirs, key=lambda path: path.as_posix())


def _build_config_diff_keys(summaries: list[SessionSummary]) -> list[str]:
    if not summaries:
        return []

    ordered_keys = list(summaries[0].config_values.keys())
    for summary in summaries[1:]:
        for key in summary.config_values:
            if key not in ordered_keys:
                ordered_keys.append(key)

    varying_keys: list[str] = []
    for key in ordered_keys:
        values = {summary.config_values.get(key, "n/a") for summary in summaries}
        if len(values) > 1:
            varying_keys.append(key)

    if varying_keys:
        return varying_keys
    return ordered_keys


def _config_key_label(key: str) -> str:
    return CONFIG_DIFF_HEADER_LABELS.get(key, key)


def _build_markdown(summaries: list[SessionSummary], source_root: Path) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    lines.append("# Experiment Results")
    lines.append("")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Source root: `{source_root.as_posix()}`")
    lines.append("- Data used per session: `config.yaml`, `data.npz`, `results.csv`")
    lines.append("- Ignored by design: detailed `exp_*.npz` files")
    lines.append("")
    lines.append("## 1. Configuration Differences")
    lines.append("")
    diff_keys = _build_config_diff_keys(summaries)
    if diff_keys:
        diff_labels = [_config_key_label(key) for key in diff_keys]
        lines.append("| Session | " + " | ".join(diff_labels) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(diff_labels)) + " |")
        for summary in summaries:
            values = [summary.config_values.get(key, "n/a") for key in diff_keys]
            lines.append(f"| `{summary.name}` | " + " | ".join(values) + " |")
        legend_items = [
            f"`{label}` = `{key}`"
            for key, label in zip(diff_keys, diff_labels)
            if label != key
        ]
        if legend_items:
            lines.append("")
            lines.append("Legend: " + ", ".join(legend_items))
    else:
        lines.append("No configuration data available.")

    lines.append("")
    lines.append("## 2. Mean L1/MSE by Model")
    lines.append("")
    lines.append(
        "| Session | Processed / Expected | Completion | "
        "DIP L1 mean | DIP MSE mean | "
        "KRG L1 mean | KRG MSE mean | "
        "IDW L1 mean | IDW MSE mean | "
        "Best L1 |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )

    if summaries:
        for s in summaries:
            lines.append(
                f"| `{s.name}` | {s.processed_rows}/{s.expected_runs} | {_format_pct(s.completion_pct)} | "
                f"{_format_float(s.dip_l1_mean)} | {_format_float(s.dip_mse_mean)} | "
                f"{_format_float(s.krg_l1_mean)} | {_format_float(s.krg_mse_mean)} | "
                f"{_format_float(s.idw_l1_mean)} | {_format_float(s.idw_mse_mean)} | "
                f"{s.best_l1_model} |"
            )
    else:
        lines.append("| _No complete sessions_ | - | - | - | - | - | - | - | - | - |")

    lines.append("")
    lines.append("## 3. Friedman Test (DIP vs KRG vs IDW)")
    lines.append("")
    lines.append(
        "| Session | Samples | Friedman stat | p-value | Significant (p < 0.05) | "
        "DIP mean rank | KRG mean rank | IDW mean rank |"
    )
    lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    if summaries:
        for s in summaries:
            significant = "yes" if not np.isnan(s.friedman_p_value) and s.friedman_p_value < 0.05 else "no"
            lines.append(
                f"| `{s.name}` | {s.friedman_n} | {_format_float(s.friedman_stat)} | "
                f"{_format_sci(s.friedman_p_value)} | {significant} | "
                f"{_format_float(s.dip_mean_rank)} | {_format_float(s.krg_mean_rank)} | {_format_float(s.idw_mean_rank)} |"
            )
    else:
        lines.append("| _No complete sessions_ | - | - | - | - | - | - | - |")

    lines.append("")
    lines.append("## Session Metadata")
    lines.append("")
    lines.append(
        "| Session | Pollutants | Epochs | Ensemble | Hours | LR | "
        "Norm | Meteo | TimeCh | Coord | Dist2Sens | Sensor groups | Time windows | Date range |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |")
    if summaries:
        for s in summaries:
            lines.append(
                f"| `{s.name}` | `{s.pollutants}` | {s.epochs} | {s.ensemble_size} | {s.hours} | {s.lr} | "
                f"{s.normalize} | {s.add_meteo} | {s.add_time_channels} | {s.add_coordinates} | {s.add_distance_to_sensors} | "
                f"{s.sensor_groups} | {s.time_windows} | {s.time_window_start} -> {s.time_window_end} |"
            )
    else:
        lines.append("| _No complete sessions_ | - | - | - | - | - | - | - | - | - | - | - | - | - |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Means are computed from rows where `processed = True`.")
    lines.append("- Friedman test is computed with `DIP_L1Loss`, `KRG_L1Loss`, and `IDW_L1Loss` from processed rows.")
    lines.append("- Mean rank interpretation: lower is better.")
    lines.append("- Direct comparison across sessions can be affected by configuration changes (for example normalization and extra channels).")

    return "\n".join(lines) + "\n"


def _is_complete_summary(summary: SessionSummary) -> bool:
    return summary.expected_runs > 0 and summary.processed_rows == summary.expected_runs


def generate_doc(source_folder: Path, output_file: Path) -> tuple[list[SessionSummary], list[SessionSummary]]:
    session_dirs = _collect_session_dirs(source_folder)
    if not session_dirs:
        raise ValueError(
            f"Invalid source folder '{source_folder}': expected one session folder or a root containing session folders "
            "with config.yaml, data.npz, and results.csv."
        )

    summaries: list[SessionSummary] = []
    for session_dir in session_dirs:
        summary = _load_session_summary(session_dir)
        if summary is not None:
            relative_name = session_dir.relative_to(source_folder).as_posix()
            if relative_name != ".":
                summary.name = relative_name
            summaries.append(summary)

    included = [summary for summary in summaries if _is_complete_summary(summary)]
    skipped = [summary for summary in summaries if not _is_complete_summary(summary)]

    markdown = _build_markdown(included, source_folder)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(markdown, encoding="utf-8")
    return included, skipped


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument(
        "source_folder",
        type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
    )
    @click.option(
        "--output-file",
        type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
        default=Path("docs/experiment_results.md"),
        show_default=True,
        help="Markdown output file.",
    )
    def cli(source_folder: Path, output_file: Path) -> None:
        """Generate an experiment results markdown page from one session folder or a sessions root folder."""
        try:
            included, skipped = generate_doc(source_folder, output_file)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(f"Generated documentation for {len(included)} complete session(s): {output_file.as_posix()}")
        if skipped:
            click.echo("Skipped incomplete session(s):")
            for summary in skipped:
                click.echo(
                    f"- {summary.name}: {summary.processed_rows}/{summary.expected_runs} "
                    f"({_format_pct(summary.completion_pct)})"
                )

    cli()
