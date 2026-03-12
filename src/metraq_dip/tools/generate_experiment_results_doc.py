from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


TRUE_VALUES = {"1", "true", "t", "yes", "y"}


@dataclass
class SessionSummary:
    name: str
    pollutants: str
    epochs: str
    ensemble_size: str
    hours: str
    normalize: str
    add_meteo: str
    add_time_channels: str
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
    time_window_start: str
    time_window_end: str


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


def _format_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _load_session_summary(session_dir: Path) -> SessionSummary | None:
    config_file = session_dir / "config.yaml"
    data_file = session_dir / "data.npz"
    results_file = session_dir / "results.csv"

    if not (config_file.exists() and data_file.exists() and results_file.exists()):
        return None

    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
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
        normalize=_format_bool(config.get("normalize", False)),
        add_meteo=_format_bool(config.get("add_meteo", False)),
        add_time_channels=_format_bool(config.get("add_time_channels", False)),
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
        time_window_start=str(time_windows.min()),
        time_window_end=str(time_windows.max()),
    )


def _build_markdown(summaries: list[SessionSummary], experiments_root: Path) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    lines.append("# Experiment Results")
    lines.append("")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Source root: `{experiments_root.as_posix()}`")
    lines.append("- Data used per session: `config.yaml`, `data.npz`, `results.csv`")
    lines.append("- Ignored by design: detailed `exp_*.npz` files")
    lines.append("")
    lines.append("## Session Comparison")
    lines.append("")
    lines.append(
        "| Session | Processed / Expected | Completion | Norm | Meteo | TimeCh | Dist2Sens | "
        "DIP L1 mean | KRG L1 mean | IDW L1 mean | DIP vs KRG | DIP vs IDW | "
        "DIP MSE mean | KRG MSE mean | IDW MSE mean | Best L1 |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | --- |"
    )

    for s in summaries:
        lines.append(
            f"| `{s.name}` | {s.processed_rows}/{s.expected_runs} | {_format_pct(s.completion_pct)} | "
            f"{s.normalize} | {s.add_meteo} | {s.add_time_channels} | {s.add_distance_to_sensors} | "
            f"{_format_float(s.dip_l1_mean)} | {_format_float(s.krg_l1_mean)} | {_format_float(s.idw_l1_mean)} | "
            f"{_format_pct(s.dip_vs_krg_pct)} | {_format_pct(s.dip_vs_idw_pct)} | "
            f"{_format_float(s.dip_mse_mean)} | {_format_float(s.krg_mse_mean)} | {_format_float(s.idw_mse_mean)} | "
            f"{s.best_l1_model} |"
        )

    lines.append("")
    lines.append("## DIP Ranking (Complete Sessions Only)")
    lines.append("")
    complete = [s for s in summaries if abs(s.completion_pct - 100.0) < 1e-9]
    if complete:
        by_l1 = sorted(complete, key=lambda s: s.dip_l1_mean)
        by_mse = sorted(complete, key=lambda s: s.dip_mse_mean)

        lines.append("### By DIP L1 mean (lower is better)")
        lines.append("")
        for idx, s in enumerate(by_l1, start=1):
            lines.append(f"{idx}. `{s.name}`: {_format_float(s.dip_l1_mean)}")
        lines.append("")
        lines.append("### By DIP MSE mean (lower is better)")
        lines.append("")
        for idx, s in enumerate(by_mse, start=1):
            lines.append(f"{idx}. `{s.name}`: {_format_float(s.dip_mse_mean)}")
    else:
        lines.append("No complete sessions available for ranking.")

    lines.append("")
    lines.append("## Session Metadata")
    lines.append("")
    lines.append("| Session | Pollutants | Epochs | Ensemble | Hours | Sensor groups | Time windows | Date range |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for s in summaries:
        lines.append(
            f"| `{s.name}` | `{s.pollutants}` | {s.epochs} | {s.ensemble_size} | {s.hours} | "
            f"{s.sensor_groups} | {s.time_windows} | {s.time_window_start} -> {s.time_window_end} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Means are computed from rows where `processed = True`.")
    lines.append("- `DIP vs KRG` and `DIP vs IDW` are relative L1 changes: positive means DIP is better.")
    lines.append("- Direct comparison across sessions can be affected by configuration changes (for example normalization and extra channels).")

    return "\n".join(lines) + "\n"


def generate_doc(experiments_root: Path, output_file: Path) -> None:
    session_dirs = sorted(path for path in experiments_root.iterdir() if path.is_dir())
    summaries: list[SessionSummary] = []
    for session_dir in session_dirs:
        summary = _load_session_summary(session_dir)
        if summary is not None:
            summaries.append(summary)

    markdown = _build_markdown(summaries, experiments_root)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(markdown, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate experiment results documentation page.")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("output/experiments"),
        help="Folder containing experiment session directories.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("docs/experiment_results.md"),
        help="Markdown output file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_doc(args.experiments_root, args.output_file)
