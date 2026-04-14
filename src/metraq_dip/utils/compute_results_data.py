#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

from metraq_dip.tools.results_stats import (
    backfill_results_stat_columns,
    ensure_results_stat_columns,
    validate_results_stat_columns,
)
from metraq_dip.tools.tools import is_truthy

METHODS = ["DIP", "KRG", "IDW"]
LOSS_METRICS = (
    ("MAE", "L1Loss"),
    ("MSE", "MSELoss"),
)
PAIRWISE_COMPARISONS = (
    ("DIP", "KRG"),
    ("DIP", "IDW"),
    ("KRG", "IDW"),
)
HISTOGRAM_HISTNORM_CHOICES = ("count", "percent", "probability", "density", "probability density")
HISTOGRAM_VALUE_LABELS = {
    "count": "Count",
    "percent": "Percent",
    "probability": "Probability",
    "density": "Density",
    "probability density": "Probability density",
}


def resolve_results_csv(experiment_folder: str | Path) -> Path:
    folder_path = Path(experiment_folder)
    csv_path = folder_path / "results.csv"
    if not csv_path.exists():
        raise ValueError(
            f"Experiment folder {folder_path} does not contain results.csv."
        )
    if not csv_path.is_file():
        raise ValueError(f"Resolved results path is not a file: {csv_path}")
    return csv_path

def load_results(
    csv_path: str | Path,
    *,
    experiment_folder: str | Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"sensor_group": "string"}, parse_dates=["time_window"]).copy()
    schema_updated = ensure_results_stat_columns(df)
    backfilled = False
    if experiment_folder is not None:
        processed_selector = (
            df["processed"].map(is_truthy)
            if "processed" in df.columns
            else pd.Series(True, index=df.index)
        )
        backfilled = backfill_results_stat_columns(
            df=df,
            experiment_folder=experiment_folder,
            row_selector=processed_selector,
        )
        if schema_updated or backfilled:
            df.to_csv(csv_path, index=False)
        validate_results_stat_columns(
            df=df,
            row_selector=processed_selector,
        )

    df["time_window_dt"] = pd.to_datetime(df["time_window"])
    df["date"] = df["time_window_dt"].dt.date.astype(str)
    df["hour"] = df["time_window_dt"].dt.strftime("%H:%M")
    df["day_type"] = np.where(df["time_window_dt"].dt.weekday >= 5, "Weekend", "Weekday")
    return df


def filter_processed_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "processed" not in df.columns:
        return df.copy()

    processed_mask = df["processed"].map(is_truthy)
    filtered = df.loc[processed_mask].copy()
    if filtered.empty:
        raise ValueError("No processed rows found in the results file.")
    return filtered


def get_group_map(df: pd.DataFrame) -> dict[str, str]:
    group_ids = df["sensor_group"].drop_duplicates().tolist()
    return {gid: f"G{i + 1}" for i, gid in enumerate(group_ids)}


def compute_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mae = df[[f"{m}_L1Loss" for m in METHODS]].to_numpy()
    mse = df[[f"{m}_MSELoss" for m in METHODS]].to_numpy()
    mae_wins = (mae == mae.min(axis=1, keepdims=True)).sum(axis=0)
    mse_wins = (mse == mse.min(axis=1, keepdims=True)).sum(axis=0)

    for i, m in enumerate(METHODS):
        rows.append({
            "Method": m,
            "Mean MAE": df[f"{m}_L1Loss"].mean(),
            "Median MAE": df[f"{m}_L1Loss"].median(),
            "Std MAE": df[f"{m}_L1Loss"].std(ddof=1),
            "Mean MSE": df[f"{m}_MSELoss"].mean(),
            "Median MSE": df[f"{m}_MSELoss"].median(),
            "Std MSE": df[f"{m}_MSELoss"].std(ddof=1),
            "MAE wins": int(mae_wins[i]),
            "MSE wins": int(mse_wins[i]),
        })
    return pd.DataFrame(rows)


def _build_histogram_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    if nbins <= 0:
        raise ValueError("nbins must be > 0.")

    data_min = float(np.min(values))
    data_max = float(np.max(values))
    if np.isclose(data_min, data_max):
        half_span = max(abs(data_min) * 0.05, 0.5)
        return np.linspace(data_min - half_span, data_max + half_span, nbins + 1)

    return np.histogram_bin_edges(values, bins=nbins)


def _build_centered_histogram_bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    if nbins <= 0:
        raise ValueError("nbins must be > 0.")

    max_abs = float(np.max(np.abs(values)))
    if np.isclose(max_abs, 0.0):
        max_abs = 0.5
    return np.linspace(-max_abs, max_abs, nbins + 1)


def _normalize_histogram_values(counts: np.ndarray, bin_edges: np.ndarray, histnorm: str) -> np.ndarray:
    total = int(counts.sum())
    if histnorm == "count":
        return counts.astype(float)
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    if histnorm == "percent":
        return counts.astype(float) * 100.0 / total
    if histnorm == "probability":
        return counts.astype(float) / total
    if histnorm in {"density", "probability density"}:
        bin_widths = np.diff(bin_edges)
        return counts.astype(float) / (total * bin_widths)

    raise ValueError(f"Unsupported histnorm: {histnorm}")


def compute_loss_histogram_plot_data(
    df: pd.DataFrame,
    *,
    nbins: int = 60,
    histnorm: str = "count",
) -> pd.DataFrame:
    if histnorm not in HISTOGRAM_HISTNORM_CHOICES:
        raise ValueError(f"histnorm must be one of: {', '.join(HISTOGRAM_HISTNORM_CHOICES)}")

    filtered = filter_processed_rows(df)
    rows: list[dict[str, Any]] = []
    sample_count = len(filtered)

    for metric_name, suffix in LOSS_METRICS:
        combined_values = filtered[[f"{method}_{suffix}" for method in METHODS]].to_numpy(dtype=float).ravel()
        bin_edges = _build_histogram_bin_edges(combined_values, nbins)

        for method in METHODS:
            loss_column = f"{method}_{suffix}"
            if loss_column not in filtered.columns:
                raise ValueError(f"Results file is missing required loss column: {loss_column}")

            loss_values = filtered[loss_column].to_numpy(dtype=float)
            counts, _ = np.histogram(loss_values, bins=bin_edges)
            histogram_values = _normalize_histogram_values(counts, bin_edges, histnorm)

            for bin_index, (left, right, count, value) in enumerate(
                zip(bin_edges[:-1], bin_edges[1:], counts, histogram_values)
            ):
                rows.append(
                    {
                        "Metric": metric_name,
                        "Method": method,
                        "Bin index": bin_index,
                        "Bin left": float(left),
                        "Bin right": float(right),
                        "Bin center": float((left + right) / 2.0),
                        "Bin width": float(right - left),
                        "Count": int(count),
                        "Value": float(value),
                        "Normalization": histnorm,
                        "Y label": HISTOGRAM_VALUE_LABELS[histnorm],
                        "Sample count": sample_count,
                    }
                )

    return pd.DataFrame(rows)


def compute_pairwise_difference_histogram_plot_data(
    df: pd.DataFrame,
    *,
    nbins: int = 60,
    histnorm: str = "count",
) -> pd.DataFrame:
    if histnorm not in HISTOGRAM_HISTNORM_CHOICES:
        raise ValueError(f"histnorm must be one of: {', '.join(HISTOGRAM_HISTNORM_CHOICES)}")

    filtered = filter_processed_rows(df)
    rows: list[dict[str, Any]] = []
    sample_count = len(filtered)

    for metric_name, suffix in LOSS_METRICS:
        differences = []
        for left_method, right_method in PAIRWISE_COMPARISONS:
            left_column = f"{left_method}_{suffix}"
            right_column = f"{right_method}_{suffix}"
            if left_column not in filtered.columns or right_column not in filtered.columns:
                raise ValueError(f"Results file is missing required loss columns: {left_column}, {right_column}")
            differences.append(
                filtered[left_column].to_numpy(dtype=float) - filtered[right_column].to_numpy(dtype=float)
            )

        combined_differences = np.concatenate(differences)
        bin_edges = _build_centered_histogram_bin_edges(combined_differences, nbins)

        for (left_method, right_method), difference_values in zip(PAIRWISE_COMPARISONS, differences):
            counts, _ = np.histogram(difference_values, bins=bin_edges)
            histogram_values = _normalize_histogram_values(counts, bin_edges, histnorm)
            comparison = f"{left_method} - {right_method}"

            for bin_index, (left, right, count, value) in enumerate(
                zip(bin_edges[:-1], bin_edges[1:], counts, histogram_values)
            ):
                rows.append(
                    {
                        "Metric": metric_name,
                        "Comparison": comparison,
                        "Left method": left_method,
                        "Right method": right_method,
                        "Bin index": bin_index,
                        "Bin left": float(left),
                        "Bin right": float(right),
                        "Bin center": float((left + right) / 2.0),
                        "Bin width": float(right - left),
                        "Count": int(count),
                        "Value": float(value),
                        "Normalization": histnorm,
                        "Y label": HISTOGRAM_VALUE_LABELS[histnorm],
                        "Sample count": sample_count,
                    }
                )

    return pd.DataFrame(rows)


def compute_friedman_and_posthoc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    friedman_rows = []
    posthoc_rows = []

    pairs = [("DIP", "KRG"), ("DIP", "IDW"), ("KRG", "IDW")]

    for metric_name, suffix in [("MAE", "L1Loss"), ("MSE", "MSELoss")]:
        a = df[f"DIP_{suffix}"]
        b = df[f"KRG_{suffix}"]
        c = df[f"IDW_{suffix}"]

        stat, pval = friedmanchisquare(a, b, c)
        ranks = np.array([
            rankdata(row, method="average")
            for row in df[[f"DIP_{suffix}", f"KRG_{suffix}", f"IDW_{suffix}"]].to_numpy()
        ])
        friedman_rows.append({
            "Metric": metric_name,
            "Friedman chi2": stat,
            "p-value": pval,
            "DIP rank": ranks[:, 0].mean(),
            "KRG rank": ranks[:, 1].mean(),
            "IDW rank": ranks[:, 2].mean(),
        })

        raw_pvals = []
        pair_names = []
        for m1, m2 in pairs:
            _, p = wilcoxon(
                df[f"{m1}_{suffix}"],
                df[f"{m2}_{suffix}"],
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            raw_pvals.append(p)
            pair_names.append((m1, m2))
        corrected = multipletests(raw_pvals, method="holm")[1]
        for (m1, m2), raw_p, corr_p in zip(pair_names, raw_pvals, corrected):
            posthoc_rows.append({
                "Metric": metric_name,
                "Comparison": f"{m1} vs {m2}",
                "Raw p-value": raw_p,
                "Holm-adjusted p-value": corr_p,
                "Significant": "Yes" if corr_p < 0.05 else "No",
            })

    return pd.DataFrame(friedman_rows), pd.DataFrame(posthoc_rows)


def compute_group_stats(df: pd.DataFrame, suffix: str, group_map: dict[str, str]) -> pd.DataFrame:
    out = (
        df.groupby("sensor_group")
        .agg({
            f"DIP_{suffix}": ["mean", "median", "std"],
            f"KRG_{suffix}": ["mean", "median", "std"],
            f"IDW_{suffix}": ["mean", "median", "std"],
        })
        .copy()
    )
    out.columns = [" ".join(col).strip() for col in out.columns.to_flat_index()]
    out = out.reset_index()
    out["Group"] = out["sensor_group"].map(group_map)
    cols = [
        "Group", "sensor_group",
        f"DIP_{suffix} mean", f"DIP_{suffix} median", f"DIP_{suffix} std",
        f"KRG_{suffix} mean", f"KRG_{suffix} median", f"KRG_{suffix} std",
        f"IDW_{suffix} mean", f"IDW_{suffix} median", f"IDW_{suffix} std",
    ]
    return out[cols]


def compute_temporal_split(df: pd.DataFrame, split_col: str) -> pd.DataFrame:
    rows = []
    for split_value, sub in df.groupby(split_col, sort=False):
        row = {split_col: split_value}
        for m in METHODS:
            row[f"{m} Mean MAE"] = sub[f"{m}_L1Loss"].mean()
            row[f"{m} Median MAE"] = sub[f"{m}_L1Loss"].median()
            row[f"{m} Mean MSE"] = sub[f"{m}_MSELoss"].mean()
            row[f"{m} Median MSE"] = sub[f"{m}_MSELoss"].median()
        rows.append(row)
    return pd.DataFrame(rows)


def compute_hardest_cases(df: pd.DataFrame, group_map: dict[str, str], q: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    tail_group_rows = []
    tail_date_rows = []

    for metric_name, suffix in [("MAE tail", "L1Loss"), ("MSE tail", "MSELoss")]:
        metric_col = f"DIP_{suffix}"
        threshold = df[metric_col].quantile(q)
        tail = df[df[metric_col] >= threshold].copy()

        arr = tail[[f"DIP_{suffix}", f"KRG_{suffix}", f"IDW_{suffix}"]].to_numpy()
        dip_wins_pct = float((arr[:, 0] == arr.min(axis=1)).mean() * 100)

        rows.append({
            "Metric": metric_name,
            "Threshold": threshold,
            "Cases": int(len(tail)),
            "08:00 (%)": float((tail["hour"] == "08:00").mean() * 100),
            "Weekend (%)": float((tail["day_type"] == "Weekend").mean() * 100),
            "Mean data std": float(tail["data_std"].mean()),
            "Mean data p90-p10": float(tail["data_p90_p10"].mean()),
            "DIP wins (%)": dip_wins_pct,
        })

        group_counts = (
            tail["sensor_group"]
            .map(group_map)
            .value_counts(normalize=True)
            .mul(100)
            .rename("Share (%)")
            .reset_index()
            .rename(columns={"index": "Group"})
        )
        group_counts.insert(0, "Metric", metric_name)
        tail_group_rows.append(group_counts)

        date_counts = (
            tail["date"]
            .value_counts()
            .rename("Count")
            .reset_index()
            .rename(columns={"index": "Date", "date": "Date"})
        )
        date_counts.insert(0, "Metric", metric_name)
        tail_date_rows.append(date_counts)

    return pd.DataFrame(rows), pd.concat(tail_group_rows, ignore_index=True), pd.concat(tail_date_rows, ignore_index=True)


def main(
    *,
    experiment_folder: Path,
    outdir: Path | None = None,
    hist_bins: int = 60,
    histnorm: str = "count",
) -> Path:
    if outdir is None:
        outdir = experiment_folder / "derived_tables_data"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_results_csv(experiment_folder)
    df = load_results(csv_path, experiment_folder=experiment_folder)
    group_map = get_group_map(df)

    pd.DataFrame(
        [{"Group": label, "sensor_group": gid} for gid, label in group_map.items()]
    ).to_csv(outdir / "group_map.csv", index=False)

    overall = compute_overall(df)
    overall.to_csv(outdir / "overall_performance.csv", index=False)

    loss_histogram_plot_data = compute_loss_histogram_plot_data(
        df,
        nbins=hist_bins,
        histnorm=histnorm,
    )
    loss_histogram_plot_data.to_csv(outdir / "loss_histogram_plot_data.csv", index=False)
    pairwise_difference_histogram_plot_data = compute_pairwise_difference_histogram_plot_data(
        df,
        nbins=hist_bins,
        histnorm=histnorm,
    )
    pairwise_difference_histogram_plot_data.to_csv(
        outdir / "pairwise_difference_histogram_plot_data.csv",
        index=False,
    )

    friedman_df, posthoc_df = compute_friedman_and_posthoc(df)
    friedman_df.to_csv(outdir / "friedman_results.csv", index=False)
    posthoc_df.to_csv(outdir / "wilcoxon_holm_results.csv", index=False)

    mae_group = compute_group_stats(df, "L1Loss", group_map)
    mse_group = compute_group_stats(df, "MSELoss", group_map)
    mae_group.to_csv(outdir / "group_mae_stats.csv", index=False)
    mse_group.to_csv(outdir / "group_mse_stats.csv", index=False)

    hour_df = compute_temporal_split(df, "hour")
    day_df = compute_temporal_split(df, "day_type")
    hour_df.to_csv(outdir / "temporal_hour.csv", index=False)
    day_df.to_csv(outdir / "temporal_daytype.csv", index=False)

    hardest_df, tail_groups_df, tail_dates_df = compute_hardest_cases(df, group_map, q=0.95)
    hardest_df.to_csv(outdir / "hardest_cases.csv", index=False)
    tail_groups_df.to_csv(outdir / "hardest_case_group_breakdown.csv", index=False)
    tail_dates_df.to_csv(outdir / "hardest_case_date_breakdown.csv", index=False)

    metadata = {
        "experiment_folder": str(experiment_folder.resolve()),
        "source_csv": str(csv_path.resolve()),
        "n_rows": int(len(df)),
        "n_groups": int(df["sensor_group"].nunique()),
        "methods": METHODS,
        "group_label_rule": "G1..Gn assigned by first appearance of sensor_group in CSV",
        "loss_histogram_plot_data": {
            "file": "loss_histogram_plot_data.csv",
            "bins": int(hist_bins),
            "histnorm": histnorm,
        },
        "pairwise_difference_histogram_plot_data": {
            "file": "pairwise_difference_histogram_plot_data.csv",
            "bins": int(hist_bins),
            "histnorm": histnorm,
            "comparison_rule": "signed difference = first method loss minus second method loss",
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return outdir.resolve()


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument(
        "experiment_folder",
        type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
    )
    @click.option(
        "--outdir",
        type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
        default=None,
        help="Directory for summary CSV/JSON files. Defaults to EXPERIMENT_FOLDER/derived_tables_data.",
    )
    @click.option(
        "--hist-bins",
        type=click.IntRange(min=1),
        default=60,
        show_default=True,
        help="Number of bins for histogram plot data.",
    )
    @click.option(
        "--histnorm",
        type=click.Choice(HISTOGRAM_HISTNORM_CHOICES, case_sensitive=True),
        default="count",
        show_default=True,
        help="Normalization mode for histogram plot data.",
    )
    def cli(experiment_folder: Path, outdir: Path | None, hist_bins: int, histnorm: str) -> None:
        """Compute summary CSV/JSON data from an experiment folder."""
        try:
            output_dir = main(
                experiment_folder=experiment_folder,
                outdir=outdir,
                hist_bins=hist_bins,
                histnorm=histnorm,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Wrote summary data files to: {output_dir}")

    cli()
