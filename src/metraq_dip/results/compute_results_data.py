#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

METHODS = ["DIP", "KRG", "IDW"]


def load_results(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    df["time_window_dt"] = pd.to_datetime(df["time_window"])
    df["date"] = df["time_window_dt"].dt.date.astype(str)
    df["hour"] = df["time_window_dt"].dt.strftime("%H:%M")
    df["day_type"] = np.where(df["time_window_dt"].dt.weekday >= 5, "Weekend", "Weekday")
    return df


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to results_with_stats.csv")
    parser.add_argument("--outdir", default="derived_tables_data", help="Directory for summary CSV/JSON files")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.csv)
    group_map = get_group_map(df)

    pd.DataFrame(
        [{"Group": label, "sensor_group": gid} for gid, label in group_map.items()]
    ).to_csv(outdir / "group_map.csv", index=False)

    overall = compute_overall(df)
    overall.to_csv(outdir / "overall_performance.csv", index=False)

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
        "source_csv": str(Path(args.csv).resolve()),
        "n_rows": int(len(df)),
        "n_groups": int(df["sensor_group"].nunique()),
        "methods": METHODS,
        "group_label_rule": "G1..Gn assigned by first appearance of sensor_group in CSV",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote summary data files to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
