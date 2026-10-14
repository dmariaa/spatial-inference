from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd


KEYS = ["sensor_group", "window_end", "sensor_id"]
METHODS = ["DIP", "KRG", "IDW", "GNN_TRAFFIC"]


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    records = []
    iterator = frame.groupby(groups, sort=True) if groups else [((), frame)]
    for key, group in iterator:
        key = key if isinstance(key, tuple) else (key,)
        records.append(
            {
                **dict(zip(groups, key, strict=True)),
                "count": len(group),
                "mae": group["absolute_error"].mean(),
                "rmse": np.sqrt(group["squared_error"].mean()),
                "wape": group["absolute_error"].sum() / group["target"].abs().sum(),
                "bias": group["error"].mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def load_matched(gnn_root: Path, reference_path: Path) -> pd.DataFrame:
    files = sorted(gnn_root.glob("group-*/predictions.csv"))
    if len(files) != 10:
        raise click.ClickException(f"expected 10 GNN prediction files, found {len(files)}")
    gnn = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    reference = pd.read_csv(reference_path)
    reference = reference[reference["method"].isin(METHODS[:-1])].copy()
    for frame in (gnn, reference):
        frame["sensor_group"] = frame["sensor_group"].astype(str)
        frame["window_end"] = pd.to_datetime(frame["window_end"])
        frame["sensor_id"] = frame["sensor_id"].astype("int64")
    expected = set(map(tuple, gnn[KEYS].itertuples(index=False, name=None)))
    if gnn.duplicated(KEYS).any():
        raise click.ClickException("GNN predictions contain duplicate matched keys")
    for method in METHODS[:-1]:
        selected = reference[reference["method"] == method]
        if selected.duplicated(KEYS).any():
            raise click.ClickException(f"{method} predictions contain duplicate matched keys")
        actual = set(map(tuple, selected[KEYS].itertuples(index=False, name=None)))
        if actual != expected:
            raise click.ClickException(
                f"{method} keys do not match GNN keys: "
                f"missing={len(expected - actual)}, extra={len(actual - expected)}"
            )
    columns = KEYS + [
        "method",
        "target",
        "prediction",
        "error",
        "absolute_error",
        "squared_error",
    ]
    combined = pd.concat([reference[columns], gnn[columns]], ignore_index=True)
    target_counts = combined.groupby(KEYS)["target"].nunique(dropna=False)
    if (target_counts != 1).any():
        raise click.ClickException("target values differ between matched methods")
    return combined


def build_comparison(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    overall = summarize(combined, ["method"])
    by_group = summarize(combined, ["sensor_group", "method"])
    by_case = summarize(combined, ["sensor_group", "window_end", "method"])
    pivot = by_case.pivot(
        index=["sensor_group", "window_end"], columns="method", values="mae"
    )
    paired = []
    for comparator in METHODS[:-1]:
        delta = pivot["GNN_TRAFFIC"] - pivot[comparator]
        paired.append(
            {
                "comparator": comparator,
                "cases": len(delta),
                "gnn_wins": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
                "gnn_losses": int((delta > 0).sum()),
                "mean_mae_delta": delta.mean(),
                "median_mae_delta": delta.median(),
            }
        )
    return {
        "overall": overall,
        "by_group": by_group,
        "by_case": by_case,
        "paired": pd.DataFrame(paired),
    }


__all__ = ["KEYS", "METHODS", "build_comparison", "load_matched", "summarize"]
