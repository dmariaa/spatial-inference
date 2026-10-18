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
    canonical_targets = (
        reference[reference["method"] == METHODS[0]][KEYS + ["target"]]
        .rename(columns={"target": "canonical_target"})
    )
    combined = combined.merge(canonical_targets, on=KEYS, validate="many_to_one")
    target_difference = (combined["target"] - combined["canonical_target"]).abs()
    if not np.allclose(
        combined["target"],
        combined["canonical_target"],
        rtol=0.0,
        atol=1e-5,
    ):
        raise click.ClickException(
            "target values differ materially between matched methods: "
            f"max_difference={target_difference.max()}"
        )
    combined["target"] = combined.pop("canonical_target")
    combined["error"] = combined["prediction"] - combined["target"]
    combined["absolute_error"] = combined["error"].abs()
    combined["squared_error"] = combined["error"] ** 2
    return combined


def append_matched_candidate(combined: pd.DataFrame, candidate: pd.DataFrame,
                             method: str) -> pd.DataFrame:
    """Add a frozen model using the historical DIP targets as canonical truth."""
    candidate = candidate.copy()
    candidate["sensor_group"] = candidate["sensor_group"].astype(str)
    candidate["window_end"] = pd.to_datetime(candidate["window_end"])
    candidate["sensor_id"] = candidate["sensor_id"].astype("int64")
    if method in combined["method"].unique() or set(candidate["method"]) != {method}:
        raise click.ClickException("unexpected or already included candidate method")
    if candidate.duplicated(KEYS).any():
        raise click.ClickException("candidate contains duplicate matched keys")
    canonical = combined.loc[combined.method == "DIP", KEYS + ["target"]].copy()
    canonical["window_end"] = pd.to_datetime(canonical["window_end"])
    canonical["sensor_group"] = canonical["sensor_group"].astype(str)
    aligned = candidate.merge(canonical.rename(columns={"target": "canonical_target"}),
                              on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise click.ClickException("candidate keys differ from historical benchmark")
    values = aligned[["target", "canonical_target", "prediction"]].to_numpy()
    if not np.isfinite(values).all():
        raise click.ClickException("candidate contains non-finite targets or predictions")
    # Inverting float32 normalization can differ by one ULP at high concentrations.
    tolerance = 1e-5 + np.finfo(np.float32).eps * aligned["canonical_target"].abs()
    if ((aligned["target"] - aligned["canonical_target"]).abs() > tolerance).any():
        raise click.ClickException("candidate targets differ materially from historical targets")
    aligned["target"] = aligned.pop("canonical_target")
    aligned["error"] = aligned["prediction"] - aligned["target"]
    aligned["absolute_error"] = aligned["error"].abs()
    aligned["squared_error"] = aligned["error"] ** 2
    return pd.concat([combined, aligned[combined.columns]], ignore_index=True)


def load_kernel_matched(combined: pd.DataFrame, kernel_root: Path) -> pd.DataFrame:
    files = sorted(kernel_root.glob("group-*/test/predictions.csv"))
    if len(files) != 10:
        raise click.ClickException(f"expected 10 kernel prediction files, found {len(files)}")
    candidate = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    return append_matched_candidate(combined, candidate, "CONVEX_KERNEL_POWER_WIND_DIRECTIONAL")


def build_comparison(combined: pd.DataFrame, *, candidate_method: str = "GNN_TRAFFIC") -> dict[str, pd.DataFrame]:
    overall = summarize(combined, ["method"])
    by_group = summarize(combined, ["sensor_group", "method"])
    by_case = summarize(combined, ["sensor_group", "window_end", "method"])
    pivot = by_case.pivot(
        index=["sensor_group", "window_end"], columns="method", values="mae"
    )
    paired = []
    prefix = "gnn" if candidate_method == "GNN_TRAFFIC" else "candidate"
    comparators = [method for method in METHODS if method != candidate_method]
    comparators += sorted(set(combined["method"]) - set(METHODS) - {candidate_method})
    for comparator in comparators:
        delta = pivot[candidate_method] - pivot[comparator]
        paired.append(
            {
                "comparator": comparator,
                "cases": len(delta),
                f"{prefix}_wins": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
                f"{prefix}_losses": int((delta > 0).sum()),
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


__all__ = ["KEYS", "METHODS", "append_matched_candidate", "build_comparison",
           "load_kernel_matched", "load_matched", "summarize"]
