from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import click

from metraq_gnn.comparison import (
    append_matched_candidate, build_comparison, load_kernel_matched, load_matched, summarize,
)


def test_load_matched_requires_identical_keys_and_summarizes(tmp_path: Path):
    root = tmp_path / "gnn"
    reference_rows = []
    for group in range(10):
        group_name = f"{group}-sensor"
        directory = root / f"group-{group + 1:02d}"
        directory.mkdir(parents=True)
        row = {
            "sensor_group": group_name,
            "window_end": "2024-01-01 00:00:00",
            "sensor_id": group,
            "method": "GNN_TRAFFIC",
            "target": 2.0,
            "prediction": 3.0,
            "error": 1.0,
            "absolute_error": 1.0,
            "squared_error": 1.0,
        }
        pd.DataFrame([row | {"target": 2.0 + 1e-6}]).to_csv(
            directory / "predictions.csv", index=False
        )
        for method in ("DIP", "KRG", "IDW"):
            reference_rows.append(row | {"method": method})
    reference = tmp_path / "reference.csv"
    pd.DataFrame(reference_rows).to_csv(reference, index=False)

    matched = load_matched(root, reference)
    metrics = summarize(matched, ["method"]).set_index("method")

    assert len(matched) == 40
    assert metrics.loc["GNN_TRAFFIC", "mae"] == 1.0
    assert metrics.loc["GNN_TRAFFIC", "wape"] == 0.5


def candidate_example():
    common = dict(sensor_group="1-2-3-4", window_end=pd.Timestamp("2024-01-01"),
                  sensor_id=1, target=160., prediction=162., error=2.,
                  absolute_error=2., squared_error=4.)
    reference = pd.DataFrame([dict(common, method=m) for m in ("DIP", "KRG", "IDW", "GNN_TRAFFIC")])
    candidate = pd.DataFrame([dict(common, method="CONVEX_KERNEL_POWER_WIND_DIRECTIONAL",
                                   target=float(np.nextafter(np.float32(160), np.float32(200))),
                                   prediction=161.)])
    return reference, candidate


def test_kernel_comparison_canonicalizes_float32_targets_and_includes_gnn():
    reference, candidate = candidate_example()
    method = candidate.method.iloc[0]
    result = append_matched_candidate(reference, candidate, method)
    kernel = result[result.method == method].iloc[0]
    assert kernel.target == 160. and kernel.absolute_error == 1.
    tables = build_comparison(result, candidate_method=method)
    assert len(tables["overall"]) == 5
    paired = tables["paired"]
    assert set(paired.comparator) == {"DIP", "KRG", "IDW", "GNN_TRAFFIC"}
    assert paired.candidate_wins.eq(1).all()


@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong-target", "nonfinite", "wrong-group"])
def test_kernel_rejects_unmatched_or_invalid_predictions(defect):
    reference, candidate = candidate_example()
    method = candidate.method.iloc[0]
    if defect == "missing":
        candidate = candidate.iloc[:0]
    elif defect == "duplicate":
        candidate = pd.concat([candidate, candidate])
    elif defect == "wrong-target":
        candidate.loc[0, "target"] += .01
    elif defect == "nonfinite":
        candidate.loc[0, "prediction"] = np.nan
    else:
        candidate.loc[0, "sensor_group"] = "5-6-7-8"
    with pytest.raises(click.ClickException):
        append_matched_candidate(reference, candidate, method)


def test_kernel_requires_all_ten_frozen_test_exports(tmp_path):
    reference, _ = candidate_example()
    with pytest.raises(click.ClickException, match="expected 10 kernel"):
        load_kernel_matched(reference, tmp_path)
