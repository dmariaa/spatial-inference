from pathlib import Path

import pandas as pd

from metraq_gnn.comparison import load_matched, summarize


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
