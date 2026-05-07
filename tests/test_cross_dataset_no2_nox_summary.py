from __future__ import annotations

from pathlib import Path

import pandas as pd

from metraq_dip.utils.cross_dataset_no2_nox_summary import (
    ExperimentSpec,
    build_paper_performance_data,
    render_paper_performance_table,
)


def test_build_paper_performance_data_keeps_values_unformatted() -> None:
    spec = ExperimentSpec(
        key="experiment_baseline",
        folder=Path("experiment_baseline"),
        dataset="METRAQ",
        pollutant="NO",
        setup="baseline",
        pollutants=[8],
        normalize=True,
    )
    frames = {
        spec.key: pd.DataFrame(
            {
                "DIP_L1Loss": [1.0, 2.0, 3.0],
                "KRG_L1Loss": [2.0, 3.0, 4.0],
                "IDW_L1Loss": [3.0, 4.0, 5.0],
                "DIP_MSELoss": [1.0, 4.0, 9.0],
                "KRG_MSELoss": [4.0, 9.0, 16.0],
                "IDW_MSELoss": [9.0, 16.0, 25.0],
            }
        )
    }
    wape_frames = {
        spec.key: pd.DataFrame(
            {
                "DIP_WAPE": [0.1, 0.2, 0.3],
                "KRG_WAPE": [0.2, 0.3, 0.4],
                "IDW_WAPE": [0.3, 0.4, 0.5],
            }
        )
    }

    summary = build_paper_performance_data([spec], frames, wape_frames)

    assert list(summary["Metric"]) == ["MAE", "MSE", "WAPE"]
    assert summary.loc[summary["Metric"] == "MAE", "DIP mean"].iloc[0] == 2.0
    assert summary.loc[summary["Metric"] == "MAE", "DIP win pct"].iloc[0] == 100.0
    assert summary.loc[summary["Metric"] == "MAE", "DIP vs KRG direction"].iloc[0] == "DIP lower"
    assert isinstance(summary.loc[summary["Metric"] == "MAE", "DIP vs KRG pvalue"].iloc[0], float)


def test_render_paper_performance_table_formats_markdown() -> None:
    summary = pd.DataFrame(
        [
            {
                "Dataset": "METRAQ",
                "Pollutant": "NO",
                "Setup": "baseline",
                "Metric": "MAE",
                "DIP mean": 1.0,
                "KRG mean": 2.0,
                "IDW mean": 3.0,
                "DIP win pct": 100.0,
                "KRG win pct": 0.0,
                "IDW win pct": 0.0,
                "DIP vs KRG direction": "DIP lower",
                "DIP vs KRG pvalue": 0.01,
                "DIP vs IDW direction": "DIP lower",
                "DIP vs IDW pvalue": 0.2,
                "KRG vs IDW direction": "KRG lower",
                "KRG vs IDW pvalue": 0.03,
            }
        ]
    )

    markdown = render_paper_performance_table(summary)

    assert "| Dataset | Pollutant | Setup | Metric | DIP | KRG | IDW | DIP win % |" in markdown
    assert "| METRAQ | NO | baseline | MAE | **1.000** | 2.000 | 3.000 | **100.0** |" in markdown
    assert "DIP lower (p=.010)" in markdown
    assert "DIP lower, n.s. (p=.200)" in markdown
