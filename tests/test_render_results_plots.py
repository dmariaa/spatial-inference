from __future__ import annotations

import pandas as pd
import pytest

from metraq_dip.utils import compute_results_data, render_results_plots


def _results_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_window": [
                "2024-01-01 08:00:00",
                "2024-01-01 10:00:00",
                "2024-01-02 08:00:00",
            ],
            "sensor_group": ["g1", "g1", "g2"],
            "processed": ["True", "False", True],
            "DIP_L1Loss": [0.1, 0.2, 0.3],
            "DIP_MSELoss": [0.01, 0.04, 0.09],
            "KRG_L1Loss": [0.2, 0.3, 0.4],
            "KRG_MSELoss": [0.02, 0.05, 0.16],
            "IDW_L1Loss": [0.3, 0.4, 0.5],
            "IDW_MSELoss": [0.03, 0.06, 0.25],
        }
    )


def test_compute_loss_distribution_data_keeps_only_processed_cases():
    distribution = compute_results_data.compute_loss_histogram_plot_data(
        _results_df(),
        nbins=2,
        histnorm="count",
    )

    assert len(distribution) == 12
    assert set(distribution["Metric"]) == {"MAE", "MSE"}
    assert set(distribution["Method"]) == {"DIP", "KRG", "IDW"}
    dip_mae = distribution.loc[
        (distribution["Metric"] == "MAE") & (distribution["Method"] == "DIP"),
    ]
    assert dip_mae["Count"].sum() == 2
    assert dip_mae["Sample count"].iloc[0] == 2
    assert dip_mae["Normalization"].iloc[0] == "count"


def test_build_loss_histogram_figure_builds_two_metric_panels():
    distribution = compute_results_data.compute_loss_histogram_plot_data(
        _results_df(),
        nbins=2,
        histnorm="count",
    )
    fig = render_results_plots.build_loss_histogram_figure(
        df=distribution,
    )

    assert len(fig.data) == 6
    assert fig.layout.barmode == "overlay"
    assert fig.layout.xaxis.title.text == "MAE loss"
    assert fig.layout.xaxis2.title.text == "MSE loss"
    assert {trace.legendgroup for trace in fig.data} == {"DIP", "KRG", "IDW"}
    assert "2 processed experiments" in fig.layout.title.text


def test_load_loss_distribution_data_rejects_missing_required_columns(tmp_path):
    data_path = tmp_path / "loss_histogram_plot_data.csv"
    pd.DataFrame({"Metric": ["MAE"], "Method": ["DIP"]}).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        render_results_plots.load_loss_distribution_data(tmp_path)


def test_compute_pairwise_difference_histogram_plot_data_builds_signed_bins():
    pairwise = compute_results_data.compute_pairwise_difference_histogram_plot_data(
        _results_df(),
        nbins=2,
        histnorm="count",
    )

    assert len(pairwise) == 12
    assert set(pairwise["Metric"]) == {"MAE", "MSE"}
    assert set(pairwise["Comparison"]) == {"DIP - KRG", "DIP - IDW", "KRG - IDW"}
    dip_minus_krg = pairwise.loc[
        (pairwise["Metric"] == "MAE") & (pairwise["Comparison"] == "DIP - KRG")
    ]
    assert dip_minus_krg["Count"].sum() == 2
    assert (dip_minus_krg["Bin right"] <= 0).any()


def test_build_pairwise_difference_histogram_figure_builds_two_metric_panels():
    pairwise = compute_results_data.compute_pairwise_difference_histogram_plot_data(
        _results_df(),
        nbins=2,
        histnorm="count",
    )
    fig = render_results_plots.build_pairwise_difference_histogram_figure(df=pairwise)

    assert len(fig.data) == 6
    assert fig.layout.barmode == "overlay"
    assert fig.layout.xaxis.title.text == "MAE difference"
    assert fig.layout.xaxis2.title.text == "MSE difference"
    assert {trace.legendgroup for trace in fig.data} == {"DIP - KRG", "DIP - IDW", "KRG - IDW"}
