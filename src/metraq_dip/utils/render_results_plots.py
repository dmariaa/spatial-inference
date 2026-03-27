#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import kaleido
kaleido.get_chrome_sync()

import pandas as pd
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from metraq_dip.utils.compute_results_data import (
    HISTOGRAM_VALUE_LABELS,
    LOSS_METRICS,
    METHODS,
    PAIRWISE_COMPARISONS,
)

METHOD_COLORS = {
    "DIP": "#1f77b4",
    "KRG": "#ff7f0e",
    "IDW": "#2ca02c",
}
PAIRWISE_COMPARISON_COLORS = {
    "DIP - KRG": "#d62728",
    "DIP - IDW": "#9467bd",
    "KRG - IDW": "#8c564b",
}
METHOD_DISPLAY_NAMES = {
    "DIP": "DIP",
    "KRG": "Kriging",
    "IDW": "IDW",
}

HISTNORM_LABELS = {None: "Count", "": "Count", **HISTOGRAM_VALUE_LABELS}
PAPER_TICK_FONT_SIZE = 10
PAPER_AXIS_FONT_SIZE = 11
PAPER_TEXT_FONT_SIZE = 12
PAPER_PANEL_TITLE_FONT_SIZE = 13
PAPER_TITLE_FONT_SIZE = 15
# PAPER_TICK_FONT_SIZE = 12
# PAPER_AXIS_FONT_SIZE = 13
# PAPER_TEXT_FONT_SIZE = 14
# PAPER_PANEL_TITLE_FONT_SIZE = 15
# PAPER_TITLE_FONT_SIZE = 17


def _validate_loss_distribution_data(df: pd.DataFrame) -> None:
    required_columns = {
        "Metric",
        "Method",
        "Bin index",
        "Bin left",
        "Bin right",
        "Bin center",
        "Bin width",
        "Count",
        "Value",
        "Normalization",
        "Y label",
        "Sample count",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Loss distribution data is missing required columns: {missing_text}")

    if df.empty:
        raise ValueError("Loss distribution data is empty.")


def _validate_pairwise_difference_data(df: pd.DataFrame) -> None:
    required_columns = {
        "Metric",
        "Comparison",
        "Left method",
        "Right method",
        "Bin index",
        "Bin left",
        "Bin right",
        "Bin center",
        "Bin width",
        "Count",
        "Value",
        "Normalization",
        "Y label",
        "Sample count",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Pairwise difference histogram data is missing required columns: {missing_text}")

    if df.empty:
        raise ValueError("Pairwise difference histogram data is empty.")


def load_loss_distribution_data(datadir: str | Path) -> pd.DataFrame:
    data_path = Path(datadir) / "loss_histogram_plot_data.csv"
    df = pd.read_csv(data_path)
    _validate_loss_distribution_data(df)
    return df


def load_pairwise_difference_data(datadir: str | Path) -> pd.DataFrame:
    data_path = Path(datadir) / "pairwise_difference_histogram_plot_data.csv"
    df = pd.read_csv(data_path)
    _validate_pairwise_difference_data(df)
    return df


def _pairwise_comparison_label(left_method: str, right_method: str) -> str:
    return f"{METHOD_DISPLAY_NAMES[left_method]} vs {METHOD_DISPLAY_NAMES[right_method]}"


def build_loss_histogram_figure(
    *,
    df: pd.DataFrame,
    histnorm: str | None = None,
    title: str | None = None,
) -> go.Figure:
    plotted_df = df.copy()
    _validate_loss_distribution_data(plotted_df)
    case_count = int(plotted_df["Sample count"].iloc[0])
    if histnorm is None:
        histnorm_values = plotted_df["Normalization"].dropna().unique().tolist()
        histnorm = histnorm_values[0] if histnorm_values else "count"
    y_axis_title = HISTNORM_LABELS.get(histnorm, str(histnorm))
    method_specs = [
        {
            "data_label": method,
            "display_label": METHOD_DISPLAY_NAMES[method],
        }
        for method in METHODS
    ]
    fig = make_subplots(
        rows=2,
        cols=3,
        column_titles=[spec["display_label"] for spec in method_specs],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
        shared_yaxes="rows",
    )

    for row_index, (metric_name, _suffix) in enumerate(LOSS_METRICS, start=1):
        for column_index, method_spec in enumerate(method_specs, start=1):
            method = method_spec["data_label"]
            series = (
                plotted_df.loc[
                    (plotted_df["Metric"] == metric_name) & (plotted_df["Method"] == method)
                ]
                .sort_values("Bin index")
                .reset_index(drop=True)
            )
            fig.add_trace(
                go.Bar(
                    x=series["Bin center"],
                    y=series["Value"],
                    width=series["Bin width"],
                    name=method_spec["display_label"],
                    legendgroup=method,
                    showlegend=False,
                    opacity=0.85,
                    marker=dict(
                        color=METHOD_COLORS[method],
                        line=dict(color="rgba(255,255,255,0.9)", width=0.8),
                    ),
                    customdata=np.column_stack(
                        [
                            series["Bin left"],
                            series["Bin right"],
                            series["Count"],
                        ]
                    ),
                    hovertemplate=(
                        f"{method_spec['display_label']}<br>{metric_name}: "
                        f"[%{{customdata[0]:.6f}}, %{{customdata[1]:.6f}}]"
                        f"<br>Count: %{{customdata[2]}}<br>{y_axis_title}: %{{y:.6f}}<extra></extra>"
                    ),
                ),
                row=row_index,
                col=column_index,
            )

            fig.update_xaxes(
                title_text="Loss" if row_index == len(LOSS_METRICS) else None,
                title_font=dict(size=PAPER_AXIS_FONT_SIZE),
                title_standoff=4,
                tickfont=dict(size=PAPER_TICK_FONT_SIZE),
                showticklabels=row_index == len(LOSS_METRICS),
                row=row_index,
                col=column_index,
            )
            if column_index == 1:
                fig.update_yaxes(
                    title_text=f"{metric_name}<br>{y_axis_title}",
                    title_font=dict(size=PAPER_AXIS_FONT_SIZE),
                    title_standoff=4,
                    tickfont=dict(size=PAPER_TICK_FONT_SIZE),
                    row=row_index,
                    col=column_index,
                )
            else:
                fig.update_yaxes(
                    tickfont=dict(size=PAPER_TICK_FONT_SIZE),
                    showticklabels=False,
                    row=row_index,
                    col=column_index,
                )

    fig.update_layout(
        barmode="overlay",
        bargap=0.03,
        template="plotly_white",
        font=dict(size=PAPER_TEXT_FONT_SIZE),
        margin=dict(l=42, r=6, t=28 if title is None else 44, b=28),
    )
    if title is not None:
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=PAPER_TITLE_FONT_SIZE))
        )
    fig.update_annotations(font=dict(size=PAPER_PANEL_TITLE_FONT_SIZE))
    return fig


def build_pairwise_difference_histogram_figure(
    *,
    df: pd.DataFrame,
    histnorm: str | None = None,
    title: str | None = None,
) -> go.Figure:
    plotted_df = df.copy()
    _validate_pairwise_difference_data(plotted_df)
    case_count = int(plotted_df["Sample count"].iloc[0])
    if histnorm is None:
        histnorm_values = plotted_df["Normalization"].dropna().unique().tolist()
        histnorm = histnorm_values[0] if histnorm_values else "count"
    y_axis_title = HISTNORM_LABELS.get(histnorm, str(histnorm))
    comparison_specs = [
        {
            "data_label": f"{left} - {right}",
            "display_label": _pairwise_comparison_label(left, right),
        }
        for left, right in PAIRWISE_COMPARISONS
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        column_titles=[spec["display_label"] for spec in comparison_specs],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
        shared_xaxes="rows",
        shared_yaxes="rows",
    )

    for row_index, (metric_name, _suffix) in enumerate(LOSS_METRICS, start=1):
        for column_index, comparison_spec in enumerate(comparison_specs, start=1):
            comparison = comparison_spec["data_label"]
            series = (
                plotted_df.loc[
                    (plotted_df["Metric"] == metric_name) & (plotted_df["Comparison"] == comparison)
                ]
                .sort_values("Bin index")
                .reset_index(drop=True)
            )
            fig.add_trace(
                go.Bar(
                    x=series["Bin center"],
                    y=series["Value"],
                    width=series["Bin width"],
                    name=comparison_spec["display_label"],
                    legendgroup=comparison,
                    showlegend=False,
                    opacity=0.85,
                    marker=dict(
                        color=PAIRWISE_COMPARISON_COLORS[comparison],
                        line=dict(color="rgba(255,255,255,0.9)", width=0.8),
                    ),
                    customdata=np.column_stack(
                        [
                            series["Bin left"],
                            series["Bin right"],
                            series["Count"],
                        ]
                    ),
                    hovertemplate=(
                        f"{comparison_spec['display_label']}<br>{metric_name} difference: "
                        f"[%{{customdata[0]:.6f}}, %{{customdata[1]:.6f}}]"
                        f"<br>Count: %{{customdata[2]}}<br>{y_axis_title}: %{{y:.6f}}<extra></extra>"
                    ),
                ),
                row=row_index,
                col=column_index,
            )

            fig.update_xaxes(
                title_text="Difference (first - second)" if row_index == len(LOSS_METRICS) else None,
                title_font=dict(size=PAPER_AXIS_FONT_SIZE),
                title_standoff=4,
                tickfont=dict(size=PAPER_TICK_FONT_SIZE),
                showticklabels=row_index == len(LOSS_METRICS),
                row=row_index,
                col=column_index,
            )
            if column_index == 1:
                fig.update_yaxes(
                    title_text=f"{metric_name}<br>{y_axis_title}",
                    title_font=dict(size=PAPER_AXIS_FONT_SIZE),
                    title_standoff=4,
                    tickfont=dict(size=PAPER_TICK_FONT_SIZE),
                    row=row_index,
                    col=column_index,
                )
            else:
                fig.update_yaxes(
                    tickfont=dict(size=PAPER_TICK_FONT_SIZE),
                    showticklabels=False,
                    row=row_index,
                    col=column_index,
                )
            fig.add_vline(
                x=0.0,
                line_dash="dash",
                line_color="#444444",
                line_width=1,
                row=row_index,
                col=column_index,
            )

    fig.update_layout(
        barmode="overlay",
        bargap=0.03,
        template="plotly_white",
        font=dict(size=PAPER_TEXT_FONT_SIZE),
        margin=dict(l=42, r=6, t=28 if title is None else 44, b=28),
    )
    if title is not None:
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", font=dict(size=PAPER_TITLE_FONT_SIZE))
        )
    fig.update_annotations(font=dict(size=PAPER_PANEL_TITLE_FONT_SIZE))
    return fig


def write_figure_outputs(
    *,
    fig: go.Figure,
    outdir: Path,
    stem: str = "loss_histograms",
    html: bool = True,
    static_format: str | None = None,
    width: int = 1400,
    height: int = 600,
    scale: int = 2,
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    if html:
        html_path = outdir / f"{stem}.html"
        fig.write_html(html_path, include_plotlyjs="cdn")
        written_paths.append(html_path)

    if static_format:
        static_path = outdir / f"{stem}.{static_format}"
        fig.write_image(static_path, width=width, height=height, scale=scale)
        written_paths.append(static_path)

    return written_paths


def main(
    *,
    datadir: Path,
    outdir: Path | None = None,
    title: str | None = None,
    static_format: str | None = None,
    html: bool = True,
) -> list[Path]:
    if outdir is None:
        outdir = datadir / "plots"
    df = load_loss_distribution_data(datadir)
    fig = build_loss_histogram_figure(
        df=df,
        title=title,
    )
    pairwise_df = load_pairwise_difference_data(datadir)
    pairwise_fig = build_pairwise_difference_histogram_figure(
        df=pairwise_df,
    )

    written_paths = write_figure_outputs(
        fig=fig,
        outdir=outdir,
        stem="loss_histograms",
        html=html,
        static_format=static_format,
        width=900,
        height=430,
    )
    written_paths.extend(
        write_figure_outputs(
            fig=pairwise_fig,
            outdir=outdir,
            stem="pairwise_difference_histograms",
            html=html,
            static_format=static_format,
            width=900,
            height=430,
        )
    )

    return [path.resolve() for path in written_paths]


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument(
        "datadir",
        type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
    )
    @click.option(
        "--outdir",
        type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
        default=None,
        help="Directory for rendered plot files. Defaults to DATADIR/plots.",
    )
    @click.option(
        "--title",
        default=None,
        help="Optional figure title override.",
    )
    @click.option(
        "--static-format",
        type=click.Choice(["png", "svg", "pdf"], case_sensitive=True),
        default=None,
        help="Optional static image format to export alongside HTML.",
    )
    @click.option(
        "--html/--no-html",
        default=True,
        show_default=True,
        help="Write the interactive HTML output.",
    )
    def cli(
        datadir: Path,
        outdir: Path | None,
        title: str | None,
        static_format: str | None,
        html: bool,
    ) -> None:
        try:
            written_paths = main(
                datadir=datadir,
                outdir=outdir,
                title=title,
                static_format=static_format,
                html=html,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        for path in written_paths:
            click.echo(f"Wrote: {path}")

    cli()
