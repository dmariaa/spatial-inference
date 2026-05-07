from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

from metraq_dip.tools.interpolator import IdwInterpolator, KrigingInterpolator
from metraq_dip.tools.tools import calculate_interpolations


DEFAULT_DIAGNOSTICS_DIR = Path("output/experiments/pollutant_diagnostics")
METHODS = ("DIP", "KRG", "IDW")


def _experiment_file(root: Path, sensor_group: str, time_window: str | pd.Timestamp) -> Path:
    ts = pd.Timestamp(time_window)
    return root / f"exp_{sensor_group}_{ts.strftime('%Y%m%dT%H%M%S')}.npz"


def _load_config(root: Path) -> dict[str, Any]:
    with (root / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _denormalize_masked(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    pollutant: int,
    normalization_stats: dict[int, tuple[float, float]],
) -> np.ndarray:
    restored = np.array(data, copy=True, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=bool)
    mean, std = normalization_stats[pollutant]
    restored[mask_array] = restored[mask_array] * (std + 1e-6) + mean
    restored[~mask_array] = 0.0
    return restored


def _denormalize_full(
    data: np.ndarray,
    *,
    pollutant: int,
    normalization_stats: dict[int, tuple[float, float]],
) -> np.ndarray:
    mean, std = normalization_stats[pollutant]
    return np.asarray(data, dtype=np.float32) * (std + 1e-6) + mean


def _load_window_arrays(exp_path: Path, config: dict[str, Any]) -> dict[str, np.ndarray]:
    pollutant = int(config["pollutants"][0])
    normalize = bool(config.get("normalize", False))

    with np.load(exp_path, allow_pickle=True) as experiment:
        train_data = np.asarray(experiment["train_data"][0, 0, -1], dtype=np.float32)
        val_data = np.asarray(experiment["val_data"][0, 0, -1], dtype=np.float32)
        test_data = np.asarray(experiment["test_data"][0, 0, -1], dtype=np.float32)
        train_mask = np.asarray(experiment["train_mask"][0, 0, -1], dtype=bool)
        val_mask = np.asarray(experiment["val_mask"][0, 0, -1], dtype=bool)
        test_mask = np.asarray(experiment["test_mask"][0, 0, -1], dtype=bool)

        dip_surface = np.asarray(experiment["train_output"], dtype=np.float32)
        if normalize:
            normalization_stats = experiment["normalization_stats"].item()
            train_data = _denormalize_masked(
                train_data,
                train_mask,
                pollutant=pollutant,
                normalization_stats=normalization_stats,
            )
            val_data = _denormalize_masked(
                val_data,
                val_mask,
                pollutant=pollutant,
                normalization_stats=normalization_stats,
            )
            test_data = _denormalize_masked(
                test_data,
                test_mask,
                pollutant=pollutant,
                normalization_stats=normalization_stats,
            )
            if "train_output_real" in experiment:
                dip_surface = np.asarray(experiment["train_output_real"], dtype=np.float32)
            else:
                dip_surface = _denormalize_full(
                    dip_surface,
                    pollutant=pollutant,
                    normalization_stats=normalization_stats,
                )

    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "DIP": dip_surface,
    }


def _baseline_surface(
    *,
    train_data: np.ndarray,
    val_data: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    method: str,
) -> np.ndarray:
    observed_mask = train_mask | val_mask
    observed_data = np.where(train_mask, train_data, val_data)
    interpolator = KrigingInterpolator if method == "KRG" else IdwInterpolator
    surface = calculate_interpolations(
        observed_data[None, None, ...],
        observed_mask[None, None, ...],
        interpolator,
    )
    return np.asarray(surface[0, 0], dtype=np.float32)


def _masked_losses(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(mask, dtype=bool)
    diff = np.asarray(y_pred, dtype=np.float32)[valid] - np.asarray(y_true, dtype=np.float32)[valid]
    return float(np.abs(diff).mean()), float(np.square(diff).mean())


def _points(mask: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.where(mask)
    return x, y, np.asarray(data, dtype=np.float32)[y, x]


def _select_rows(
    row_metrics: pd.DataFrame,
    *,
    pollutant: str,
    cases: list[str],
    rank: int,
    selection: str,
) -> pd.DataFrame:
    selected: list[pd.Series] = []
    pollutant_rows = row_metrics[row_metrics["pollutant"] == pollutant].copy()

    for case in cases:
        ascending = case == "best_l1_vs_krg"
        window_scores = (
            pollutant_rows.groupby("time_window", as_index=False)["gap_l1_dip_minus_krg"]
            .mean()
            .sort_values("gap_l1_dip_minus_krg", ascending=ascending)
            .reset_index(drop=True)
        )
        if rank > len(window_scores):
            raise ValueError(f"Rank {rank} is outside available windows for {pollutant} {case}.")

        window = window_scores.iloc[rank - 1]
        same_window = pollutant_rows[pollutant_rows["time_window"] == window["time_window"]].copy()
        if selection == "representative":
            same_window["distance_to_window_gap"] = (
                same_window["gap_l1_dip_minus_krg"] - window["gap_l1_dip_minus_krg"]
            ).abs()
            row = same_window.sort_values("distance_to_window_gap").iloc[0].copy()
        else:
            row = same_window.sort_values("gap_l1_dip_minus_krg", ascending=ascending).iloc[0].copy()

        row["case_type"] = case
        row["window_gap_l1_dip_minus_krg"] = window["gap_l1_dip_minus_krg"]
        selected.append(row)

    return pd.DataFrame(selected)


def _add_surface_panel(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    method: str,
    surface: np.ndarray,
    arrays: dict[str, np.ndarray],
    zmin: float,
    zmax: float,
    residual_max: float,
) -> None:
    fig.add_trace(
        go.Heatmap(
            z=surface,
            coloraxis="coloraxis",
            hovertemplate="x=%{x}<br>y=%{y}<br>prediction=%{z:.2f}<extra>" + method + "</extra>",
        ),
        row=row,
        col=col,
    )

    for name, marker, color in (
        ("train", "circle", "#1f77b4"),
        ("val", "diamond", "#ffbf00"),
    ):
        x, y, values = _points(arrays[f"{name}_mask"], arrays[f"{name}_data"])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                legendgroup=name,
                showlegend=(col == 1),
                marker={"symbol": marker, "color": color, "size": 8, "line": {"color": "white", "width": 1}},
                customdata=values,
                hovertemplate=f"{name}<br>x=%{{x}}<br>y=%{{y}}<br>actual=%{{customdata:.2f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    test_x, test_y, test_values = _points(arrays["test_mask"], arrays["test_data"])
    test_pred = surface[test_y, test_x]
    abs_error = np.abs(test_pred - test_values)
    fig.add_trace(
        go.Scatter(
            x=test_x,
            y=test_y,
            mode="markers",
            name="test abs error",
            legendgroup="test",
            showlegend=(col == 1),
            marker={
                "symbol": "x",
                "size": 12,
                "color": abs_error,
                "colorscale": "Reds",
                "cmin": 0,
                "cmax": residual_max,
                "line": {"color": "black", "width": 1},
                "colorbar": {
                    "title": "|error|",
                    "x": 1.07,
                    "y": 0.74,
                    "len": 0.34,
                    "thickness": 14,
                    "yanchor": "middle",
                }
                if col == 3
                else None,
            },
            customdata=np.stack([test_values, test_pred, test_pred - test_values], axis=1),
            hovertemplate=(
                "test<br>x=%{x}<br>y=%{y}<br>"
                "actual=%{customdata[0]:.2f}<br>"
                "pred=%{customdata[1]:.2f}<br>"
                "residual=%{customdata[2]:.2f}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )

    fig.update_xaxes(title_text=method, row=row, col=col)
    fig.update_yaxes(autorange="reversed", scaleanchor=f"x{col}", scaleratio=1, row=row, col=col)
    fig.update_traces(zmin=zmin, zmax=zmax, selector={"type": "heatmap"})


def _render_window(
    *,
    row: pd.Series,
    outdir: Path,
) -> dict[str, Any]:
    root = Path(row["root"])
    config = _load_config(root)
    exp_path = _experiment_file(root, str(row["sensor_group"]), row["time_window"])
    arrays = _load_window_arrays(exp_path, config)

    surfaces = {
        "DIP": arrays["DIP"],
        "KRG": _baseline_surface(
            train_data=arrays["train_data"],
            val_data=arrays["val_data"],
            train_mask=arrays["train_mask"],
            val_mask=arrays["val_mask"],
            method="KRG",
        ),
        "IDW": _baseline_surface(
            train_data=arrays["train_data"],
            val_data=arrays["val_data"],
            train_mask=arrays["train_mask"],
            val_mask=arrays["val_mask"],
            method="IDW",
        ),
    }

    metrics = {
        method: _masked_losses(arrays["test_data"], surface, arrays["test_mask"])
        for method, surface in surfaces.items()
    }
    test_y, test_x = np.where(arrays["test_mask"])
    test_values = arrays["test_data"][test_y, test_x]
    predictions = {method: surface[test_y, test_x] for method, surface in surfaces.items()}
    residual_max = max(
        float(np.max(np.abs(predictions[method] - test_values)))
        for method in METHODS
    )

    observed_values = np.concatenate(
        [
            arrays["train_data"][arrays["train_mask"]],
            arrays["val_data"][arrays["val_mask"]],
            arrays["test_data"][arrays["test_mask"]],
        ]
    )
    zmin = float(np.nanmin([*(surface.min() for surface in surfaces.values()), observed_values.min()]))
    zmax = float(np.nanmax([*(surface.max() for surface in surfaces.values()), observed_values.max()]))

    titles = [f"{method} surface" for method in METHODS] + [
        "Test values",
        "Absolute error",
        "Window metrics",
    ]
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "table"}],
        ],
        subplot_titles=titles,
        vertical_spacing=0.16,
    )

    for col, method in enumerate(METHODS, start=1):
        _add_surface_panel(
            fig,
            row=1,
            col=col,
            method=method,
            surface=surfaces[method],
            arrays=arrays,
            zmin=zmin,
            zmax=zmax,
            residual_max=residual_max,
        )

    x_labels = [f"({int(x)},{int(y)})" for y, x in zip(test_y, test_x)]
    fig.add_trace(go.Bar(name="actual", x=x_labels, y=test_values, marker_color="#555555"), row=2, col=1)
    for method, color in zip(METHODS, ("#00897b", "#5e35b1", "#f4511e")):
        fig.add_trace(go.Bar(name=method, x=x_labels, y=predictions[method], marker_color=color), row=2, col=1)
        fig.add_trace(
            go.Bar(
                name=f"{method} |error|",
                x=x_labels,
                y=np.abs(predictions[method] - test_values),
                marker_color=color,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    table_rows = [
        ["case", str(row["case_type"])],
        ["time", str(row["time_window"])],
        ["sensor_group", str(row["sensor_group"])],
        ["window DIP-KRG L1", f"{float(row['window_gap_l1_dip_minus_krg']):.3f}"],
        ["row DIP/KRG/IDW L1", " / ".join(f"{metrics[m][0]:.3f}" for m in METHODS)],
        ["row DIP/KRG/IDW MSE", " / ".join(f"{metrics[m][1]:.3f}" for m in METHODS)],
        ["train/val/test cells", f"{arrays['train_mask'].sum()} / {arrays['val_mask'].sum()} / {arrays['test_mask'].sum()}"],
        ["target mean abs", f"{float(row['target_mean_abs']):.3f}"],
        ["target p90-p10", f"{float(row['target_p90_p10']):.3f}"],
    ]
    fig.add_trace(
        go.Table(
            header={"values": ["field", "value"], "fill_color": "#eeeeee", "align": "left"},
            cells={"values": [list(x) for x in zip(*table_rows)], "align": "left"},
        ),
        row=2,
        col=3,
    )

    pollutant = str(row["pollutant"]).lower()
    case = str(row["case_type"]).replace("_l1_vs_krg", "")
    timestamp = pd.Timestamp(row["time_window"]).strftime("%Y%m%dT%H%M%S")
    filename = f"{pollutant}_{case}_{timestamp}_{row['sensor_group']}.html"
    output_file = outdir / filename

    fig.update_layout(
        title=(
            f"{row['pollutant']} {row['case_type']} | "
            f"{pd.Timestamp(row['time_window']).strftime('%Y-%m-%d %H:%M')}"
        ),
        coloraxis={
            "colorscale": "Viridis",
            "cmin": zmin,
            "cmax": zmax,
            "colorbar": {
                "title": "value",
                "x": 1.02,
                "y": 0.74,
                "len": 0.34,
                "thickness": 14,
                "yanchor": "middle"
            },
        },
        barmode="group",
        legend={
            "orientation": "v",
            "x": 1.09,
            "xanchor": "center",
            "y": 0.35,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "rgba(0,0,0,0.15)",
            "borderwidth": 1,
            "itemsizing": "constant",
        },
        height=980,
        width=1320,
        margin={"l": 50, "r": 140, "t": 95, "b": 125},
    )
    fig.write_html(output_file)

    return {
        "pollutant": row["pollutant"],
        "case_type": row["case_type"],
        "time_window": row["time_window"],
        "sensor_group": row["sensor_group"],
        "window_gap_l1_dip_minus_krg": row["window_gap_l1_dip_minus_krg"],
        "DIP_L1": metrics["DIP"][0],
        "KRG_L1": metrics["KRG"][0],
        "IDW_L1": metrics["IDW"][0],
        "DIP_MSE": metrics["DIP"][1],
        "KRG_MSE": metrics["KRG"][1],
        "IDW_MSE": metrics["IDW"][1],
        "target_mean_abs": row["target_mean_abs"],
        "target_p90_p10": row["target_p90_p10"],
        "train_cells": int(arrays["train_mask"].sum()),
        "val_cells": int(arrays["val_mask"].sum()),
        "test_cells": int(arrays["test_mask"].sum()),
        "html": str(output_file),
    }


def main(
    *,
    diagnostics_dir: Path = DEFAULT_DIAGNOSTICS_DIR,
    outdir: Path | None = None,
    pollutants: list[str] | None = None,
    cases: list[str] | None = None,
    rank: int = 1,
    selection: str = "representative",
) -> None:
    pollutants = pollutants or ["NO", "NO2", "NOX"]
    cases = cases or ["best_l1_vs_krg", "worst_l1_vs_krg"]

    diagnostics_dir = diagnostics_dir.resolve()
    outdir = (outdir or diagnostics_dir / "window_visualizations").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    row_metrics = pd.read_csv(diagnostics_dir / "row_level_metrics.csv", parse_dates=["time_window"])
    manifests: list[dict[str, Any]] = []
    for pollutant in pollutants:
        selected = _select_rows(
            row_metrics,
            pollutant=pollutant,
            cases=cases,
            rank=rank,
            selection=selection,
        )
        for _, row in selected.iterrows():
            manifests.append(_render_window(row=row, outdir=outdir))

    manifest = pd.DataFrame(manifests)
    manifest_file = outdir / "manifest.csv"
    manifest.to_csv(manifest_file, index=False)

    print(f"Wrote {len(manifest)} visualizations to {outdir}")
    print(f"Wrote {manifest_file}")


if __name__ == "__main__":
    import click

    @click.command(context_settings={"show_default": True})
    @click.option(
        "--diagnostics-dir",
        type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
        default=DEFAULT_DIAGNOSTICS_DIR,
    )
    @click.option(
        "--outdir",
        type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
        default=None,
    )
    @click.option("--pollutants", multiple=True)
    @click.option(
        "--cases",
        multiple=True,
        type=click.Choice(["best_l1_vs_krg", "worst_l1_vs_krg"]),
    )
    @click.option("--rank", type=int, default=1, help="Window rank within each pollutant/case.")
    @click.option(
        "--selection",
        type=click.Choice(["representative", "strongest"]),
        default="representative",
        help="Pick a sensor group closest to the window-average gap, or the strongest row in that window.",
    )
    def cli(
        diagnostics_dir: Path,
        outdir: Path | None,
        pollutants: tuple[str, ...],
        cases: tuple[str, ...],
        rank: int,
        selection: str,
    ) -> None:
        main(
            diagnostics_dir=diagnostics_dir,
            outdir=outdir,
            pollutants=list(pollutants) or None,
            cases=list(cases) or None,
            rank=rank,
            selection=selection,
        )

    cli()
