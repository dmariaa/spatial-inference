import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pyproj import Transformer

from metraq_dip.data.aq_backends import get_aq_backend_for_config
from metraq_dip.data.data import get_grid
from metraq_dip.tools.config_tools import SessionConfig, load_session_config
from metraq_dip.tools.grid import DEFAULT_METRIC_CRS, find_grid_cell
from metraq_dip.tools.random_tools import sensor_group_hash


@dataclass(frozen=True)
class ExperimentSensorGroups:
    experiment_folder: Path
    config: SessionConfig
    grid_ctx: dict
    sensor_ids: list[int]
    sensor_groups: list[dict[str, list[int]]]
    group_labels: list[str]


@lru_cache(maxsize=None)
def _get_to_deg_transformer(metric_crs: str) -> Transformer:
    normalized = str(metric_crs).strip() or DEFAULT_METRIC_CRS
    return Transformer.from_crs(normalized, "epsg:4326", always_xy=True)


def _resolve_group_labels(
    sensor_groups: Sequence[dict],
    group_labels: Optional[Sequence[str]],
) -> list[str]:
    labels = (
        list(group_labels)
        if group_labels is not None
        else [sensor_group_hash(group.get("test_sensors", [])) for group in sensor_groups]
    )
    if len(labels) != len(sensor_groups):
        raise ValueError("group_labels length must match sensor_groups length")
    return labels


def _build_id_to_cell(grid_ctx: dict) -> dict[int, tuple[int, int]]:
    df = grid_ctx["df"].reset_index(drop=True)
    xs = grid_ctx["xs"]
    ys = grid_ctx["ys"]

    id_to_cell: dict[int, tuple[int, int]] = {}
    for idx, sensor_id in enumerate(df["id"].to_numpy()):
        try:
            cell = find_grid_cell(grid_ctx, xs[idx], ys[idx])
        except IndexError:
            continue
        if cell is None:
            continue
        id_to_cell[int(sensor_id)] = cell

    return id_to_cell


def _group_to_grid(
    *,
    grid_shape: tuple[int, int],
    id_to_cell: dict[int, tuple[int, int]],
    train_sensors: Iterable[int],
    val_sensors: Iterable[int],
    test_sensors: Iterable[int],
) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=np.int8)

    for sensor_id in train_sensors:
        cell = id_to_cell.get(int(sensor_id))
        if cell is None:
            continue
        r, c = cell
        grid[r, c] = 1

    for sensor_id in val_sensors:
        cell = id_to_cell.get(int(sensor_id))
        if cell is None:
            continue
        r, c = cell
        grid[r, c] = 2

    for sensor_id in test_sensors:
        cell = id_to_cell.get(int(sensor_id))
        if cell is None:
            continue
        r, c = cell
        grid[r, c] = 3

    return grid


def _cell_ring_ll(grid_ctx: dict, cell: tuple[int, int]) -> Optional[list[tuple[float, float]]]:
    grid = grid_ctx["grid"]
    r, c = cell
    cell_obj = grid[r, c]
    if cell_obj is None:
        return None

    to_deg = _get_to_deg_transformer(grid_ctx.get("metric_crs", DEFAULT_METRIC_CRS))
    x_coords, y_coords = cell_obj.exterior.coords.xy
    lons, lats = to_deg.transform(np.array(x_coords), np.array(y_coords))
    return list(zip(lats.tolist(), lons.tolist()))


def _rings_from_sensors(
    *,
    grid_ctx: dict,
    id_to_cell: dict[int, tuple[int, int]],
    sensors: Iterable[int],
    cell_rings: dict[tuple[int, int], list[tuple[float, float]]],
) -> tuple[list[float], list[float]]:
    lats: list[float] = []
    lons: list[float] = []

    for sensor_id in sensors:
        cell = id_to_cell.get(int(sensor_id))
        if cell is None:
            continue
        if cell not in cell_rings:
            ring = _cell_ring_ll(grid_ctx, cell)
            if ring is None:
                continue
            cell_rings[cell] = ring
        ring = cell_rings[cell]
        lat_ring, lon_ring = zip(*ring)
        lats.extend(lat_ring)
        lons.extend(lon_ring)
        lats.append(np.nan)
        lons.append(np.nan)

    return lats, lons


def _grid_lines_trace(grid_ctx: dict, line_color: str, line_width: int) -> go.Scattermap:
    lats: list[float] = []
    lons: list[float] = []

    for ring in grid_ctx["grid_cells_ll"]:
        lat_ring, lon_ring = zip(*ring)
        lats.extend(lat_ring)
        lons.extend(lon_ring)
        lats.append(np.nan)
        lons.append(np.nan)

    return go.Scattermap(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(color=line_color, width=line_width),
        hoverinfo="skip",
        showlegend=False,
        name="grid",
    )


def _normalize_test_sensor_groups(raw_groups: np.ndarray) -> list[list[int]]:
    if raw_groups.ndim == 0:
        raise ValueError("data.npz must store test_sensors as an array of sensor groups.")

    if raw_groups.ndim == 1:
        first_item = raw_groups[0] if raw_groups.size else None
        if raw_groups.dtype != object or np.isscalar(first_item):
            groups_iterable = [raw_groups]
        else:
            groups_iterable = raw_groups.tolist()
    else:
        groups_iterable = raw_groups.tolist()

    normalized_groups: list[list[int]] = []
    for group in groups_iterable:
        group_array = np.asarray(group).reshape(-1)
        if group_array.size == 0:
            raise ValueError("test_sensors contains an empty sensor group.")
        normalized_groups.append([int(sensor_id) for sensor_id in group_array.tolist()])

    if not normalized_groups:
        raise ValueError("data.npz does not contain any test sensor groups.")

    return normalized_groups


def load_experiment_sensor_groups(
    experiment_folder: str | Path,
    *,
    include_train_sensors: bool = True,
    group_labels: Optional[Sequence[str]] = None,
) -> ExperimentSensorGroups:
    experiment_path = Path(experiment_folder)
    config_path = experiment_path / "config.yaml"
    data_path = experiment_path / "data.npz"

    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config file not found: {config_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Experiment data file not found: {data_path}")

    config = load_session_config(config_path)
    aq_backend = get_aq_backend_for_config(config)
    grid_ctx, sensor_ids = get_grid(pollutants=list(config.pollutants), aq_backend=aq_backend)

    with np.load(data_path, allow_pickle=True) as data:
        if "test_sensors" not in data.files:
            raise ValueError(f"Experiment data file does not contain 'test_sensors': {data_path}")
        raw_groups = _normalize_test_sensor_groups(data["test_sensors"])

    sensor_id_set = {int(sensor_id) for sensor_id in sensor_ids}
    resolved_labels = _resolve_group_labels(
        [{"test_sensors": group} for group in raw_groups],
        group_labels,
    )

    sensor_groups: list[dict[str, list[int]]] = []
    for group in raw_groups:
        missing_sensors = [sensor_id for sensor_id in group if sensor_id not in sensor_id_set]
        if missing_sensors:
            warnings.warn(
                "Discarding sensor ids that are not available in the configured backend grid: "
                f"{missing_sensors}",
                stacklevel=2,
            )

        test_sensors = [sensor_id for sensor_id in group if sensor_id in sensor_id_set]
        group_sensor_set = set(group)
        train_sensors = (
            [int(sensor_id) for sensor_id in sensor_ids if int(sensor_id) not in group_sensor_set]
            if include_train_sensors
            else []
        )

        sensor_groups.append(
            {
                "train_sensors": train_sensors,
                "val_sensors": [],
                "test_sensors": test_sensors,
            }
        )

    return ExperimentSensorGroups(
        experiment_folder=experiment_path,
        config=config,
        grid_ctx=grid_ctx,
        sensor_ids=[int(sensor_id) for sensor_id in sensor_ids],
        sensor_groups=sensor_groups,
        group_labels=resolved_labels,
    )


def plot_sensor_groups(
    *,
    grid_ctx: dict,
    sensor_groups: Sequence[dict],
    n_cols: Optional[int] = None,
    group_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    cell_gap: int = 1,
    grid_line_color: str = "#cccccc",
    subplot_size: int = 480,
    subplot_gap: float = 0.02,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """
    Plot sensor groups on a grid as colored cells.

    Each item in sensor_groups must be a dict with keys:
      - "train_sensors"
      - "val_sensors"
      - "test_sensors"
    """
    if not sensor_groups:
        raise ValueError("sensor_groups is empty")

    n_groups = len(sensor_groups)
    if n_cols is None:
        n_cols = int(math.ceil(math.sqrt(n_groups)))
    n_rows = int(math.ceil(n_groups / n_cols))

    labels = _resolve_group_labels(sensor_groups, group_labels)

    grid_shape = grid_ctx["grid"].shape
    id_to_cell = _build_id_to_cell(grid_ctx)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=labels,
        horizontal_spacing=subplot_gap,
        vertical_spacing=subplot_gap,
    )

    colorscale = [
        [0.0, "#ffffff"],
        [0.249, "#ffffff"],
        [0.25, "#d62728"],
        [0.499, "#d62728"],
        [0.5, "#ffeb3b"],
        [0.749, "#ffeb3b"],
        [0.75, "#2ca02c"],
        [1.0, "#2ca02c"],
    ]

    for idx, group in enumerate(sensor_groups):
        train_sensors = group.get("train_sensors", [])
        val_sensors = group.get("val_sensors", [])
        test_sensors = group.get("test_sensors", [])

        grid = _group_to_grid(
            grid_shape=grid_shape,
            id_to_cell=id_to_cell,
            train_sensors=train_sensors,
            val_sensors=val_sensors,
            test_sensors=test_sensors,
        )

        row = idx // n_cols + 1
        col = idx % n_cols + 1
        fig.add_trace(
            go.Heatmap(
                z=grid,
                zmin=0,
                zmax=3,
                colorscale=colorscale,
                xgap=cell_gap,
                ygap=cell_gap,
                showscale=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#d62728"), name="train"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#ffeb3b"), name="validation"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#2ca02c"), name="test"),
        row=1,
        col=1,
    )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed")

    fig.update_layout(
        title=title,
        showlegend=True,
        plot_bgcolor=grid_line_color,
        height=subplot_size * n_rows,
        width=subplot_size * n_cols,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def plot_sensor_groups_map(
    *,
    grid_ctx: dict,
    sensor_groups: Sequence[dict],
    n_cols: Optional[int] = None,
    group_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    map_style: str = "carto-positron",
    zoom: Optional[float] = None,
    fixed_view: bool = True,
    subplot_gap: float = 0.02,
    subplot_size: int = 480,
    grid_line_color: str = "rgba(0,0,0,0.25)",
    grid_line_width: int = 1,
    html_config: Optional[dict] = None,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """
    Plot sensor groups on a map background, coloring each grid cell.

    Each item in sensor_groups must be a dict with keys:
      - "train_sensors"
      - "val_sensors"
      - "test_sensors"
    """
    if not sensor_groups:
        raise ValueError("sensor_groups is empty")

    n_groups = len(sensor_groups)
    if n_cols is None:
        n_cols = int(math.ceil(math.sqrt(n_groups)))
    n_rows = int(math.ceil(n_groups / n_cols))

    labels = _resolve_group_labels(sensor_groups, group_labels)
    id_to_cell = _build_id_to_cell(grid_ctx)
    cell_rings: dict[tuple[int, int], list[tuple[float, float]]] = {}

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=labels,
        horizontal_spacing=subplot_gap,
        vertical_spacing=subplot_gap,
        specs=[[{"type": "map"} for _ in range(n_cols)] for _ in range(n_rows)],
    )

    train_color = "rgba(214, 39, 40, 0.5)"
    val_color = "rgba(255, 235, 59, 0.55)"
    test_color = "rgba(44, 160, 44, 0.5)"

    for idx, group in enumerate(sensor_groups):
        train_sensors = group.get("train_sensors", [])
        val_sensors = group.get("val_sensors", [])
        test_sensors = group.get("test_sensors", [])

        row = idx // n_cols + 1
        col = idx % n_cols + 1

        train_lats, train_lons = _rings_from_sensors(
            grid_ctx=grid_ctx,
            id_to_cell=id_to_cell,
            sensors=train_sensors,
            cell_rings=cell_rings,
        )
        val_lats, val_lons = _rings_from_sensors(
            grid_ctx=grid_ctx,
            id_to_cell=id_to_cell,
            sensors=val_sensors,
            cell_rings=cell_rings,
        )
        test_lats, test_lons = _rings_from_sensors(
            grid_ctx=grid_ctx,
            id_to_cell=id_to_cell,
            sensors=test_sensors,
            cell_rings=cell_rings,
        )

        fig.add_trace(
            go.Scattermap(
                lat=train_lats,
                lon=train_lons,
                mode="lines",
                fill="toself",
                fillcolor=train_color,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=(idx == 0),
                name="train",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scattermap(
                lat=val_lats,
                lon=val_lons,
                mode="lines",
                fill="toself",
                fillcolor=val_color,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=(idx == 0),
                name="validation",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scattermap(
                lat=test_lats,
                lon=test_lons,
                mode="lines",
                fill="toself",
                fillcolor=test_color,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=(idx == 0),
                name="test",
            ),
            row=row,
            col=col,
        )

        fig.add_trace(_grid_lines_trace(grid_ctx, grid_line_color, grid_line_width), row=row, col=col)

    to_deg = _get_to_deg_transformer(grid_ctx.get("metric_crs", DEFAULT_METRIC_CRS))
    minx, miny, maxx, maxy = grid_ctx["bbox"].bounds
    center_lon, center_lat = to_deg.transform((minx + maxx) / 2, (miny + maxy) / 2)
    bounds = None
    if zoom is None:
        corner_lons, corner_lats = to_deg.transform(
            np.array([minx, minx, maxx, maxx]),
            np.array([miny, maxy, miny, maxy]),
        )
        bounds = dict(
            west=float(np.min(corner_lons)),
            east=float(np.max(corner_lons)),
            south=float(np.min(corner_lats)),
            north=float(np.max(corner_lats)),
        )

    for map_idx in range(1, n_rows * n_cols + 1):
        layout_key = "map" if map_idx == 1 else f"map{map_idx}"
        map_layout = dict(
            style=map_style,
            center=dict(lat=center_lat, lon=center_lon),
            bearing=0,
            pitch=0,
        )
        if bounds is not None:
            map_layout["bounds"] = bounds
        else:
            map_layout["zoom"] = zoom
        fig.update_layout(**{layout_key: map_layout})

    fig.update_layout(
        title=title,
        height=subplot_size * n_rows,
        width=subplot_size * n_cols,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    if fixed_view:
        fig.update_layout(dragmode=False, uirevision="fixed")

    if output_path:
        config = html_config
        if config is None and fixed_view:
            config = {"scrollZoom": False, "displayModeBar": False}
        fig.write_html(output_path, config=config)

    return fig


def plot_experiment_sensor_groups(
    experiment_folder: str | Path,
    *,
    include_train_sensors: bool = True,
    group_labels: Optional[Sequence[str]] = None,
    **plot_kwargs,
) -> go.Figure:
    loaded = load_experiment_sensor_groups(
        experiment_folder,
        include_train_sensors=include_train_sensors,
        group_labels=group_labels,
    )
    return plot_sensor_groups(
        grid_ctx=loaded.grid_ctx,
        sensor_groups=loaded.sensor_groups,
        group_labels=loaded.group_labels,
        **plot_kwargs,
    )


def plot_experiment_sensor_groups_map(
    experiment_folder: str | Path,
    *,
    include_train_sensors: bool = True,
    group_labels: Optional[Sequence[str]] = None,
    **plot_kwargs,
) -> go.Figure:
    loaded = load_experiment_sensor_groups(
        experiment_folder,
        include_train_sensors=include_train_sensors,
        group_labels=group_labels,
    )
    return plot_sensor_groups_map(
        grid_ctx=loaded.grid_ctx,
        sensor_groups=loaded.sensor_groups,
        group_labels=loaded.group_labels,
        **plot_kwargs,
    )


def write_figure_outputs(
    *,
    fig: go.Figure,
    outdir: Path,
    stem: str = "sensor_groups_map",
    html: bool = True,
    static_format: str | None = None,
    html_config: Optional[dict] = None,
    scale: int = 2,
) -> list[Path]:
    if not html and static_format is None:
        raise ValueError("At least one output must be enabled: HTML or a static format.")

    outdir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    if html:
        html_path = outdir / f"{stem}.html"
        fig.write_html(html_path, config=html_config)
        written_paths.append(html_path)

    if static_format:
        static_path = outdir / f"{stem}.{static_format}"
        width = int(fig.layout.width) if fig.layout.width is not None else 1200
        height = int(fig.layout.height) if fig.layout.height is not None else 800
        fig.write_image(static_path, width=width, height=height, scale=scale)
        written_paths.append(static_path)

    return written_paths


def main(
    *,
    experiment_folder: str | Path,
    outdir: Path | None = None,
    title: str | None = None,
    n_cols: int | None = None,
    include_train_sensors: bool = True,
    fixed_view: bool = True,
    map_style: str = "carto-positron",
    zoom: float | None = None,
    html: bool = True,
    static_format: str | None = None,
) -> list[Path]:
    experiment_path = Path(experiment_folder)
    if outdir is None:
        outdir = experiment_path / "plots"

    fig = plot_experiment_sensor_groups_map(
        experiment_path,
        include_train_sensors=include_train_sensors,
        n_cols=n_cols,
        title=title,
        fixed_view=fixed_view,
        map_style=map_style,
        zoom=zoom,
    )

    html_config = {"scrollZoom": False, "displayModeBar": False} if fixed_view else None
    return [
        path.resolve()
        for path in write_figure_outputs(
            fig=fig,
            outdir=outdir,
            stem="sensor_groups_map",
            html=html,
            static_format=static_format,
            html_config=html_config,
        )
    ]


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
        help="Directory for rendered map files. Defaults to EXPERIMENT_FOLDER/plots.",
    )
    @click.option(
        "--title",
        default=None,
        help="Optional figure title override.",
    )
    @click.option(
        "--n-cols",
        type=click.IntRange(min=1),
        default=None,
        help="Number of subplot columns. Defaults to a square-ish layout.",
    )
    @click.option(
        "--include-train/--test-only",
        default=True,
        show_default=True,
        help="Show non-test cells in red. Use --test-only to render only unseen cells.",
    )
    @click.option(
        "--fixed-view/--interactive",
        default=True,
        show_default=True,
        help="Disable map dragging/scroll zoom in the HTML output.",
    )
    @click.option(
        "--map-style",
        default="carto-positron",
        show_default=True,
        help="Plotly map style name.",
    )
    @click.option(
        "--zoom",
        type=float,
        default=None,
        help="Optional fixed zoom level. If omitted, the figure fits the grid bounds.",
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
        experiment_folder: Path,
        outdir: Path | None,
        title: str | None,
        n_cols: int | None,
        include_train: bool,
        fixed_view: bool,
        map_style: str,
        zoom: float | None,
        static_format: str | None,
        html: bool,
    ) -> None:
        try:
            written_paths = main(
                experiment_folder=experiment_folder,
                outdir=outdir,
                title=title,
                n_cols=n_cols,
                include_train_sensors=include_train,
                fixed_view=fixed_view,
                map_style=map_style,
                zoom=zoom,
                html=html,
                static_format=static_format,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        for path in written_paths:
            click.echo(f"Wrote: {path}")

    cli()
