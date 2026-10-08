"""Build the spatial audit map with the same geographic grid as sensor_groups.html."""
from pathlib import Path
import html
import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale
from metraq_dip.utils.plot import (
    load_experiment_sensor_groups, _build_id_to_cell, _cell_ring_ll,
    _grid_lines_trace, _get_to_deg_transformer,
)

LABELS = {"NO2_2024": "NO₂ · 2024", "NOX_2023": "NOX · 2023", "NOX_2024": "NOX · 2024", "NO_2024": "NO · 2024"}


def build_map(output: Path) -> go.Figure:
    metrics = pd.read_csv(output / "metrics_by_sensor.csv")
    dip = metrics[metrics.method == "DIP"]
    loaded = load_experiment_sensor_groups("output/experiments/basic/metraq_nox_add24_ks_355")
    ctx = loaded.grid_ctx
    cells = _build_id_to_cell(ctx)
    catalog = ctx["df"].set_index("id")
    maximum = float(dip.wape.max())
    experiments = sorted(dip.experiment.unique())
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "map"}] * 2] * 2,
                        subplot_titles=[LABELS[e] for e in experiments], horizontal_spacing=.025, vertical_spacing=.065)
    for i, exp in enumerate(experiments):
        row, col = i // 2 + 1, i % 2 + 1
        frame = dip[dip.experiment == exp].sort_values("sensor_id")
        fig.add_trace(_grid_lines_trace(ctx, "rgba(70,80,95,0.20)", .6), row=row, col=col)
        center_lat, center_lon, details, text_colors = [], [], [], []
        for station in frame.itertuples():
            cell = cells[int(station.sensor_id)]
            if cell != (station.grid_row, station.grid_column):
                raise RuntimeError(f"Grid mapping differs for {station.sensor_id}")
            ring = _cell_ring_ll(ctx, cell)
            lat, lon = zip(*ring)
            color = sample_colorscale("Viridis", station.wape / maximum)[0]
            # Contrast against the translucent cell fill on the light basemap.
            rgb = [float(component) for component in color.removeprefix("rgb(").removesuffix(")").split(",")]
            blended = [(.72 * component + .28 * 245) / 255 for component in rgb]
            linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in blended]
            luminance = sum(weight * v for weight, v in zip((.2126, .7152, .0722), linear))
            white_contrast = 1.05 / (luminance + .05)
            black_contrast = (luminance + .05) / .05
            text_colors.append("#ffffff" if white_contrast > black_contrast else "#000000")
            fig.add_trace(go.Scattermap(lat=lat, lon=lon, mode="lines", fill="toself", fillcolor=color,
                                        opacity=.72, line=dict(color="rgba(30,40,50,.6)", width=.7),
                                        hoverinfo="skip", showlegend=False), row=row, col=col)
            center_lat.append(float(np.mean(lat[:-1])))
            center_lon.append(float(np.mean(lon[:-1])))
            name = html.escape(str(catalog.loc[int(station.sensor_id), "name"]))
            details.append([station.sensor_id, name, station.mae, station.wape * 100, station.target_mean,
                            station.bias, station.nearest_km, station.outside_hull_fraction * 100])
        # Scattermap uses one text color per trace, so split labels by contrast.
        for text_color in ("#ffffff", "#000000"):
            indices = [j for j, color in enumerate(text_colors) if color == text_color]
            if not indices:
                continue
            fig.add_trace(go.Scattermap(lat=[center_lat[j] for j in indices],
                                        lon=[center_lon[j] for j in indices],
                                        mode="markers+text",
                                        text=[str(details[j][0])[-2:] for j in indices],
                                        textposition="middle center",
                                        textfont=dict(size=12, family="Open Sans Bold", color=text_color),
                                        marker=dict(
                                            size=1,
                                            opacity=0,
                                            color=[float(frame.wape.iloc[j]) for j in indices],
                                            coloraxis="coloraxis"),
                                        customdata=[details[j] for j in indices],
                                        showlegend=False,
                                        hovertemplate=("<b>%{customdata[1]}</b> · %{customdata[0]}<br>"
                                                       "WAPE: %{customdata[3]:.1f}%<br>MAE: %{customdata[2]:.2f}<br>"
                                                       "Concentración media: %{customdata[4]:.2f}<br>Sesgo: %{customdata[5]:+.2f}<br>"
                                                       "Vecino TRAIN más cercano: %{customdata[6]:.2f} km<br>"
                                                       "Fuera de la envolvente TRAIN: %{customdata[7]:.1f}%<extra></extra>")),
                          row=row, col=col)
    # Use the complete grid, including outer cells, rather than the sensor bbox.
    rings = np.asarray(ctx["grid_cells_ll"], dtype=float)
    south, north = float(rings[:, :, 0].min()), float(rings[:, :, 0].max())
    west, east = float(rings[:, :, 1].min()), float(rings[:, :, 1].max())

    def mercator_y(latitude):
        radians = np.radians(latitude)
        return (1 - np.log(np.tan(np.pi / 4 + radians / 2)) / np.pi) / 2

    ymin, ymax = mercator_y(north), mercator_y(south)
    centerlon = (west + east) / 2
    centerlat = float(np.degrees(2 * np.arctan(np.exp(np.pi * (1 - 2 * (ymin + ymax) / 2))) - np.pi / 2))
    extent = dict(west=west, east=east, ymin=float(ymin), ymax=float(ymax))
    # Match panel aspect to the complete grid in the map's Mercator projection.
    world_x = (east - west) / 360
    world_y = ymax - ymin
    domain = fig.layout.map.domain
    domain_x = domain.x[1] - domain.x[0]
    domain_y = domain.y[1] - domain.y[0]
    initial_width = 1100
    initial_height = round((initial_width - 110) * domain_x * (world_y / world_x) / domain_y + 155)
    initial_zoom = float(np.log2(((initial_width - 110) * domain_x - 2) / world_x / 512))
    for i in range(1, 5):
        key = "map" if i == 1 else f"map{i}"
        fig.update_layout(**{key: dict(style="carto-positron", center=dict(lon=centerlon, lat=centerlat),
                                       zoom=initial_zoom, bearing=0, pitch=0)})
    fig.update_layout(title=dict(text="DIP: error relativo por estación y campaña", x=.02),
                      height=initial_height, autosize=True, dragmode=False, margin=dict(l=15, r=95, t=90, b=65),
                      coloraxis=dict(colorscale="Viridis", cmin=0, cmax=maximum, colorbar=dict(title="WAPE",
                                                                                               tickvals=[0, .25, .5,
                                                                                                         .75, 1, 1.25,
                                                                                                         1.5],
                                                                                               ticktext=["0%", "25%",
                                                                                                         "50%", "75%",
                                                                                                         "100%", "125%",
                                                                                                         "150%"],
                                                                                               len=.7)),
                      annotations=list(fig.layout.annotations) + [
                          dict(text="Cada celda representa 1 km × 1 km · Misma escala en los cuatro mapas",
                               x=.5, y=-.055, xref="paper", yref="paper", showarrow=False,
                               font=dict(size=12, color="#586174"))])
    # Resize the canvas to the grid aspect before fitting each map tightly.
    import json
    fit_script = """
    const gd = document.getElementById('{plot_id}');
    gd.style.margin = '0 auto';
    const extent = __EXTENT__;
    let timer;
    function fitGrid() {
        const margin = gd.layout.margin;
        const width = Math.max(1, gd.clientWidth - margin.l - margin.r);
        const firstDomain = gd.layout.map.domain;
        const worldX = (extent.east - extent.west) / 360;
        const worldY = extent.ymax - extent.ymin;
        const domainX = firstDomain.x[1] - firstDomain.x[0];
        const domainY = firstDomain.y[1] - firstDomain.y[0];
        const targetHeight = Math.round(width * domainX * (worldY / worldX) / domainY
                                        + margin.t + margin.b);
        const height = Math.max(1, targetHeight - margin.t - margin.b);
        const update = {};
        if (Math.abs(gd.clientHeight - targetHeight) > 1) update.height = targetHeight;
        for (let i = 1; i <= 4; i++) {
            const key = i === 1 ? 'map' : 'map' + i;
            const domain = gd.layout[key].domain;
            const panelWidth = width * (domain.x[1] - domain.x[0]);
            const panelHeight = height * (domain.y[1] - domain.y[0]);
            const worldX = (extent.east - extent.west) / 360;
            const worldY = extent.ymax - extent.ymin;
            const scale = Math.min(Math.max(20, panelWidth - 2) / worldX,
                                   Math.max(20, panelHeight - 2) / worldY);
            update[key + '.zoom'] = Math.log2(scale / 512);
            update[key + '.center'] = __CENTER__;
        }
        Plotly.relayout(gd, update);
    }
    const observer = new ResizeObserver(() => {
        clearTimeout(timer);
        timer = setTimeout(fitGrid, 100);
    });
    observer.observe(gd);
    fitGrid();
    """.replace("__EXTENT__", json.dumps(extent)).replace("__CENTER__", json.dumps(dict(lon=centerlon, lat=centerlat)))
    fig.write_html(output / "spatial_error_map.html",
                   include_plotlyjs=True,
                   default_width="70%",
                   config=dict(
                       scrollZoom=False,
                       responsive=True,
                       displaylogo=False,
                       displayModeBar=False),
                   post_script=fit_script)
    return fig


@click.command()
@click.option("--output", type=click.Path(path_type=Path), default="output/spatial_sensor_audit")
def main(output):
    fig = build_map(output)
    click.echo(f"{output / 'spatial_error_map.html'}: 4 maps, 88 evaluated station cells")


if __name__ == "__main__":
    main()
