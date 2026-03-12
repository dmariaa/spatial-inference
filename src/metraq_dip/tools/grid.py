from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyproj import Transformer
from shapely.geometry import Polygon, box, Point

# --- CRS transformers (same as your Voronoi script) ---
TO_M   = Transformer.from_crs("epsg:4326", "epsg:25830", always_xy=True)  # lon,lat -> x,y (meters)
TO_DEG = Transformer.from_crs("epsg:25830", "epsg:4326", always_xy=True)  # x,y -> lon,lat (degrees)


def _project_ll_to_m(df: pd.DataFrame):
    """Project lon/lat columns to meters arrays (x, y)."""
    xs, ys = TO_M.transform(
        df["longitude"].to_numpy(dtype=float),
        df["latitude"].to_numpy(dtype=float)
    )
    return np.asarray(xs), np.asarray(ys)

def _cells_with_points_ll(ctx: dict):
    """Return list of lat/lon rings for all grid cells that contain >=1 sensor."""
    xs, ys = ctx["xs"], ctx["ys"]
    bbox   = ctx["bbox"]
    cell   = ctx["cell_size_m"]

    minx, miny, maxx, maxy = bbox.bounds
    start_x = np.floor((minx) / cell) * cell
    start_y = np.floor((miny) / cell) * cell
    end_x   = np.ceil((maxx) / cell) * cell
    end_y   = np.ceil((maxy) / cell) * cell

    x_edges = np.arange(start_x, end_x + cell, cell)
    y_edges = np.arange(start_y, end_y + cell, cell)

    H, _, _ = np.histogram2d(ys, xs, bins=[y_edges, x_edges])  # rows=y, cols=x

    rings = []
    nz = np.argwhere(H > 0)
    for iy, ix in nz:
        x0, x1 = x_edges[ix],   x_edges[ix+1]
        y0, y1 = y_edges[iy],   y_edges[iy+1]
        # Build square ring in meters (counter-clockwise)
        ring_x = [x0, x1, x1, x0, x0]
        ring_y = [y0, y0, y1, y1, y0]
        lons, lats = TO_DEG.transform(np.array(ring_x), np.array(ring_y))
        rings.append(list(zip(lats.tolist(), lons.tolist())))
    return rings

def prepare_grid_context(df: pd.DataFrame, cell_size_m: int = 500, margin_m_x: int = 1000, margin_m_y: int = 1000):
    """
    Build grid context once from sensor points.
    - Projects points to meters (ETRS89 / UTM 30N)
    - Computes a padded bbox
    - Generates a square grid of cell_size_m covering the bbox

    :param df: da
    :param cell_size_m:
    :param margin_m:
    :return:

    Returns a dict with:
      - df: original dataframe
      - xs, ys: projected point arrays (meters)
      - bbox: shapely Polygon in meters
      - grid_cells_m: list[Polygon] of grid cells in meters
      - grid_cells_ll: list[list[(lat, lon)]] same cells in lat/lon rings
    """
    xs, ys = _project_ll_to_m(df)

    # bbox in meters (with margin)
    minx, miny = xs.min(), ys.min()
    maxx, maxy = xs.max(), ys.max()
    bbox = box(minx - margin_m_x, miny - margin_m_y, maxx + margin_m_x, maxy + margin_m_y)

    # Build grid aligned to cell_size_m
    def _floor_to(v, base):
        return np.floor(v / base) * base

    def _ceil_to(v, base):
        return np.ceil(v / base) * base

    start_x = _floor_to(minx - margin_m_x, cell_size_m)
    end_x   = _ceil_to(maxx + margin_m_x, cell_size_m)
    start_y = _floor_to(miny - margin_m_y, cell_size_m)
    end_y   = _ceil_to(maxy + margin_m_y, cell_size_m)

    xs_edges = np.arange(start_x, end_x + cell_size_m, cell_size_m)
    ys_edges = np.arange(start_y, end_y + cell_size_m, cell_size_m)

    grid_cells_m = []
    y_size = len(ys_edges) - 1
    x_size = len(xs_edges) - 1
    grid = np.empty((y_size, x_size), dtype=object)

    for x, x0 in enumerate(xs_edges[:-1]):
        x1 = x0 + cell_size_m
        for y, y0 in enumerate(ys_edges[:-1]):
            y1 = y0 + cell_size_m
            cell = box(x0, y0, x1, y1)
            # keep ONLY full cells that intersect bbox (no clipping), so border cells remain square
            if not cell.intersects(bbox):
                continue
            grid_cells_m.append(cell)
            grid[y_size - y - 1, x] = cell

    # Convert grid cells to lat/lon rings
    grid_cells_ll = []
    for poly in grid_cells_m:
        x_coords, y_coords = poly.exterior.coords.xy
        lons_out, lats_out = TO_DEG.transform(np.array(x_coords), np.array(y_coords))
        grid_cells_ll.append(list(zip(lats_out.tolist(), lons_out.tolist())))

    return {
        "df": df,
        "xs": xs,
        "ys": ys,
        "bbox": bbox,
        "grid": grid,
        "grid_cells_m": grid_cells_m,
        "grid_cells_ll": grid_cells_ll,
        "cell_size_m": cell_size_m,
        "margin": (margin_m_x, margin_m_y)
    }


def _add_grid_to_figure(
    fig: go.Figure,
    ctx: dict,
    fill_color: str = "rgba(0, 114, 178, 0.15)",  # light blue
    show_borders: bool = True,
    border_color: str = "#444",
    border_width: int = 1,
):
    """
    Add ALL grid cells as a single Scattermap trace for speed.
    - One uniform fill color for every cell (cannot vary per polygon in one trace).
    - Cells are concatenated with NaN separators.
    """
    cells_ll = ctx["grid_cells_ll"]
    if not cells_ll:
        return

    lats, lons = [], []
    for ring in cells_ll:
        lat_ring, lon_ring = zip(*ring)
        lats.extend(lat_ring)
        lons.extend(lon_ring)
        lats.append(np.nan)
        lons.append(np.nan)

    fig.add_trace(go.Scattermap(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(
            width=(border_width if show_borders else 0),
            color=(border_color if show_borders else "rgba(0,0,0,0)")
        ),
        hoverinfo="skip",
        showlegend=False,
        name="grid"
    ))

def _add_highlight_cells(
    fig: go.Figure,
    ctx: dict,
    val_sensors: list = [],
    fill_color: str = "rgba(255, 187, 0, 0.35)",
    highlight_color: str = "rgba(0, 255, 0, 0.35)",
    show_borders: bool = False,
    border_color: str = "#aa8800",
    border_width: int = 1,
):
    """Overlay ONE trace that fills all cells containing >=1 sensor."""
    rings = _cells_with_points_ll(ctx)
    if not rings:
        return

    df = ctx['df']
    lats, lons = [], []
    lats_h, lons_h = [], []

    for ring in rings:
        is_highlight = False
        utm_ring = [TO_M.transform(*b) for b in ring]
        quad = Polygon(utm_ring)

        for sensor_id in val_sensors:
            latitude, longitude = df[df['id']==sensor_id][['latitude', 'longitude']].values[0]
            utm_lat, utm_lon = TO_M.transform(latitude, longitude)
            point = Point(utm_lat, utm_lon)
            is_highlight = quad.contains(point)

            if is_highlight:
                break

        lat_ring, lon_ring = zip(*ring)
        if is_highlight:
            lats_h.extend(lat_ring)
            lons_h.extend(lon_ring)
            lats_h.append(np.nan)
            lons_h.append(np.nan)
        else:
            lats.extend(lat_ring)
            lons.extend(lon_ring)
            lats.append(np.nan)
            lons.append(np.nan)

    fig.add_trace(go.Scattermap(
        lat=lats,
        lon=lons,
        mode="lines",
        fill="toself",
        fillcolor=fill_color,
        line=dict(
            width=(border_width if show_borders else 0),
            color=(border_color if show_borders else "rgba(0,0,0,0)")
        ),
        hoverinfo="skip",
        showlegend=False,
        name="cells_with_sensors"
    ))

    if len(lats_h) > 0:
        fig.add_trace(go.Scattermap(
            lat=lats_h,
            lon=lons_h,
            mode="lines",
            fill="toself",
            fillcolor=highlight_color,
            line=dict(
                width=(border_width if show_borders else 0),
                color=(border_color if show_borders else "rgba(0,0,0,0)")
            ),
            hoverinfo="skip",
            showlegend=False,
            name="cells_with_sensors"
        ))

def _add_points(fig: go.Figure, df: pd.DataFrame):
    fig.add_trace(go.Scattermap(
        lat=df["latitude"],
        lon=df["longitude"],
        mode="markers",
        marker=dict(
            symbol="circle-stroked",
            size=5,
            opacity=1
        ),
        text=df.get("name", pd.Series([None]*len(df))),
        hoverinfo="text+lat+lon",
        customdata=(df[["id"]] if "id" in df.columns else None),
        hovertemplate=(
            "<b>%{text}</b><br>" +
            ("ID: %{customdata[0]}<br>" if "id" in df.columns else "") +
            "Lat: %{lat:.4f}, Lon: %{lon:.4f}<extra></extra>"
        )
    ))


def plot_grid_map(
    ctx: dict,
    val_sensors: list = [],
    map_style: str = "carto-positron",
    zoom: int = 11,
    title: Optional[str] = None,
    fill_color: str = "rgba(0, 114, 178, 0.15)",
    show_borders: bool = True,
    border_color: str = "#444",
    border_width: int = 1,
):
    """Render the grid as ONE Scattermap trace (fast), plus points."""
    df = ctx["df"]
    fig = go.Figure()

    _add_grid_to_figure(
        fig, ctx,
        fill_color=fill_color,
        show_borders=show_borders,
        border_color=border_color,
        border_width=border_width,
    )
    _add_highlight_cells(fig, ctx, val_sensors)
    _add_points(fig, df)

    fig.update_layout(
        title=title or f"Square Grid ({ctx['cell_size_m']} m) — Single Trace",
        showlegend=False,
        map=dict(
            style=map_style,
            zoom=zoom,
            # center on the grid bbox centroid (not data mean)
            center=dict(
                lat=TO_DEG.transform((ctx["bbox"].bounds[0] + ctx["bbox"].bounds[2]) / 2,
                                      (ctx["bbox"].bounds[1] + ctx["bbox"].bounds[3]) / 2)[1],
                lon=TO_DEG.transform((ctx["bbox"].bounds[0] + ctx["bbox"].bounds[2]) / 2,
                                      (ctx["bbox"].bounds[1] + ctx["bbox"].bounds[3]) / 2)[0]
            )
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.write_html(f"aq_sensors_grid.html")


def count_points_per_cell(ctx: dict) -> pd.DataFrame:
    xs, ys = ctx["xs"], ctx["ys"]
    bbox   = ctx["bbox"]
    cell   = ctx["cell_size_m"]

    minx, miny, maxx, maxy = bbox.bounds
    # Align bin edges to the same grid you built
    start_x = np.floor((minx) / cell) * cell
    start_y = np.floor((miny) / cell) * cell
    end_x   = np.ceil((maxx) / cell) * cell
    end_y   = np.ceil((maxy) / cell) * cell

    x_edges = np.arange(start_x, end_x + cell, cell)
    y_edges = np.arange(start_y, end_y + cell, cell)

    H, _, _ = np.histogram2d(ys, xs, bins=[y_edges, x_edges])  # rows=y, cols=x

    # Flatten to rows with cell indices & counts
    rows = []
    for iy in range(H.shape[0]):
        for ix in range(H.shape[1]):
            rows.append({
                "cell_row": iy,
                "cell_col": ix,
                "count": int(H[iy, ix]),
                "xmin": float(x_edges[ix]),
                "ymin": float(y_edges[iy]),
                "xmax": float(x_edges[ix+1]),
                "ymax": float(y_edges[iy+1]),
            })
    return pd.DataFrame(rows)


def find_grid_cell(ctx: dict, utm_x: float, utm_y: float, return_polygon: bool = False
                  ) -> Optional[Union[Tuple[int, int], Tuple[int, int, object]]]:
    """
    Return the (row, col) of the grid cell containing the point (utm_x, utm_y).

    - row corresponds to the y-bin.
    - col corresponds to the x-bin.
    - On success returns `(row, col)`. If `return_polygon` is True returns `(row, col, polygon)`.
    - Raises `IndexError` if the point lies outside the reconstructed grid
    """
    cell = ctx["cell_size_m"]
    grid = ctx.get("grid")

    # UTM coordinates, for the northern hemisphere, grow top-bottom, left-right
    # our returning grid is vertically inverted, as the array indexes grow in the
    # in the opposite direction as the contained utm y coordinates do
    start_x, start_y = grid[-1, 0].bounds[0], grid[-1, 0].bounds[1]
    n_rows, n_cols = grid.shape

    col = int((utm_x - start_x) // cell)
    row = n_rows - 1 - int((utm_y - start_y) // cell)

    if row < 0 or col < 0 or row >= n_rows or col >= n_cols:
        raise IndexError

    cell_obj = grid[row, col]
    if cell_obj is None:
        return None
    return (row, col, cell_obj) if return_polygon else (row, col)


if __name__ == "__main__":
    from metraq_dip.data.metraq_db import metraq_db


    query = R"""
            SELECT id,
                   name,
                   latitude,
                   longitude
            FROM merged_sensors
            WHERE id IN (SELECT DISTINCT sensor_id FROM MAD_merged_aq_data) 
            """
    df = pd.read_sql_query(query, con=metraq_db.connection)

    # Build grid with 1000 m cells and 1000 m margin around sensors
    ctx = prepare_grid_context(df, cell_size_m=1000, margin_m_x=0, margin_m_y=0)

    # test find_grid_cell
    # utms = _project_ll_to_m(df)
    # utm_x = utms[0][0]
    # utm_y = utms[1][0]
    # row, col, cell = find_grid_cell(ctx=ctx, utm_x=utm_x, utm_y=utm_y, return_polygon=True)

    validation_sensors = [28079039, 28079059, 28079018, 28079054]
    plot_grid_map(ctx, val_sensors=validation_sensors, zoom=11)

    # Optional: compute counts per cell and print a small preview
    counts = count_points_per_cell(ctx)
    print(counts.sort_values("count", ascending=False))
