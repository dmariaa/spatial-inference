"""Responsive native SVG grid over georeferenced OpenStreetMap tiles."""
from html import escape
import math

from pyproj import Transformer
from shapely.ops import unary_union

# Viridis samples interpolated without loading a plotting engine.
PALETTE = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]


def color_for(value, low, high):
    fraction = min(1.0, max(0.0, (value - low) / (high - low)))
    position = fraction * (len(PALETTE) - 1)
    index = min(int(position), len(PALETTE) - 2)
    weight = position - index
    rgb = [round(a + weight * (b - a)) for a, b in zip(PALETTE[index], PALETTE[index + 1])]
    return '#%02x%02x%02x' % tuple(rgb)


def build_map(ctx, cells, observations, unit, bounds=None, show_sensors=True):
    """Return an SVG with exact projected geometry; no camera or panning needed."""
    project = Transformer.from_crs(ctx['metric_crs'], 'EPSG:3857', always_xy=True)

    def points(polygon):
        x, y = polygon.exterior.coords.xy
        projected_x, projected_y = project.transform(x, y)
        return list(zip(projected_x, [-value for value in projected_y]))

    def coordinates(vertices):
        return ' '.join(f'{x:.3f},{y:.3f}' for x, y in vertices)

    perimeter = points(unary_union(ctx['grid_cells_m']))
    left, right = min(x for x, _ in perimeter), max(x for x, _ in perimeter)
    top, bottom = min(y for _, y in perimeter), max(y for _, y in perimeter)
    side = max(right - left, bottom - top) * 1.02
    origin_x, origin_y = (left + right - side) / 2, (top + bottom - side) / 2
    if bounds is None:
        low, high = (float(cells.value.min()), float(cells.value.max())) if not cells.empty else (0.0, 1.0)
    else:
        low, high = map(float, bounds)
    if high <= low:
        high = low + 1.0
    values = {(int(cell.row), int(cell.col)): cell for cell in cells.itertuples()}
    pieces = [f'''<div class="metraq-native-map">
        <svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Madrid: grid completo de observaciones"
             viewBox="{origin_x:.3f} {origin_y:.3f} {side:.3f} {side:.3f}" preserveAspectRatio="xMidYMid meet">
        <defs><clipPath id="metraq-grid-clip"><polygon points="{coordinates(perimeter)}"/></clipPath></defs>
        <g clip-path="url(#metraq-grid-clip)">
        <polygon points="{coordinates(perimeter)}" fill="#f1f3f3"/>''']
    # Web Mercator tiles at z=12: the SVG positions each tile in projected meters.
    world = 20037508.342789244
    zoom = 12
    tile_size = 2 * world / (2 ** zoom)
    start_x, end_x = math.floor((left + world) / tile_size), math.floor((right + world) / tile_size)
    start_y, end_y = math.floor((top + world) / tile_size), math.floor((bottom + world) / tile_size)
    for tile_x in range(start_x, end_x + 1):
        for tile_y in range(start_y, end_y + 1):
            pieces.append(f'<image href="https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png" '
                          f'x="{tile_x * tile_size - world:.3f}" y="{tile_y * tile_size - world:.3f}" '
                          f'width="{tile_size:.3f}" height="{tile_size:.3f}" preserveAspectRatio="none" opacity="0.75"/>')
    pieces.append('</g>')
    for row in range(ctx['grid'].shape[0]):
        for col in range(ctx['grid'].shape[1]):
            cell = values.get((row, col))
            fill = 'none' if cell is None else color_for(float(cell.value), low, high)
            title = 'Sin medidas' if cell is None else (
                f'Media: {cell.value:.2f} {unit}\nSensores: {cell.count}\n' + cell.stations.replace('<br>', '\n'))
            pieces.append(f'<polygon class="metraq-cell" data-cell="{row}:{col}" '
                          f'points="{coordinates(points(ctx["grid"][row, col]))}" fill="{fill}" '
                          f'fill-opacity="0.72" stroke="#59636b" stroke-opacity="0.65" '
                          f'stroke-width="0.65" vector-effect="non-scaling-stroke" tabindex="0">'
                          f'<title>{escape(title)}</title></polygon>')
    if show_sensors:
        for station in observations.itertuples():
            x, y = project.transform(station.utm_x, station.utm_y)
            pieces.append(f'<circle cx="{x:.3f}" cy="{-y:.3f}" r="{side * 0.004:.3f}" fill="#202020">'
                          f'<title>{escape(station.detail)} {escape(unit)}</title></circle>')
    stops = ','.join(f'rgb({r},{g},{b})' for r, g, b in PALETTE)
    pieces.append(f'''</svg>
        <div class="metraq-scale"><span>{low:g}</span><div style="background:linear-gradient(to right,{stops})"></div>
        <span>{high:g} {escape(unit)}</span></div>
        <div class="metraq-attribution">© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a></div>
        </div>''')
    return ''.join(pieces)
