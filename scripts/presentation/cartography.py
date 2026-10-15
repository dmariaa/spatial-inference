"""Georeferenced OSM context, clipped to the shared UTM grid."""
from pathlib import Path
import json
import shutil
import urllib.request
import urllib.parse
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from .common import font

def basemap(bounds, crs, output):
    cache = output / 'osm_features.json'
    (x0, y0, x1, y1) = bounds
    to_ll = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    (lon, lat) = to_ll.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    bbox = f'{min(lat)},{min(lon)},{max(lat)},{max(lon)}'
    if not cache.exists():
        query = f'[out:json][timeout:45];(way[highway~"^(motorway|trunk|primary|secondary)$"]({bbox});way[waterway=river]({bbox});way[leisure=park]({bbox});way[landuse=forest]({bbox});way[natural=water]({bbox}););out geom;'
        request = urllib.request.Request('https://overpass-api.de/api/interpreter', data=urllib.parse.urlencode({'data': query}).encode(), headers={'User-Agent': 'METRAQ-Research-VideoPrototype/0.3'})
        with urllib.request.urlopen(request, timeout=60) as response:
            cache.write_bytes(response.read())
    data = json.loads(cache.read_text(encoding='utf-8'))
    if not data.get('elements'):
        raise ValueError('No map features returned')
    size = 1000
    image = Image.new('RGBA', (size, size), '#edf0ec')
    draw = ImageDraw.Draw(image)
    ink = Image.new('RGBA', (size, size))
    idraw = ImageDraw.Draw(ink)
    project = Transformer.from_crs('EPSG:4326', crs, always_xy=True)

    def coords(geom):
        (xs, ys) = project.transform([v['lon'] for v in geom], [v['lat'] for v in geom])
        return [((x - x0) / (x1 - x0) * size, (y1 - y) / (y1 - y0) * size) for (x, y) in zip(xs, ys)]
    elements = data['elements']
    for item in elements:
        tags = item.get('tags', {})
        pts = coords(item.get('geometry', []))
        if len(pts) < 3 or 'highway' in tags or 'waterway' in tags:
            continue
        if item.get('nodes', [0])[0] != item.get('nodes', [1])[-1]:
            continue
        water = tags.get('natural') == 'water'
        draw.polygon(pts, fill='#b5d7e7' if water else '#c4d5bc')
        idraw.polygon(pts, fill=(95, 160, 200, 95) if water else (65, 110, 55, 55))
    for item in elements:
        tags = item.get('tags', {})
        pts = coords(item.get('geometry', []))
        if len(pts) < 2:
            continue
        if 'waterway' in tags:
            draw.line(pts, fill='#73a9c5', width=5)
            idraw.line(pts, fill=(40, 110, 160, 230), width=4)
        elif 'highway' in tags:
            width = 4 if tags['highway'] in ('motorway', 'trunk') else 2
            draw.line(pts, fill='#ffffff', width=width + 2)
            draw.line(pts, fill='#9f9e97', width=width)
            idraw.line(pts, fill=(245, 246, 243, 185), width=width + 2)
            idraw.line(pts, fill=(45, 55, 65, 155), width=max(1, width - 1))
    for (name, lon, lat) in [('Madrid', -3.7038, 40.4168), ('El Retiro', -3.6815, 40.4153), ('Casa de Campo', -3.756, 40.419), ('Manzanares', -3.722, 40.404)]:
        (xx, yy) = coords([{'lon': lon, 'lat': lat}])[0]
        draw.text((xx, yy), name, font=font(24), fill='#293d46', stroke_width=2, stroke_fill='#f6f7f4')
        idraw.text((xx, yy), name, font=font(22), fill=(20, 30, 40, 255), stroke_width=2, stroke_fill=(248, 249, 250, 210))
    image.save(output / 'map.png')
    ink.save(output / 'map_overlay.png')
    return (image, ink, len(elements))

def load_map(bounds, crs, output, map_cache=None):
    """Create a standalone map cache, or import compatible existing cartography."""
    directory = output / 'cartography'
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / 'bounds.json'
    expected = {'bounds_utm': list(map(float, bounds)), 'metric_crs': crs}
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding='utf-8')) != expected:
            raise ValueError('Map cache bounds differ from the reconstruction grid.')
    elif (directory / 'osm_features.json').exists():
        raise ValueError('Unverified map cache; use a new output directory.')
    if map_cache is not None and (not (directory / 'osm_features.json').exists()):
        map_cache = Path(map_cache)
        source_directory = map_cache / 'cartography' if (map_cache / 'cartography').is_dir() else map_cache
        source_metadata = source_directory / 'bounds.json'
        if source_metadata.exists():
            if json.loads(source_metadata.read_text(encoding='utf-8')) != expected:
                raise ValueError('Imported map bounds differ from the reconstruction grid.')
        else:
            candidate = map_cache / 'data.npz'
            if not candidate.exists():
                raise ValueError('Map cache must contain bounds.json or data.npz with its geographic bounds.')
            with np.load(candidate) as saved:
                if not np.array_equal(saved['bounds'], bounds):
                    raise ValueError('Imported map bounds differ from the reconstruction grid.')
        files = ('osm_features.json', 'map.png', 'map_overlay.png')
        if not (source_directory / files[0]).exists() and not all(
                (source_directory / name).exists() for name in files[1:]):
            raise ValueError('The imported map cache has no OSM features or rendered maps.')
        for name in files:
            source = source_directory / name
            if source.exists() and source.resolve() != (directory / name).resolve():
                shutil.copyfile(source, directory / name)
    metadata_path.write_text(json.dumps(expected, indent=2), encoding='utf-8')
    image_path = directory / 'map.png'
    overlay_path = directory / 'map_overlay.png'
    if image_path.exists() and overlay_path.exists():
        with Image.open(image_path) as image, Image.open(overlay_path) as overlay:
            return (image.convert('RGBA'), overlay.convert('RGBA'))
    (image, overlay, _) = basemap(bounds, crs, directory)
    return (image, overlay)
