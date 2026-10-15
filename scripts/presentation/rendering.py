"""Compact pollutant cube composition and synchronized monthly animation."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from .common import PALETTE, font, mapped, video_writer
from .data import DEFAULT_POLLUTANTS, prepare
from .cartography import load_map

MONTH_NAMES = ('Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre',
               'Noviembre', 'Diciembre')


def render(month, output, seconds=62, fps=24, preview_only=False, pollutant_ids=DEFAULT_POLLUTANTS, map_cache=None):
    """Render one to nine cubes, newest hour at the front and history behind."""
    output = Path(output)
    if seconds < 1 or fps < 1:
        raise ValueError('Duration and FPS must be positive.')
    if not preview_only and shutil.which('ffmpeg') is None:
        raise ValueError('FFmpeg is required for video export.')
    output.mkdir(parents=True, exist_ok=True)
    (meta, times) = prepare(output, month, pollutant_ids)
    (h, w) = meta['grid_shape']
    res = 384
    (mapimage, ink) = load_map(meta['bounds_utm'], meta['metric_crs'], output, map_cache)
    mapimage = mapimage.resize((res, res), Image.Resampling.LANCZOS)
    ink = ink.resize((res, res), Image.Resampling.LANCZOS)
    map_pixels = np.asarray(ink)
    # Extend the north/east map-border intersections along the time dimension.
    north = Image.fromarray(np.repeat(map_pixels[:1], res, axis=0))
    east = Image.fromarray(np.repeat(map_pixels[:, -1:], res, axis=1))
    cubes = []
    columns = int(np.ceil(np.sqrt(len(pollutant_ids))))
    rows = int(np.ceil(len(pollutant_ids) / columns))
    panel_width = 900 / columns
    panel_height = 870 / rows
    cube_scale = 600 * min(panel_width / 300, panel_height / 290)
    for (n, detail) in enumerate(meta['pollutants']):
        with np.load(output / f"pollutant_{detail['id']}.npz") as data:
            pred = data['idw']
        (low, high) = detail['limits']
        values = np.clip((pred - low) / max(high - low, 1e-06), 0, 1) * 4
        indices = np.minimum(values.astype(int), 3)
        fraction = (values - indices)[..., None]
        colors = np.round(PALETTE[indices] * (1 - fraction) + PALETTE[indices + 1] * fraction).astype(np.uint8)
        col = n % columns
        row = n // columns
        cx = 90 + panel_width * (col + 170 / 300)
        cy = 102 + panel_height * (row + 93 / 290)

        def p(x, y, z):
            angle = 0.42
            el = 0.26
            world = np.array([(x - 0.5) * w / h, y - 0.5, (z - 0.5) * 2.05])
            right = np.array([np.cos(angle), 0, np.sin(angle)])
            up = np.array([-np.sin(el) * np.sin(angle), np.cos(el), np.sin(el) * np.cos(angle)])
            toward = np.array([np.cos(el) * np.sin(angle), np.sin(el), -np.cos(el) * np.cos(angle)])
            depth = 4 - float(toward @ world)
            return np.array(
                [cx + cube_scale * float(right @ world) / depth, cy - cube_scale * float(up @ world) / depth])

        frontpts = [p(0, 1, 0), p(1, 1, 0), p(1, 0, 0), p(0, 0, 0)]
        toppts = [p(0, 1, 0), p(1, 1, 0), p(1, 1, 1), p(0, 1, 1)]
        rightpts = [p(1, 1, 0), p(1, 1, 1), p(1, 0, 1), p(1, 0, 0)]
        cubes.append((colors, frontpts, toppts, rightpts, detail))
    background = Image.new('RGBA', (1080, 1080), '#f8f9fb')
    d = ImageDraw.Draw(background)
    d.text((540, 18), 'Madrid · Nueve contaminantes · IDW' if len(
        pollutant_ids) == 9 else f'Madrid · {len(pollutant_ids)} contaminante(s) · IDW', anchor='mt', font=font(30),
           fill='#172b42')
    for (_, frontpts, toppts, rightpts, detail) in cubes:
        vertices = np.vstack((frontpts, toppts, rightpts))
        center_x = (vertices[:, 0].min() + vertices[:, 0].max()) / 2
        bottom_y = vertices[:, 1].max()
        d.text((center_x, bottom_y + 8), detail['name'], anchor='mt', font=font(25), fill='#172b42')
    d.text((540, 1020),
           f'{MONTH_NAMES[int(month[5:]) - 1]} {month[:4]} · Ventana móvil de 24 horas · El tiempo avanza hacia el fondo',
           anchor='mt', font=font(17), fill='#596a7e')
    d.text((540, 1050), '© OpenStreetMap contributors · Rejilla 1 km', anchor='mt', font=font(14), fill='#778595')

    def scene(frameid):
        position = 23 + (len(times) - 24) * frameid / max(1, seconds * fps - 1)
        indices = np.clip(position - np.arange(24), 0, len(times) - 1)
        i0 = indices.astype(int)
        i1 = np.minimum(i0 + 1, len(times) - 1)
        # Blend hourly colors only; do not alter the saved IDW reconstructions.
        f = (indices - i0)[:, None, None, None]
        im = background.copy()
        for (colors, frontpts, toppts, rightpts, detail) in cubes:
            window = np.round(colors[i0] * (1 - f) + colors[i1] * f).astype(np.uint8)
            front = Image.fromarray(window[0]).resize((res, res), Image.Resampling.BILINEAR).convert('RGBA')
            front = Image.blend(front, mapimage, 0.16)
            front.alpha_composite(ink)
            top = Image.fromarray(window[:, 0]).resize((res, res), Image.Resampling.BILINEAR).convert('RGBA')
            top.alpha_composite(north)
            right = Image.fromarray(window[:, :, -1].transpose(1, 0, 2)).resize((res, res),
                                                                                Image.Resampling.BILINEAR).convert(
                'RGBA')
            right.alpha_composite(east)
            mapped(im, top, toppts)
            mapped(im, right, rightpts)
            mapped(im, front, frontpts)
            ImageDraw.Draw(im).line([tuple(v) for v in frontpts + [frontpts[0]]], fill='#65758a', width=1)
        d = ImageDraw.Draw(im)
        current = times[int(position)]
        past = current - pd.Timedelta(hours=23)
        d.text((540, 62), f'Frente: {current:%d/%m %H:%M}  |  Fondo: {past:%d/%m %H:%M}', anchor='mt', font=font(19),
               fill='#596a7e')
        return im.convert('RGB')

    for n in (0, seconds * fps // 2, seconds * fps - 1):
        scene(n).save(output / f'preview_{n:04d}.png')
    scene(seconds * fps // 2).save(output / 'poster.png')
    if preview_only:
        print('Previews ready')
        return output / 'poster.png'
    path = output / 'idw_cubes.mp4'
    with video_writer(path, (1080, 1080), fps) as stream:
        for frameid in range(seconds * fps):
            stream.write(scene(frameid).tobytes())
            if frameid % (fps * 15) == 0:
                print(f'Rendered {frameid // fps}/{seconds} seconds')
    meta.update(duration_seconds=seconds, fps=fps, resolution=[1080, 1080],
                layout=f'compact {rows}x{columns}, centered labels below cubes, no displayed scales',
                time_direction='Newest at front; history moves backwards',
                color_limits='Independent per pollutant, fixed over the full month',
                visual_smoothing='Linear color blend between adjacent hourly reconstructions')
    (output / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(path.resolve()))
    return path
