"""PyVista/VTK scene for textured space-time cubes and monthly animation."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from .cartography import load_map
from .common import PALETTE, font, video_writer
from .configuration import RenderStyle, load_style
from .data import DEFAULT_POLLUTANTS, prepare

MONTH_NAMES = ('Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre')
FACE_NAMES = ('front', 'north', 'east', 'back', 'south', 'west')


def face_geometry(width, depth):
    """Quad vertices: east=x, north=y, older time=-z; images start at top left."""
    left, right = -width / 2, width / 2
    bottom, top = -0.5, 0.5
    front, back = depth / 2, -depth / 2
    return {
        'front': [(left, top, front), (right, top, front),
                  (right, bottom, front), (left, bottom, front)],
        'north': [(left, top, front), (right, top, front),
                  (right, top, back), (left, top, back)],
        'east': [(right, top, front), (right, top, back),
                 (right, bottom, back), (right, bottom, front)],
        'back': [(left, top, back), (right, top, back),
                 (right, bottom, back), (left, bottom, back)],
        'south': [(left, bottom, front), (right, bottom, front),
                  (right, bottom, back), (left, bottom, back)],
        'west': [(left, top, front), (left, top, back),
                 (left, bottom, back), (left, bottom, front)],
    }


def hourly_colors(predictions, limits):
    """Keep the original fixed per-pollutant palette and valid-zero handling."""
    low, high = limits
    values = np.clip((predictions - low) / max(high - low, 1e-6), 0, 1) * (len(PALETTE) - 1)
    indices = np.minimum(values.astype(int), len(PALETTE) - 2)
    fraction = (values - indices)[..., None]
    return np.round(PALETTE[indices] * (1 - fraction)
                    + PALETTE[indices + 1] * fraction).astype(np.uint8)


def moving_window(colors, position, hours):
    """Newest at index zero; smooth colors without modifying saved IDW fields."""
    indices = np.clip(position - np.arange(hours), 0, len(colors) - 1)
    first = indices.astype(int)
    second = np.minimum(first + 1, len(colors) - 1)
    fraction = (indices - first)[:, None, None, None]
    return np.round(colors[first] * (1 - fraction)
                    + colors[second] * fraction).astype(np.uint8)


class CubeTextures:
    """Geographic textures for all six faces, independent of camera position."""
    def __init__(self, map_image, ink, style):
        self.style = style
        self.size = (style.texture_size, style.texture_size)
        self.map = map_image.resize(self.size, Image.Resampling.LANCZOS)
        self.ink = ink.resize(self.size, Image.Resampling.LANCZOS)
        pixels = np.asarray(self.ink)
        self.edges = {
            'north': Image.fromarray(np.repeat(pixels[:1], self.size[1], axis=0)),
            'south': Image.fromarray(np.repeat(pixels[-1:], self.size[1], axis=0)),
            'east': Image.fromarray(np.repeat(pixels[:, -1:], self.size[0], axis=1)),
            'west': Image.fromarray(np.repeat(pixels[:, :1], self.size[0], axis=1)),
        }

    def images(self, window):
        """Time increases from front to back on each longitudinal face."""
        arrays = {
            'front': window[0], 'back': window[-1],
            'north': window[:, 0], 'south': window[:, -1],
            'east': window[:, :, -1].transpose(1, 0, 2),
            'west': window[:, :, 0].transpose(1, 0, 2),
        }
        result = {}
        for name, array in arrays.items():
            image = Image.fromarray(array).resize(self.size, Image.Resampling.BILINEAR).convert('RGBA')
            if name in ('front', 'back'):
                image = Image.blend(image, self.map, self.style.map_blend)
                image.alpha_composite(self.ink)
            elif self.style.extrude_map_lines:
                image.alpha_composite(self.edges[name])
            result[name] = image.convert('RGB')
        return result


@dataclass
class Cube:
    colors: np.ndarray
    actors: dict


def fit_geometry(renderer, vertices, size, padding=3):
    """Center the projected bounds and fit every corner inside the viewport."""
    x0, y0, x1, y1 = renderer.GetViewport()
    width, height = size
    center = np.array([(x0 + x1) * width / 2, (y0 + y1) * height / 2])
    available = np.array([(x1 - x0) * width / 2, (y1 - y0) * height / 2]) - padding
    if np.any(available <= 0):
        raise ValueError('Cube viewport is too small for the selected output size.')
    projected = []
    for vertex in vertices:
        renderer.SetWorldPoint(*vertex, 1)
        renderer.WorldToDisplay()
        projected.append(renderer.GetDisplayPoint()[:2])
    projected = np.asarray(projected)
    camera = renderer.GetActiveCamera()
    # Perspective makes the projected bounds asymmetric around the focal point.
    # Center those bounds before fitting, rather than wasting the opposite margin.
    bounds_center = (projected.min(axis=0) + projected.max(axis=0)) / 2
    half_viewport = available + padding
    shift = (bounds_center - center) / half_viewport
    old_center = np.asarray(camera.GetWindowCenter())
    camera.SetWindowCenter(*(old_center + shift))
    projected -= bounds_center - center
    scale = np.max(np.abs(projected - center) / available)
    angle = math.radians(camera.GetViewAngle())
    camera.SetViewAngle(math.degrees(2 * math.atan(math.tan(angle / 2) * scale)))
    camera.SetWindowCenter(*((old_center + shift) / scale))


class CubeScene:
    """VTK manages projection, visibility, camera fitting and texture mapping."""
    def __init__(self, output, meta, textures, style, times, month):
        try:
            import pyvista as pv
        except ImportError as exc:
            raise RuntimeError('Install presentation dependencies: uv sync --group presentation') from exc
        self.pv = pv
        self.style = style
        self.textures = textures
        self.cubes = []
        self.labels = []
        count = len(meta['pollutants'])
        columns = math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)
        self.shape = rows, columns
        self.plotter = pv.Plotter(off_screen=True,
                                  window_size=style.size, border=False)
        self.plotter.set_background(style.background, all_renderers=True)
        self.plotter.enable_anti_aliasing('fxaa', all_renderers=True)
        height, width = meta['grid_shape']
        geometry = face_geometry(width / height, style.cube_depth)
        self.hours = meta['window_hours']
        self.times = times
        # One full-window background; cube viewports render above it.
        from vtkmodules.vtkRenderingCore import vtkRenderer
        self.canvas = vtkRenderer()
        self.canvas.SetBackground(*pv.Color(style.background).float_rgb)
        self.canvas.SetLayer(0)
        self.plotter.render_window.SetNumberOfLayers(2)
        self.plotter.render_window.AddRenderer(self.canvas)
        for renderer in self.plotter.renderers:
            renderer.SetLayer(1)
        self.add_captions(count, month)
        renderer = self.plotter.renderer
        body_width, body_height = style.body_size
        margin_x = (style.width - body_width) / (2 * style.width)
        margin_y = 1 - (style.header_height + body_height) / style.height
        renderer.SetViewport(margin_x, margin_y, 1 - margin_x,
                             1 - style.header_height / style.height)
        all_vertices = []
        try:
            for index, detail in enumerate(meta['pollutants']):
                row, column = divmod(index, columns)
                offset = np.array([(column - (columns - 1) / 2) * (width / height + style.column_gap),
                                   ((rows - 1) / 2 - row) * (1 + style.row_gap), 0])
                cube_geometry = {name: np.asarray(vertices) + offset
                                 for name, vertices in geometry.items()}
                vertices = np.vstack(list(cube_geometry.values()))
                all_vertices.append(vertices)
                self.labels.append((detail['name'], vertices))

                with np.load(output / f"pollutant_{detail['id']}.npz") as saved:
                    colors = hourly_colors(saved['idw'], detail['limits'])
                images = textures.images(colors[:self.hours][::-1])
                actors = {}
                for name, vertices in cube_geometry.items():
                    mesh = pv.PolyData(np.asarray(vertices), faces=[4, 0, 1, 2, 3])
                    mesh.active_texture_coordinates = np.array([(0, 1), (1, 1), (1, 0), (0, 0)])
                    texture = pv.numpy_to_texture(np.asarray(images[name]))
                    texture.interpolate = True
                    actors[name] = self.plotter.add_mesh(mesh, texture=texture,
                        lighting=False, show_scalar_bar=False, reset_camera=False,
                        name=f"{detail['id']}_{name}")
                from vtkmodules.vtkRenderingCore import vtkTextActor3D
                label = vtkTextActor3D()
                label.SetInput(detail['name'])
                label.SetTextProperty(pv.TextProperty(
                    font_file=str(font(style.label_font_size).path),
                    font_size=style.label_font_size * 4, color=style.label_color,
                    background_opacity=0))
                bounds = label.GetBounds()
                label.SetScale(style.label_height / max(bounds[3] - bounds[2], 1))
                bounds = np.asarray(label.GetBounds()).reshape(3, 2)
                anchor = offset + np.array([0, -.5 - style.label_height * .7,
                                            style.cube_depth / 2 + .01])
                label.SetPosition(*(anchor - bounds.mean(axis=1)))
                self.plotter.add_actor(label, reset_camera=False)
                bounds = np.asarray(label.GetBounds()).reshape(3, 2)
                all_vertices.append(np.array([[x, y, z] for x in bounds[0]
                                               for y in bounds[1] for z in bounds[2]]))
                self.cubes.append(Cube(colors, actors))
            # All actors share this renderer, one camera and one depth buffer.
            self.plotter.camera_position = [(0, 0, 1), (0, 0, 0), (0, 1, 0)]
            self.plotter.camera.azimuth = style.camera_azimuth
            self.plotter.camera.elevation = style.camera_elevation
            self.plotter.camera.view_angle = style.camera_view_angle
            self.plotter.reset_camera()
            self.plotter.camera.zoom(style.camera_zoom)
            self.plotter.show(auto_close=False, interactive=False)
            fit_geometry(renderer, np.vstack(all_vertices), style.size,
                         padding=style.label_font_size * 1.5)
        except BaseException:
            self.close()
            raise

    def add_text(self, text, position, size, color, vertical='top'):
        actor = self.pv.Text(text=text, position=position)
        actor.prop = self.pv.TextProperty(font_file=str(font(size).path), font_size=size,
            color=color, background_opacity=0,
            justification_horizontal='center', justification_vertical=vertical)
        self.canvas.AddViewProp(actor)
        return actor

    def add_captions(self, count, month):
        style = self.style
        center = style.width / 2
        title = 'Madrid · Nueve contaminantes · IDW' if count == 9 else f'Madrid · {count} contaminante(s) · IDW'
        self.add_text(title, (center, style.height - style.header_height * .15),
                      style.title_font_size, style.text_color)
        self.date_actor = self.add_text('', (center, style.height - style.header_height * .65),
                                       style.subtitle_font_size, style.muted_color)
        footer_height = style.height - style.header_height - style.body_size[1]
        self.add_text(f'{MONTH_NAMES[int(month[5:])-1]} {month[:4]} · Ventana móvil de {self.hours} horas · El tiempo avanza hacia el fondo',
                      (center, footer_height * .9), style.footer_font_size, style.muted_color)
        self.add_text('© OpenStreetMap contributors · Rejilla 1 km',
                      (center, footer_height * .4), style.footer_font_size, style.muted_color)

    def image(self, position):
        current = self.times[int(position)]
        past = current - pd.Timedelta(hours=self.hours - 1)
        self.date_actor.input = f'Frente: {current:%d/%m %H:%M}  |  Fondo: {past:%d/%m %H:%M}'
        for cube in self.cubes:
            window = moving_window(cube.colors, position, self.hours)
            for name, image in self.textures.images(window).items():
                texture = self.pv.numpy_to_texture(np.asarray(image))
                texture.interpolate = True
                cube.actors[name].texture = texture
        self.plotter.render()
        return self.plotter.screenshot(return_img=True)

    def close(self):
        self.plotter.close()


def render(month, output, seconds=62, fps=24, preview_only=False,
           pollutant_ids=DEFAULT_POLLUTANTS, map_cache=None, config=None, data_cache=None):
    """Create synchronized IDW cubes in a real offscreen 3D scene."""
    style = load_style(config)
    output = Path(output)
    if seconds < 1 or fps < 1:
        raise ValueError('Duration and FPS must be positive.')
    if not preview_only and shutil.which('ffmpeg') is None:
        raise ValueError('FFmpeg is required for video export.')
    output.mkdir(parents=True, exist_ok=True)
    data_directory = Path(data_cache) if data_cache is not None else output
    if data_cache is not None and not (data_directory / 'data_manifest.json').is_file():
        raise ValueError('Data cache must contain data_manifest.json from a previous presentation run.')
    meta, times = prepare(data_directory, month, pollutant_ids)
    if map_cache is None and data_cache is not None:
        map_cache = data_directory
    map_image, ink = load_map(meta['bounds_utm'], meta['metric_crs'], output, map_cache)
    textures = CubeTextures(map_image, ink, style)
    scene = CubeScene(data_directory, meta, textures, style, times, month)
    hours = meta['window_hours']
    total_frames = seconds * fps
    prefix = style.artifact_prefix

    def frame(frameid):
        position = hours - 1 + (len(times) - hours) * frameid / max(1, total_frames - 1)
        return scene.image(position)

    try:
        for index in sorted({0, total_frames // 2, total_frames - 1}):
            Image.fromarray(frame(index)).save(output / f'{prefix}_preview_{index:04d}.png')
        poster = output / f'{prefix}_poster.png'
        Image.fromarray(frame(total_frames // 2)).save(poster)
        path = poster
        if not preview_only:
            path = output / f'{prefix}.mp4'
            with video_writer(path, style.size, fps) as stream:
                for index in range(total_frames):
                    stream.write(frame(index).tobytes())
                    if index % (fps * 15) == 0:
                        print(f'Rendered {index // fps}/{seconds} seconds', flush=True)
        metadata = dict(meta, renderer='PyVista/VTK', duration_seconds=seconds,
                        fps=fps, layout=list(scene.shape), style=style.as_dict(),
                        preview_only=preview_only, camera_mode='Single shared perspective camera',
                        time_direction='Newest at front; history moves backwards',
                        color_limits='Independent per pollutant, fixed over the month',
                        visual_smoothing='Linear color blend of hourly reconstructions')
        (output / f'{prefix}_metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        print(str(path.resolve()))
        return path
    finally:
        scene.close()
