"""Named, validated visual settings, independent of the rendering engine."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class RenderStyle:
    width: int = 1080
    height: int = 1080
    texture_size: int = 384
    cube_depth: float = 2.7
    column_gap: float = 0.6
    row_gap: float = 0.65
    camera_azimuth: float = 24
    camera_elevation: float = 15
    camera_view_angle: float = 30
    camera_zoom: float = 1.35
    background: str = '#10141b'
    text_color: str = '#f5f5f5'
    muted_color: str = '#bdc8d5'
    map_blend: float = 0.16
    extrude_map_lines: bool = True
    body_width_fraction: float = 0.96
    header_fraction: float = 0.09
    footer_fraction: float = 0.06
    label_color: str = '#f5f5f5'
    label_height: float = 0.12
    label_font_size: int = 25
    title_font_size: int = 30
    subtitle_font_size: int = 19
    footer_font_size: int = 14
    artifact_prefix: str = 'idw_cubes_scene_labels_full'

    def __post_init__(self):
        for name in ('width', 'height', 'texture_size', 'label_font_size',
                     'title_font_size', 'subtitle_font_size', 'footer_font_size'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if self.width % 2 or self.height % 2:
            raise ValueError('Video width and height must be even.')
        for name in ('cube_depth', 'camera_zoom', 'camera_view_angle', 'label_height'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        for name in ('column_gap', 'row_gap'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')
        if not 0 < self.camera_view_angle < 180:
            raise ValueError('Camera view angle must be between 0 and 180 degrees.')
        if not all(math.isfinite(v) for v in (self.camera_azimuth, self.camera_elevation)):
            raise ValueError('Camera angles must be finite.')
        if not 0 <= self.map_blend <= 1:
            raise ValueError('Map blend must be between 0 and 1.')
        if not (0 < self.header_fraction < 0.4 and 0 < self.footer_fraction < 0.4):
            raise ValueError('Header and footer fractions must be between 0 and 0.4.')
        if not 0 < self.body_width_fraction <= 1:
            raise ValueError('Body width fraction must be between 0 and 1.')
        if not self.artifact_prefix or Path(self.artifact_prefix).name != self.artifact_prefix:
            raise ValueError('Artifact prefix must be a filename, without a directory.')

    @property
    def size(self):
        return self.width, self.height

    @property
    def header_height(self):
        return round(self.height * self.header_fraction)

    @property
    def body_size(self):
        footer_height = round(self.height * self.footer_fraction)
        return round(self.width * self.body_width_fraction), self.height - self.header_height - footer_height

    def as_dict(self):
        return asdict(self)


def load_style(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name('style.yaml')
    values = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(values, dict):
        raise ValueError('Visual configuration must be a YAML mapping.')
    try:
        return RenderStyle(**values)
    except TypeError as exc:
        raise ValueError(f'Invalid visual configuration: {exc}') from exc
