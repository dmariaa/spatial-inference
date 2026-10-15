"""Shared palette, fonts, perspective texture mapping and video encoding."""
from contextlib import contextmanager, suppress
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageFont
PALETTE = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]])

def font(size):
    """Prefer the original font, with a portable DejaVu fallback."""
    for name in ('C:/Windows/Fonts/segoeui.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise ValueError('No suitable font found. Install Segoe UI or DejaVu Sans.')

@contextmanager
def video_writer(path, size, fps):
    """Always close the encoder, including when rendering fails."""
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise ValueError('FFmpeg is required for video export; use --preview-only for images.')
    (width, height) = size
    command = [executable, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        yield encoder.stdin
    except BaseException:
        encoder.terminate()
        with suppress(BrokenPipeError):
            encoder.stdin.close()
        encoder.wait()
        raise
    else:
        try:
            encoder.stdin.close()
        finally:
            status = encoder.wait()
        if status:
            raise RuntimeError('Video encoding failed.')

def mapped(canvas, tex, pts):
    """Project a texture onto a face using a perspective homography."""
    pts = np.asarray(pts)
    mn = np.floor(pts.min(axis=0)).astype(int)
    mx = np.ceil(pts.max(axis=0)).astype(int)
    local = pts - mn
    (tw, th) = tex.size
    rows = []
    target = []
    for ((x, y), (u, v)) in zip(local, [(0, 0), (tw, 0), (tw, th), (0, th)]):
        rows.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]])
        target.extend([u, v])
    coeff = np.linalg.solve(np.asarray(rows), np.asarray(target))
    warped = tex.transform(tuple(mx - mn), Image.Transform.PERSPECTIVE, tuple(coeff), Image.Resampling.BILINEAR)
    mask = Image.new('L', tuple(mx - mn))
    ImageDraw.Draw(mask).polygon([tuple(v) for v in local], fill=255)
    warped.putalpha(ImageChops.multiply(warped.getchannel('A'), mask))
    canvas.alpha_composite(warped, tuple(mn))
