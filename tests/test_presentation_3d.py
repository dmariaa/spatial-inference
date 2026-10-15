"""Temporal direction and geographic orientation of the 3D presentation."""
from dataclasses import replace
import os
import numpy as np
import pytest
from PIL import Image
from scripts.presentation.configuration import RenderStyle, load_style
from scripts.presentation.rendering import CubeTextures, face_geometry, moving_window


def test_window_advances_history_away_from_front():
    colors = np.repeat(np.arange(30, dtype=np.uint8)[:, None, None, None], 3, axis=3)
    now = moving_window(colors, 24, 24)
    later = moving_window(colors, 25, 24)
    assert now[0, 0, 0, 0] == 24
    assert now[-1, 0, 0, 0] == 1
    np.testing.assert_array_equal(later[1:], now[:-1])
    midway = moving_window(colors, 24.5, 24)
    assert midway.dtype == np.uint8
    assert now[0, 0, 0, 0] <= midway[0, 0, 0, 0] <= later[0, 0, 0, 0]


def test_all_side_textures_join_the_correct_geographic_edges():
    style = replace(RenderStyle(), texture_size=2, map_blend=0, extrude_map_lines=False)
    blank = Image.new('RGBA', (2, 2), (0, 0, 0, 0))
    textures = CubeTextures(blank, blank, style)
    window = np.array([
        [[[255,0,0],[0,255,0]],[[0,0,255],[255,255,0]]],
        [[[50,0,0],[0,50,0]],[[0,0,50],[50,50,0]]],
    ], dtype=np.uint8)
    images = {name: np.asarray(image) for name, image in textures.images(window).items()}
    np.testing.assert_array_equal(images['front'], window[0])
    np.testing.assert_array_equal(images['back'], window[-1])
    np.testing.assert_array_equal(images['north'][0], images['front'][0])
    np.testing.assert_array_equal(images['east'][:, 0], images['front'][:, -1])
    np.testing.assert_array_equal(images['south'][0], images['front'][-1])
    np.testing.assert_array_equal(images['west'][:, 0], images['front'][:, 0])


def test_configuration_rejects_unknown_keys_and_invalid_video_size(tmp_path):
    config = tmp_path / 'style.yaml'
    config.write_text('camera_typo: 24', encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid visual configuration'):
        load_style(config)
    with pytest.raises(ValueError, match='even'):
        RenderStyle(width=1079)
    with pytest.raises(ValueError, match='between 0 and 180'):
        RenderStyle(camera_view_angle=180)


@pytest.mark.skipif(os.environ.get('METRAQ_TEST_3D') != '1', reason='Requires an OpenGL rendering context')
def test_vtk_texture_preserves_north_south_and_east_west():
    pv = pytest.importorskip('pyvista')
    mesh = pv.PolyData(np.array(face_geometry(2, 2)['front']), faces=[4,0,1,2,3])
    mesh.active_texture_coordinates = np.array([(0,1),(1,1),(1,0),(0,0)])
    pixels = np.zeros((16,16,3), dtype=np.uint8)
    expected = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
    pixels[:8,:8], pixels[:8,8:], pixels[8:,:8], pixels[8:,8:] = expected
    plotter = pv.Plotter(off_screen=True, window_size=(256,256))
    try:
        plotter.add_mesh(mesh, texture=pv.numpy_to_texture(pixels), lighting=False)
        plotter.camera_position = [(0,0,5),(0,0,0),(0,1,0)]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 1.25
        plotter.reset_camera_clipping_range()
        plotter.show(auto_close=False, interactive=False)
        image = plotter.screenshot(return_img=True)
        samples = [image[100,64], image[100,192], image[156,64], image[156,192]]
        np.testing.assert_allclose(samples, expected, atol=3)
    finally:
        plotter.close()


@pytest.mark.skipif(os.environ.get('METRAQ_TEST_3D') != '1', reason='Requires an OpenGL rendering context')
def test_camera_fit_keeps_every_cube_corner_inside_viewport():
    from scripts.presentation.rendering import fit_geometry
    pv = pytest.importorskip('pyvista')
    geometry = face_geometry(23 / 24, 2.05)
    vertices = np.vstack(list(geometry.values()))
    plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
    try:
        for face in geometry.values():
            plotter.add_mesh(pv.PolyData(np.asarray(face), faces=[4, 0, 1, 2, 3]), reset_camera=False)
        plotter.renderer.SetViewport(.1, .2, .9, .9)
        plotter.camera_position = [(0, 0, 1), (0, 0, 0), (0, 1, 0)]
        plotter.camera.azimuth = 24
        plotter.camera.elevation = 15
        plotter.reset_camera()
        plotter.camera.zoom(2)
        plotter.show(auto_close=False, interactive=False)
        fit_geometry(plotter.renderer, vertices, (400, 400))
        for vertex in vertices:
            plotter.renderer.SetWorldPoint(*vertex, 1)
            plotter.renderer.WorldToDisplay()
            x, y, _ = plotter.renderer.GetDisplayPoint()
            assert 43 - 1e-5 <= x <= 357 + 1e-5
            assert 83 - 1e-5 <= y <= 357 + 1e-5
    finally:
        plotter.close()


@pytest.mark.skipif(os.environ.get('METRAQ_TEST_3D') != '1', reason='Requires an OpenGL rendering context')
@pytest.mark.parametrize('count', [1, 4, 9])
def test_cubes_share_one_renderer_and_camera(tmp_path, count):
    import pandas as pd
    from scripts.presentation.rendering import CubeScene
    style = replace(RenderStyle(), width=480, height=480, texture_size=16,
                    label_font_size=12, title_font_size=14,
                    subtitle_font_size=10, footer_font_size=6)
    blank = Image.new('RGBA', (16, 16), (0, 0, 0, 0))
    textures = CubeTextures(blank, blank, style)
    pollutants = []
    for index in range(count):
        np.savez(tmp_path / f'pollutant_{index}.npz', idw=np.ones((3, 2, 2)))
        pollutants.append(dict(id=index, name=f'Gas {index}', limits=[0, 2]))
    meta = dict(pollutants=pollutants, grid_shape=[2, 2], window_hours=2)
    times = pd.date_range('2024-01-01', periods=3, freq='h')
    scene = CubeScene(tmp_path, meta, textures, style, times, '2024-01')
    try:
        assert len(scene.plotter.renderers) == 1
        renderer = scene.plotter.renderer
        assert renderer.GetActors().GetNumberOfItems() == count * 6
        labels = [prop for prop in renderer.GetViewProps() if prop.IsA('vtkTextActor3D')]
        assert len(labels) == count
        assert all(label.GetTextProperty().GetBackgroundOpacity() == 0 for label in labels)
        assert len(scene.cubes) == count
        assert scene.image(1.5).shape == (480, 480, 3)
        x0, y0, x1, y1 = renderer.GetViewport()
        for _, vertices in scene.labels:
            for vertex in vertices:
                renderer.SetWorldPoint(*vertex, 1)
                renderer.WorldToDisplay()
                x, y, _ = renderer.GetDisplayPoint()
                assert x0 * 480 <= x <= x1 * 480
                assert y0 * 480 <= y <= y1 * 480
    finally:
        scene.close()
