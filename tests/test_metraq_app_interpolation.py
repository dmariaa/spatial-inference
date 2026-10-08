import numpy as np
import pandas as pd
import pytest

from metraq_app.interpolation import reconstruct


def sample_cells():
    return pd.DataFrame({'row': [0, 3, 7], 'col': [0, 4, 7], 'value': [0., 20., 50.]})


@pytest.mark.parametrize('method', ['idw', 'krg', 'dip-cnn'])
def test_reconstruction_fills_grid_and_preserves_observations(method):
    surface = reconstruct((8, 8), sample_cells(), method, dip_epochs=3)
    assert surface.shape == (8, 8)
    assert np.isfinite(surface).all()
    assert surface[0, 0] == 0
    assert surface[3, 4] == 20
    assert surface[7, 7] == 50


@pytest.mark.parametrize('method', ['observed', 'idw', 'krg', 'dip-cnn'])
def test_no_observations_does_not_invent_a_surface(method):
    empty = pd.DataFrame(columns=['row', 'col', 'value'])
    assert np.isnan(reconstruct((8, 8), empty, method)).all()


@pytest.mark.parametrize('method', ['idw', 'krg', 'dip-cnn'])
def test_single_valid_zero_gives_constant_zero_field(method):
    cells = pd.DataFrame({'row': [2], 'col': [4], 'value': [0.]})
    assert (reconstruct((8, 8), cells, method) == 0).all()


def test_map_distinguishes_estimates_and_observations():
    from xml.etree import ElementTree as ET
    from metraq_app.grid import aggregate_cells, make_grid
    from metraq_app.map import build_map
    sensors = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B'], 'utm_x': [440100., 441100.],
        'utm_y': [4470100., 4471100.], 'latitude': [40.38] * 2, 'longitude': [-3.70] * 2})
    ctx = make_grid(sensors)
    cells, stations = aggregate_cells(ctx, sensors, pd.DataFrame({'sensor_id': [1, 2], 'value': [0., 10.]}))
    surface = reconstruct(ctx['grid'].shape, cells, 'idw')
    markup = build_map(ctx, cells, stations, 'µg/m³', surface=surface, method='IDW')
    svg = ET.fromstring(markup[markup.index('<svg'):markup.index('</svg>') + 6])
    polygons = svg.findall('.//{http://www.w3.org/2000/svg}polygon[@data-cell]')
    observed = [p for p in polygons if p.attrib['data-kind'] == 'observed']
    estimated = [p for p in polygons if p.attrib['data-kind'] == 'estimated']
    assert len(observed) == 2
    assert len(estimated) == ctx['grid'].size - 2
    assert all(p.attrib['fill'] != 'none' for p in polygons)
    assert 'Estimación IDW' in estimated[0].find('{http://www.w3.org/2000/svg}title').text
