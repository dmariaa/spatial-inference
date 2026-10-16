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


def test_dip_validation_cells_never_enter_training_inputs():
    from metraq_app.interpolation import dip_split
    data = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    mask = np.zeros_like(data, dtype=bool)
    mask[:, 1:5, 1:5] = True
    mask[1] = False
    split, count = dip_split(data, mask, np.random.default_rng(1))
    train, val = split['train_mask'][0], split['val_mask'][0]
    assert count == 3
    assert not (train & val).any()
    assert not (train.any(axis=0) & val.any(axis=0)).any()
    assert np.array_equal(train | val, mask)
    assert not split['input_data'][-2:][:, val].any()
    assert not train[1].any() and not val[1].any()
    mean, std = split['normalization_stats'][0]
    assert mean == pytest.approx(data[train].mean())
    assert std == pytest.approx(data[train].std())


@pytest.mark.parametrize('hour, expected_hours', [(2, 3), (30, 24)])
def test_app_dip_window_is_clipped_and_keeps_missing_hours(monkeypatch, hour, expected_hours):
    from datetime import datetime, timedelta
    from metraq_app import app as backend
    sensors = pd.DataFrame({'id': [1], 'name': ['A'], 'utm_x': [440100.],
        'utm_y': [4470100.], 'latitude': [40.38], 'longitude': [-3.70]})
    first = datetime(2024, 1, 1)
    end = first + timedelta(hours=hour)
    monkeypatch.setattr(backend, 'load_catalog', lambda source: pd.DataFrame({
        'magnitude_id': [12], 'first_date': [pd.Timestamp(first)], 'last_date': [pd.Timestamp(end)],
        'low': [0.], 'high': [100.]}))
    monkeypatch.setattr(backend, 'load_sensors', lambda *args: sensors)
    monkeypatch.setattr(backend, 'load_measurements', lambda *args: pd.DataFrame({'sensor_id': [1], 'value': [2.]}))
    def history(source, magnitude, start, stop):
        assert stop == end and start == max(first, end - timedelta(hours=23))
        dates = pd.date_range(start, stop, freq='h').delete(1)
        return pd.DataFrame({'entry_date': dates, 'sensor_id': 1, 'value': 2.})
    monkeypatch.setattr(backend, 'load_history', history)
    def reconstruct(shape, cells, method, epochs, *, history, details):
        data, mask = history
        assert len(data) == expected_hours and not mask[1].any() and mask[-1].any()
        return np.full(shape, 2., dtype=np.float32)
    monkeypatch.setattr(backend, 'reconstruct', reconstruct)
    backend.clear_cache()
    try:
        surface, details = backend.surface_for('files', 12, end, 1000, 'dip-cnn', 3)
        assert details['window_hours'] == expected_hours
        assert details['observed_hours'] == expected_hours - 1
    finally:
        backend.clear_cache()
