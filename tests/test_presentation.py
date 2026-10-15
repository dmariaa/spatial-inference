"""Data integrity and cache boundaries of the presentation generator."""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.presentation import data
from scripts.presentation.cartography import load_map


@pytest.fixture
def cached_month(tmp_path):
    times = pd.date_range('2024-01-31 01:00', '2024-02-29 23:00', freq='h')
    meta = dict(month='2024-02', pollutant_ids=[8], pollutants=[dict(id=8, limits=[0., 10.])],
                grid_shape=[2, 2], bounds_utm=[431000., 4464000., 433000., 4466000.],
                metric_crs='EPSG:25830', cell_size_m=1000, window_hours=24)
    np.savez_compressed(tmp_path / 'pollutant_8.npz', idw=np.zeros((len(times), 2, 2)),
                        timestamps=times.to_numpy(), counts=np.full(len(times), 2),
                        bounds=meta['bounds_utm'], limits=[0., 10.])
    (tmp_path / 'data_manifest.json').write_text(json.dumps(meta), encoding='utf-8')
    return tmp_path, meta, times


def test_cached_month_does_not_read_source(monkeypatch, cached_month):
    output, expected, times = cached_month
    def unavailable(**kwargs):
        raise AssertionError('Cached rendering must not query source data')
    monkeypatch.setattr(data, 'get_aq_backend', unavailable)
    meta, loaded_times = data.prepare(output, '2024-02', (8,))
    assert meta == expected
    assert loaded_times.equals(times)
    assert loaded_times[23] == pd.Timestamp('2024-02-01')
    assert loaded_times[-1] == pd.Timestamp('2024-02-29 23:00')


@pytest.mark.parametrize('month,ids', [('2024-01', (8,)), ('2024-02', (8, 14))])
def test_incompatible_request_rejects_cache(cached_month, month, ids):
    output, _, _ = cached_month
    with pytest.raises(ValueError, match='Cache differs'):
        data.prepare(output, month, ids)


def test_cache_with_shifted_hours_is_rejected(cached_month):
    output, meta, times = cached_month
    np.savez_compressed(output / 'pollutant_8.npz', idw=np.zeros((len(times), 2, 2)),
                        timestamps=(times + pd.Timedelta(hours=1)).to_numpy(),
                        counts=np.full(len(times), 2), bounds=meta['bounds_utm'], limits=[0., 10.])
    with pytest.raises(ValueError, match='timestamps differ'):
        data.prepare(output, '2024-02', (8,))


def test_map_extent_is_checked_before_using_images(tmp_path):
    directory = tmp_path / 'cartography'
    directory.mkdir()
    (directory / 'bounds.json').write_text(json.dumps(dict(bounds_utm=[0., 0., 1., 1.], metric_crs='EPSG:25830')))
    Image.new('RGBA', (2, 2)).save(directory / 'map.png')
    Image.new('RGBA', (2, 2)).save(directory / 'map_overlay.png')
    with pytest.raises(ValueError, match='bounds differ'):
        load_map([0., 0., 2., 2.], 'EPSG:25830', tmp_path)


def test_importing_entrypoint_does_not_build_click_command():
    module = importlib.import_module('scripts.presentation.render_idw')
    assert not hasattr(module, 'main')


def test_fresh_idw_preserves_zero_and_ignores_interpolated_rows(tmp_path, monkeypatch):
    class Backend:
        def get_measurements(self, start_date, end_date, magnitudes):
            times = pd.date_range(start_date, end_date, freq='h')
            return pd.DataFrame(dict(sensor_id=np.tile([1, 2, 1], len(times)),
                entry_date=np.repeat(times.to_numpy(), 3), magnitude_id=8,
                value=np.tile([0., 10., 999.], len(times)),
                is_interpolated=np.tile([False, False, True], len(times))))

        def get_sensors(self, magnitudes):
            return pd.DataFrame(dict(id=[1, 2], name=['A', 'B'],
                utm_x=[440000., 442000.], utm_y=[4470000., 4471000.]))

    monkeypatch.setattr(data, 'get_aq_backend', lambda **kwargs: Backend())
    meta, times = data.prepare(tmp_path, '2024-02', (8,))
    assert len(times) == 29 * 24 + 23
    assert meta['pollutants'][0]['limits'] == [0., 10.]
    assert meta['pollutants'][0]['records'] == len(times) * 2
    with np.load(tmp_path / 'pollutant_8.npz') as saved:
        assert saved['idw'].min() == 0
        assert saved['idw'].max() == 10
        assert np.isfinite(saved['idw']).all()
        assert np.all(saved['counts'] == 2)


def test_import_modern_map_cache_without_network(tmp_path, monkeypatch):
    import scripts.presentation.cartography as cartography
    source = tmp_path / 'source' / 'cartography'
    source.mkdir(parents=True)
    bounds = [0., 0., 2., 2.]
    (source / 'bounds.json').write_text(json.dumps(dict(bounds_utm=bounds, metric_crs='EPSG:25830')))
    Image.new('RGBA', (2, 2), (10, 20, 30, 255)).save(source / 'map.png')
    Image.new('RGBA', (2, 2)).save(source / 'map_overlay.png')
    def unavailable(*args, **kwargs):
        raise AssertionError('Reusing a map must not access the network')
    monkeypatch.setattr(cartography, 'basemap', unavailable)
    images = load_map(bounds, 'EPSG:25830', tmp_path / 'new', source.parent)
    assert images[0].getpixel((0, 0)) == (10, 20, 30, 255)
    assert (tmp_path / 'new' / 'cartography' / 'map.png').read_bytes() == (source / 'map.png').read_bytes()
    with pytest.raises(ValueError, match='Imported map bounds differ'):
        load_map([0., 0., 3., 3.], 'EPSG:25830', tmp_path / 'other', source.parent)


def test_explicit_data_cache_requires_existing_manifest(tmp_path):
    from scripts.presentation.rendering import render
    with pytest.raises(ValueError, match='Data cache must contain data_manifest.json'):
        render('2024-01', tmp_path / 'new', preview_only=True, data_cache=tmp_path)
