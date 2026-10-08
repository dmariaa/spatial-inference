from pathlib import Path
import numpy as np
import pandas as pd
from metraq_app.grid import make_grid, aggregate_cells
from metraq_app.map import build_map


def test_cells_preserve_zero_average_sensors_and_exclude_missing():
    sensors = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"],
        "utm_x": [440100., 440200., 441100.], "utm_y": [4470100.] * 3,
        "latitude": [40.38] * 3, "longitude": [-3.70] * 3})
    ctx = make_grid(sensors)
    measurements = pd.DataFrame({"sensor_id": [1, 1, 2, 3], "value": [0., 0., 10., np.nan]})
    cells, observations = aggregate_cells(ctx, sensors, measurements)
    assert len(cells) == 1
    assert cells.iloc[0].value == 5
    assert cells.iloc[0]["count"] == 2
    assert observations.value.tolist() == [0., 10.]
    from xml.etree import ElementTree as ET
    markup = build_map(ctx, cells, observations, "µg/m³", (0, 100))
    svg = ET.fromstring(markup[markup.index('<svg'):markup.index('</svg>') + 6])
    _, _, width, height = map(float, svg.attrib['viewBox'].split())
    assert width == height
    polygons = svg.findall('.//{http://www.w3.org/2000/svg}polygon[@data-cell]')
    assert len(polygons) == ctx['grid'].size
    colored = [polygon for polygon in polygons if polygon.attrib['fill'] != 'none']
    assert len(colored) == 1
    assert 'Media: 5.00' in colored[0].find('{http://www.w3.org/2000/svg}title').text
    # Every vertex is inside the square viewport, including all border cells.
    x0, y0, _, _ = map(float, svg.attrib['viewBox'].split())
    for polygon in polygons:
        for vertex in polygon.attrib['points'].split():
            x, y = map(float, vertex.split(','))
            assert x0 <= x <= x0 + width
            assert y0 <= y <= y0 + height


def test_empty_hour_keeps_grid_without_colored_cells():
    sensors = pd.DataFrame({"id": [1], "name": ["A"], "utm_x": [440100.],
        "utm_y": [4470100.], "latitude": [40.38], "longitude": [-3.70]})
    ctx = make_grid(sensors)
    cells, observations = aggregate_cells(ctx, sensors, pd.DataFrame({"sensor_id": [], "value": []}))
    assert cells.empty
    markup = build_map(ctx, cells, observations, "µg/m³")
    assert markup.count('data-cell=') == ctx['grid'].size
    assert 'fill-opacity="0.72"' in markup
    assert 'Sin medidas' in markup



def test_api_changes_hour_pollutant_and_serves_separate_assets(monkeypatch):
    import pytest
    pytest.importorskip('fastapi')
    pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    from metraq_app import app as backend
    sensors = pd.DataFrame({'id': [1], 'name': ['A'], 'utm_x': [440100.],
        'utm_y': [4470100.], 'latitude': [40.38], 'longitude': [-3.70]})
    monkeypatch.setattr(backend, 'load_catalog', lambda source: pd.DataFrame({
        'magnitude_id': [8, 12], 'first_date': pd.to_datetime(['2024-01-01'] * 2),
        'last_date': pd.to_datetime(['2024-01-02 23:00'] * 2), 'low': [0., 0.], 'high': [100., 100.]}))
    monkeypatch.setattr(backend, 'load_sensors', lambda source, magnitudes: sensors)
    calls = []
    def measurements(source, magnitude, timestamp):
        calls.append(timestamp)
        return pd.DataFrame({'sensor_id': [1], 'value': [float(timestamp.hour)]})
    monkeypatch.setattr(backend, 'load_measurements', measurements)
    backend.clear_cache()
    client = TestClient(backend.app)
    try:
        html = client.get('/').text
        assert '/static/styles.css' in html and '/static/app.js' in html
        assert '<style' not in html and 'style=' not in html
        assert client.get('/static/styles.css').status_code == 200
        assert client.get('/static/app.js').status_code == 200
        catalog = client.get('/api/catalog?source=files').json()
        assert [item['id'] for item in catalog['pollutants']] == [8, 12]
        params = {'source': 'files', 'magnitude': 12, 'timestamp': '2024-01-01T00:00:00'}
        initial = client.get('/api/view', params=params)
        assert initial.status_code == 200
        assert initial.json()['stations'][0]['value'] == 0
        assert initial.json()['cell_count'] == 1
        assert f'data-cell="{initial.json()["stations"][0]["cell_id"]}"' in initial.json()['map']
        assert 'style=' not in initial.json()['map'] and '<script' not in initial.json()['map']
        client.get('/api/view', params=params)
        assert len(calls) == 1
        interpolated = client.get('/api/view', params={**params, 'method': 'idw'}).json()
        assert interpolated['method'] == 'IDW'
        assert interpolated['interpolated_count'] > 0
        assert interpolated['stations'][0]['value'] == 0
        assert client.get('/api/view', params={**params, 'method': 'unknown'}).status_code == 422
        params.update(magnitude=8, timestamp='2024-01-01T01:00:00')
        following = client.get('/api/view', params=params).json()
        assert following['name'] == 'NO₂' and following['mean'] == 1
        assert set(following['stations'][0]) == {'id', 'name', 'value', 'cell_id'}
        assert client.post('/api/refresh').status_code == 200
        client.get('/api/view', params=params)
        assert len(calls) == 3
        assert client.get('/api/catalog?source=invalid').status_code == 422
        params['timestamp'] = '2024-01-01T01:15:00'
        assert client.get('/api/view', params=params).status_code == 422
        params['timestamp'] = '2025-01-01T00:00:00'
        assert client.get('/api/view', params=params).status_code == 422
    finally:
        backend.clear_cache()


def test_api_failure_does_not_expose_backend_details(monkeypatch):
    import pytest
    pytest.importorskip('fastapi')
    pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    from metraq_app import app as backend
    def fail(source):
        raise RuntimeError('private connection details')
    monkeypatch.setattr(backend, 'load_catalog', fail)
    backend.clear_cache()
    try:
        response = TestClient(backend.app).get('/api/catalog?source=db')
        assert response.status_code == 503
        assert 'private connection details' not in response.text
    finally:
        backend.clear_cache()
