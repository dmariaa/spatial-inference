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


def test_interface_changes_hour_and_pollutant(monkeypatch):
    import pytest
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from metraq_app import data_source
    sensors = pd.DataFrame({"id": [1], "name": ["A"], "utm_x": [440100.],
        "utm_y": [4470100.], "latitude": [40.38], "longitude": [-3.70]})
    monkeypatch.setattr(data_source, "load_catalog", lambda source: pd.DataFrame({
        "magnitude_id": [8, 12], "first_date": pd.to_datetime(["2024-01-01"] * 2),
        "last_date": pd.to_datetime(["2024-01-02 23:00"] * 2), "low": [0., 0.], "high": [100., 100.]}))
    monkeypatch.setattr(data_source, "load_sensors", lambda source, magnitudes: sensors)
    monkeypatch.setattr(data_source, "load_measurements", lambda source, magnitude, timestamp:
        pd.DataFrame({"sensor_id": [1], "value": [float(timestamp.hour)]}))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "src/metraq_app/app.py").run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "1"
    app.button[1].click().run()
    assert not app.exception
    assert "01:00" in app.subheader[0].value
    app.selectbox[1].set_value(8).run()
    assert not app.exception
    assert "NO₂" in app.subheader[0].value
    app.selectbox[2].set_value(None).run()
    assert not app.exception
    assert app.info


