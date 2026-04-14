from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_ns_dtype, is_integer_dtype

from metraq_dip.data.airparif_files import AirparifFiles
from metraq_dip.tools.grid import prepare_grid_context


def test_airparif_files_backend_reads_curated_yearly_files(tmp_path):
    data_dir = tmp_path / "AIRPARIF"
    data_dir.mkdir()

    station_catalog_path = data_dir / "station_catalog.csv"
    station_catalog_path.write_text(
        "\n".join(
            [
                "id,station_code,station_name,latitude,longitude,utm_x,utm_y,metric_crs,use_for_experiments",
                "1,A1,Station A1,48.9000,2.3000,448000.0,5416500.0,EPSG:32631,True",
                "2,B1,Station B1,48.8500,2.3500,452000.0,5411000.0,EPSG:32631,False",
            ]
        ),
        encoding="utf-8",
    )

    magnitude_catalog_path = data_dir / "magnitude_catalog.csv"
    magnitude_catalog_path.write_text(
        "\n".join(
            [
                "id,magnitude_code,magnitude_name,unit,source_dataset,years_present,n_years_present",
                "1,NO2,dioxyde d azote,microg/m3,airparif,2024,1",
            ]
        ),
        encoding="utf-8",
    )

    measurements_path = data_dir / "airparif_aq-2024.csv"
    measurements_path.write_text(
        "\n".join(
            [
                "sensor_id,sensor_code,sensor_name,latitude,longitude,utm_x,utm_y,magnitude_id,magnitude_code,magnitude_name,unit,entry_date,value",
                "1,A1,Station A1,48.9,2.3,448000.0,5416500.0,1,NO2,dioxyde d azote,microg/m3,2024-01-01T00:00:00Z,10.0",
                "2,B1,Station B1,48.85,2.35,452000.0,5411000.0,1,NO2,dioxyde d azote,microg/m3,2024-01-01T00:00:00Z,20.0",
                "1,A1,Station A1,48.9,2.3,448000.0,5416500.0,1,NO2,dioxyde d azote,microg/m3,2024-01-01T01:00:00Z,12.0",
            ]
        ),
        encoding="utf-8",
    )

    backend = AirparifFiles(
        data_dir=data_dir,
        station_catalog_path=station_catalog_path,
        magnitude_catalog_path=magnitude_catalog_path,
    )

    sensors = backend.get_sensors(magnitudes=[1])
    assert sensors["id"].tolist() == [1]
    assert sensors["station_code"].tolist() == ["A1"]
    assert sensors["name"].tolist() == ["Station A1"]
    assert list(sensors.columns).count("name") == 1
    assert sensors["metric_crs"].tolist() == ["EPSG:32631"]

    explicit_sensors = backend.get_sensors(sensors=[2])
    assert explicit_sensors["id"].tolist() == [2]

    measurements = backend.get_measurements(
        start_date=pd.Timestamp("2024-01-01 00:00:00").to_pydatetime(),
        end_date=pd.Timestamp("2024-01-01 01:00:00").to_pydatetime(),
        magnitudes=[1],
    )
    assert measurements["sensor_id"].tolist() == [1, 1, 2]
    assert is_integer_dtype(measurements["sensor_id"])
    assert is_integer_dtype(measurements["magnitude_id"])
    assert is_datetime64_ns_dtype(measurements["entry_date"])
    assert measurements["entry_date"].dt.tz is None

    bounds = backend.get_magnitude_bounds([1])
    assert bounds == {1: (10.0, 20.0)}


def test_prepare_grid_context_uses_metric_crs_and_catalog_utm_coordinates():
    sensors = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["A1", "B1"],
            "latitude": [48.9000, 48.8500],
            "longitude": [2.3000, 2.3500],
            "utm_x": [448000.0, 452000.0],
            "utm_y": [5416500.0, 5411000.0],
            "metric_crs": ["EPSG:32631", "EPSG:32631"],
        }
    )

    ctx = prepare_grid_context(sensors, cell_size_m=1000, margin_m_x=1000, margin_m_y=1000)

    assert ctx["metric_crs"] == "EPSG:32631"
    np.testing.assert_allclose(ctx["xs"], np.array([448000.0, 452000.0]))
    np.testing.assert_allclose(ctx["ys"], np.array([5416500.0, 5411000.0]))
