from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_ns_dtype, is_integer_dtype

from metraq_dip.data.aq_backends import get_aq_backend
from metraq_dip.data.barcelona_files import BarcelonaFiles
from metraq_dip.tools.grid import prepare_grid_context


def test_barcelona_files_backend_reads_curated_yearly_files(tmp_path):
    data_dir = tmp_path / "BARCELONA"
    data_dir.mkdir()

    station_catalog_path = data_dir / "station_catalog.csv"
    station_catalog_path.write_text(
        "\n".join(
            [
                "id,station_code,station_name,name,latitude,longitude,utm_x,utm_y,metric_crs,use_for_experiments",
                "1,4,Barcelona - Poblenou,Barcelona - Poblenou,41.4039,2.2045,434000.0,4584000.0,EPSG:25831,True",
                "2,42,Barcelona - Sants,Barcelona - Sants,41.3788,2.1331,428000.0,4581000.0,EPSG:25831,False",
            ]
        ),
        encoding="utf-8",
    )

    magnitude_catalog_path = data_dir / "magnitude_catalog.csv"
    magnitude_catalog_path.write_text(
        "\n".join(
            [
                "id,magnitude_code,magnitude_name,unit,source_dataset,years_present,n_years_present,station_codes,n_stations_present,metadata_source,is_missing_official_metadata",
                "1,8,NO2,microg/m3,barcelona,2024,1,\"4,42\",2,official_catalog,False",
            ]
        ),
        encoding="utf-8",
    )

    measurements_path = data_dir / "barcelona_aq-2024.csv"
    measurements_path.write_text(
        "\n".join(
            [
                "sensor_id,sensor_code,sensor_name,latitude,longitude,utm_x,utm_y,magnitude_id,magnitude_code,magnitude_name,unit,entry_date,value",
                "1,4,Barcelona - Poblenou,41.4039,2.2045,434000.0,4584000.0,1,8,NO2,microg/m3,2024-01-01T00:00:00,10.0",
                "2,42,Barcelona - Sants,41.3788,2.1331,428000.0,4581000.0,1,8,NO2,microg/m3,2024-01-01T00:00:00,20.0",
                "1,4,Barcelona - Poblenou,41.4039,2.2045,434000.0,4584000.0,1,8,NO2,microg/m3,2024-01-01T01:00:00,12.0",
            ]
        ),
        encoding="utf-8",
    )

    backend = BarcelonaFiles(
        data_dir=data_dir,
        station_catalog_path=station_catalog_path,
        magnitude_catalog_path=magnitude_catalog_path,
    )

    sensors = backend.get_sensors(magnitudes=[1])
    assert sensors["id"].tolist() == [1]
    assert sensors["station_code"].tolist() == [4]
    assert sensors["name"].tolist() == ["Barcelona - Poblenou"]
    assert list(sensors.columns).count("name") == 1
    assert sensors["metric_crs"].tolist() == ["EPSG:25831"]

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

    bounds = backend.get_magnitude_bounds([1])
    assert bounds == {1: (10.0, 20.0)}


def test_prepare_grid_context_uses_barcelona_metric_crs_and_catalog_utm_coordinates():
    sensors = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Poblenou", "Sants"],
            "latitude": [41.4039, 41.3788],
            "longitude": [2.2045, 2.1331],
            "utm_x": [434000.0, 428000.0],
            "utm_y": [4584000.0, 4581000.0],
            "metric_crs": ["EPSG:25831", "EPSG:25831"],
        }
    )

    ctx = prepare_grid_context(sensors, cell_size_m=1000, margin_m_x=1000, margin_m_y=1000)

    assert ctx["metric_crs"] == "EPSG:25831"


def test_barcelona_backend_is_registered():
    backend = get_aq_backend(dataset="barcelona", backend="files")

    assert backend.dataset_name == "barcelona"
    assert backend.backend_name == "files"
