from __future__ import annotations

import csv

import pandas as pd

from metraq_dip.data.airparif_transform import (
    build_airparif_magnitude_catalog,
    build_airparif_measurement_store,
)


def _write_airparif_raw_csv(
    path,
    *,
    pollutant_code: str,
    pollutant_name: str,
    unit: str,
    stations: list[tuple[str, str]],
    rows: list[tuple[str, list[float | str | None]]],
) -> None:
    header_rows = [
        [""] + [f"{station_code}:{pollutant_code}" for station_code, _ in stations],
        [""] + [station_name for _, station_name in stations],
        [""] + [station_code for station_code, _ in stations],
        [""] + [pollutant_name] * len(stations),
        [""] + [pollutant_code] * len(stations),
        [""] + [unit] * len(stations),
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(header_rows)
        for entry_date, values in rows:
            writer.writerow([entry_date, *values])


def test_build_airparif_measurement_store_writes_long_yearly_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    station_catalog_path = tmp_path / "station_catalog.csv"
    station_catalog_path.write_text(
        "\n".join(
            [
                "id,station_code,station_name,name,latitude,longitude,utm_x,utm_y",
                "1,A1,Station A1,Station A1,48.9000,2.3000,448000.0,5416500.0",
                "2,B1,Station B1,Station B1,48.8500,2.3500,452000.0,5411000.0",
            ]
        ),
        encoding="utf-8",
    )

    _write_airparif_raw_csv(
        raw_dir / "2018-NO2.csv",
        pollutant_code="NO2",
        pollutant_name="dioxyde d azote",
        unit="microg/m3",
        stations=[("A1", "Station A1"), ("B1", "Station B1")],
        rows=[
            ("2018-01-01 01:00:00Z", [10.0, None]),
            ("2018-01-01 02:00:00Z", [11.0, 12.0]),
        ],
    )
    _write_airparif_raw_csv(
        raw_dir / "2018-PM10.csv",
        pollutant_code="PM10",
        pollutant_name="particules PM10",
        unit="microg/m3",
        stations=[("A1", "Station A1"), ("B1", "Station B1")],
        rows=[
            ("2018-01-01 01:00:00Z", [20.0, 21.0]),
            ("2018-01-01 02:00:00Z", [None, 22.0]),
        ],
    )

    output_dir = tmp_path / "out"
    magnitude_catalog_path = output_dir / "magnitude_catalog.csv"

    written = build_airparif_measurement_store(
        raw_dir=raw_dir,
        station_catalog_path=station_catalog_path,
        magnitude_catalog_path=magnitude_catalog_path,
        output_dir=output_dir,
    )

    assert written.keys() == {2018}
    output_path = written[2018]
    assert output_path.exists()
    assert magnitude_catalog_path.exists()

    magnitudes = pd.read_csv(magnitude_catalog_path)
    assert magnitudes["magnitude_code"].tolist() == ["NO2", "PM10"]
    assert magnitudes["id"].tolist() == [1, 2]

    measurements = pd.read_csv(output_path)
    assert measurements.columns.tolist() == [
        "sensor_id",
        "sensor_code",
        "sensor_name",
        "latitude",
        "longitude",
        "utm_x",
        "utm_y",
        "magnitude_id",
        "magnitude_code",
        "magnitude_name",
        "unit",
        "entry_date",
        "value",
    ]
    assert len(measurements) == 6
    assert set(measurements["magnitude_code"]) == {"NO2", "PM10"}
    assert set(measurements["sensor_code"]) == {"A1", "B1"}
    assert measurements["entry_date"].tolist()[0] == "2018-01-01T01:00:00Z"


def test_build_airparif_magnitude_catalog_preserves_existing_ids(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    _write_airparif_raw_csv(
        raw_dir / "2018-NO2.csv",
        pollutant_code="NO2",
        pollutant_name="dioxyde d azote",
        unit="microg/m3",
        stations=[("A1", "Station A1")],
        rows=[("2018-01-01 01:00:00Z", [10.0])],
    )
    _write_airparif_raw_csv(
        raw_dir / "2018-PM10.csv",
        pollutant_code="PM10",
        pollutant_name="particules PM10",
        unit="microg/m3",
        stations=[("A1", "Station A1")],
        rows=[("2018-01-01 01:00:00Z", [20.0])],
    )

    existing_catalog_path = tmp_path / "magnitude_catalog.csv"
    existing_catalog_path.write_text(
        "\n".join(
            [
                "id,magnitude_code,magnitude_name,unit,source_dataset,years_present,n_years_present",
                "7,PM10,particules PM10,microg/m3,airparif,2018,1",
            ]
        ),
        encoding="utf-8",
    )

    catalog = build_airparif_magnitude_catalog(
        raw_dir=raw_dir,
        output_path=existing_catalog_path,
        existing_catalog_path=existing_catalog_path,
    )

    ids = dict(zip(catalog["magnitude_code"], catalog["id"], strict=False))
    assert ids["PM10"] == 7
    assert ids["NO2"] == 8
