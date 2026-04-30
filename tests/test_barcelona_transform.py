from __future__ import annotations

import pandas as pd

from metraq_dip.data.barcelona_transform import (
    build_barcelona_magnitude_catalog,
    build_barcelona_measurement_store,
)


def _write_barcelona_measurement_csv(path, rows: list[str]) -> None:
    header = (
        "CODI_PROVINCIA,PROVINCIA,CODI_MUNICIPI,MUNICIPI,ESTACIO,CODI_CONTAMINANT,ANY,MES,DIA,"
        "H01,V01,H02,V02,H03,V03,H04,V04,H05,V05,H06,V06,H07,V07,H08,V08,H09,V09,H10,V10,"
        "H11,V11,H12,V12,H13,V13,H14,V14,H15,V15,H16,V16,H17,V17,H18,V18,H19,V19,H20,V20,"
        "H21,V21,H22,V22,H23,V23,H24,V24"
    )
    path.write_text("\n".join([header, *rows]), encoding="utf-8")


def _measurement_row(
    *,
    station: int,
    magnitude: int,
    year: int,
    month: int,
    day: int,
    valid_hours: dict[int, float],
) -> str:
    parts = ["8", "Barcelona", "19", "Barcelona", str(station), str(magnitude), str(year), str(month), str(day)]
    for hour in range(1, 25):
        if hour in valid_hours:
            parts.extend([str(valid_hours[hour]), "V"])
        else:
            parts.extend(["0", "N"])
    return ",".join(parts)


def test_build_barcelona_measurement_store_writes_long_yearly_files_and_spills_h24_to_next_year(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    _write_barcelona_measurement_csv(
        raw_dir / "2023_12_Desembre_qualitat_aire_BCN.csv",
        [
            _measurement_row(station=4, magnitude=8, year=2023, month=12, day=31, valid_hours={1: 10, 24: 99}),
            _measurement_row(station=4, magnitude=995, year=2023, month=12, day=31, valid_hours={1: 1}),
        ],
    )
    _write_barcelona_measurement_csv(
        raw_dir / "2024_01_Gener_qualitat_aire_BCN.csv",
        [
            _measurement_row(station=4, magnitude=8, year=2024, month=1, day=1, valid_hours={1: 20}),
        ],
    )

    station_catalog_path = tmp_path / "station_catalog.csv"
    station_catalog_path.write_text(
        "\n".join(
            [
                "id,station_code,station_name,name,latitude,longitude,utm_x,utm_y",
                "1,4,Barcelona - Poblenou,Barcelona - Poblenou,41.4039,2.2045,434000.0,4584000.0",
            ]
        ),
        encoding="utf-8",
    )

    pollutant_catalog_path = tmp_path / "qualitat_aire_contaminants.csv"
    pollutant_catalog_path.write_text(
        "\n".join(
            [
                "Codi_Contaminant,Desc_Contaminant,Unitats",
                "8,NO2,microg/m3",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    magnitude_catalog_path = output_dir / "magnitude_catalog.csv"

    written = build_barcelona_measurement_store(
        raw_dir=raw_dir,
        station_catalog_path=station_catalog_path,
        pollutant_catalog_path=pollutant_catalog_path,
        magnitude_catalog_path=magnitude_catalog_path,
        output_dir=output_dir,
    )

    assert written.keys() == {2023, 2024}
    assert magnitude_catalog_path.exists()

    magnitudes = pd.read_csv(magnitude_catalog_path)
    assert magnitudes["magnitude_code"].tolist() == [8, 995]
    assert magnitudes["metadata_source"].tolist() == ["official_catalog", "observed_only"]
    assert magnitudes["is_missing_official_metadata"].tolist() == [False, True]

    measurements_2023 = pd.read_csv(written[2023])
    measurements_2024 = pd.read_csv(written[2024])

    assert measurements_2023["entry_date"].tolist() == ["2023-12-31T01:00:00", "2023-12-31T01:00:00"]
    assert set(measurements_2023["magnitude_code"].astype(str)) == {"8", "995"}

    assert measurements_2024["entry_date"].tolist() == ["2024-01-01T00:00:00", "2024-01-01T01:00:00"]
    assert measurements_2024["value"].tolist() == [99.0, 20.0]


def test_build_barcelona_magnitude_catalog_preserves_existing_ids(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    _write_barcelona_measurement_csv(
        raw_dir / "2023_01_Gener_qualitat_aire_BCN.csv",
        [
            _measurement_row(station=4, magnitude=8, year=2023, month=1, day=1, valid_hours={1: 10}),
            _measurement_row(station=4, magnitude=995, year=2023, month=1, day=1, valid_hours={1: 1}),
        ],
    )

    pollutant_catalog_path = tmp_path / "qualitat_aire_contaminants.csv"
    pollutant_catalog_path.write_text(
        "\n".join(
            [
                "Codi_Contaminant,Desc_Contaminant,Unitats",
                "995,Instrument Internal,microg/m3",
                "8,NO2,microg/m3",
            ]
        ),
        encoding="utf-8",
    )

    existing_catalog_path = tmp_path / "magnitude_catalog.csv"
    existing_catalog_path.write_text(
        "\n".join(
            [
                "id,magnitude_code,magnitude_name,unit,source_dataset,years_present,n_years_present,station_codes,n_stations_present,metadata_source,is_missing_official_metadata",
                "7,995,Instrument Internal,microg/m3,barcelona,2023,1,4,1,official_catalog,False",
            ]
        ),
        encoding="utf-8",
    )

    catalog = build_barcelona_magnitude_catalog(
        raw_dir=raw_dir,
        pollutant_catalog_path=pollutant_catalog_path,
        output_path=existing_catalog_path,
        existing_catalog_path=existing_catalog_path,
    )

    ids = dict(zip(catalog["magnitude_code"].astype(str), catalog["id"], strict=False))
    assert ids["995"] == 7
    assert ids["8"] == 8
