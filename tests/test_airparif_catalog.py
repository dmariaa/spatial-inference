from __future__ import annotations

import csv

import pandas as pd

from metraq_dip.data.airparif_catalog import build_airparif_station_catalog


def _write_airparif_header_csv(path, stations: list[tuple[str, str]]) -> None:
    rows = [
        [""] + [f"{station_code}:NO2" for station_code, _ in stations],
        [""] + [station_name for _, station_name in stations],
        [""] + [station_code for station_code, _ in stations],
        [""] + ["dioxyde d azote"] * len(stations),
        [""] + ["NO2"] * len(stations),
        [""] + ["microg/m3"] * len(stations),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def test_build_airparif_station_catalog_computes_selection_flags(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    for year in range(2018, 2025):
        stations = [("A1", "Autoroute A1 - Saint-Denis"), ("EIFF3", "Tour Eiffel 3ème étage")]
        if year == 2018:
            stations.append(("X1", "Extra Station"))
        _write_airparif_header_csv(raw_dir / f"{year}-NO2.csv", stations)

    coords_path = raw_dir / "common_no2_stations_2018_2024_coordinates.csv"
    coords_path.write_text(
        "\n".join(
            [
                "station_code,station_name,latitude,longitude,source",
                "A1,Autoroute A1 - Saint-Denis,48.925300,2.356700,EEA",
                "EIFF3,Tour Eiffel 3ème étage,48.857850,2.292423,Proxy",
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "station_catalog.csv"
    catalog = build_airparif_station_catalog(
        raw_dir=raw_dir,
        coords_path=coords_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert catalog["station_code"].tolist() == ["A1", "EIFF3", "X1"]

    a1 = catalog.loc[catalog["station_code"] == "A1"].iloc[0]
    assert a1["years_present"] == "2018,2019,2020,2021,2022,2023,2024"
    assert bool(a1["is_common_2018_2024"]) is True
    assert bool(a1["use_for_experiments"]) is True
    assert pd.notna(a1["utm_x"])
    assert pd.notna(a1["utm_y"])

    eiff3 = catalog.loc[catalog["station_code"] == "EIFF3"].iloc[0]
    assert bool(eiff3["is_common_2018_2024"]) is True
    assert bool(eiff3["use_for_experiments"]) is False
    assert "excluded_station" in str(eiff3["experiment_exclusion_reason"])

    x1 = catalog.loc[catalog["station_code"] == "X1"].iloc[0]
    assert bool(x1["is_common_2018_2024"]) is False
    assert bool(x1["has_coordinates"]) is False
    assert bool(x1["use_for_experiments"]) is False
    assert "missing_coordinates" in str(x1["experiment_exclusion_reason"])
    assert "not_common_2018_2024" in str(x1["experiment_exclusion_reason"])


def test_build_airparif_station_catalog_preserves_existing_ids(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    for year in range(2018, 2025):
        _write_airparif_header_csv(
            raw_dir / f"{year}-NO2.csv",
            [("A1", "Autoroute A1 - Saint-Denis"), ("B1", "Station B1"), ("C1", "Station C1")],
        )

    coords_path = raw_dir / "common_no2_stations_2018_2024_coordinates.csv"
    coords_path.write_text(
        "\n".join(
            [
                "station_code,station_name,latitude,longitude,source",
                "A1,Autoroute A1 - Saint-Denis,48.925300,2.356700,EEA",
                "B1,Station B1,48.856600,2.352200,Test",
                "C1,Station C1,48.860000,2.350000,Test",
            ]
        ),
        encoding="utf-8",
    )

    existing_catalog_path = tmp_path / "station_catalog.csv"
    existing_catalog_path.write_text(
        "\n".join(
            [
                "id,station_code",
                "9,A1",
                "42,B1",
            ]
        ),
        encoding="utf-8",
    )

    catalog = build_airparif_station_catalog(
        raw_dir=raw_dir,
        coords_path=coords_path,
        output_path=existing_catalog_path,
        existing_catalog_path=existing_catalog_path,
    )

    ids = dict(zip(catalog["station_code"], catalog["id"], strict=False))
    assert ids["A1"] == 9
    assert ids["B1"] == 42
    assert ids["C1"] == 43
