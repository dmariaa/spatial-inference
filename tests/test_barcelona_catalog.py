from __future__ import annotations

import pandas as pd

from metraq_dip.data.barcelona_catalog import build_barcelona_station_catalog


def _write_barcelona_measurement_csv(path, rows: list[str]) -> None:
    header = (
        "CODI_PROVINCIA,PROVINCIA,CODI_MUNICIPI,MUNICIPI,ESTACIO,CODI_CONTAMINANT,ANY,MES,DIA,"
        "H01,V01,H02,V02,H03,V03,H04,V04,H05,V05,H06,V06,H07,V07,H08,V08,H09,V09,H10,V10,"
        "H11,V11,H12,V12,H13,V13,H14,V14,H15,V15,H16,V16,H17,V17,H18,V18,H19,V19,H20,V20,"
        "H21,V21,H22,V22,H23,V23,H24,V24"
    )
    path.write_text("\n".join([header, *rows]), encoding="utf-8")


def test_build_barcelona_station_catalog_dedupes_snapshot_and_computes_selection_flags(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    _write_barcelona_measurement_csv(
        raw_dir / "2023_01_Gener_qualitat_aire_BCN.csv",
        [
            "8,Barcelona,19,Barcelona,4,8,2023,1,1,10,V,11,V,12,V,13,V,14,V,15,V,16,V,17,V,18,V,19,V,20,V,21,V,22,V,23,V,24,V,25,V,26,V,27,V,28,V,29,V,30,V,31,V,32,V,33,V",
            "8,Barcelona,19,Barcelona,42,8,2023,1,1,5,V,6,V,7,V,8,V,9,V,10,V,11,V,12,V,13,V,14,V,15,V,16,V,17,V,18,V,19,V,20,V,21,V,22,V,23,V,24,V,25,V,26,V,27,V,28,V",
        ],
    )
    _write_barcelona_measurement_csv(
        raw_dir / "2023_02_Febrer_qualitat_aire_BCN.csv",
        [
            "8,Barcelona,19,Barcelona,4,8,2023,2,1,10,V,11,V,12,V,13,V,14,V,15,V,16,V,17,V,18,V,19,V,20,V,21,V,22,V,23,V,24,V,25,V,26,V,27,V,28,V,29,V,30,V,31,V,32,V,33,V",
        ],
    )

    stations_path = raw_dir / "2026_qualitat_aire_estacions.csv"
    stations_path.write_text(
        "\n".join(
            [
                "Estacio,nom_cabina,codi_dtes,zqa,codi_eoi,Longitud,Latitud,ubicacio,Codi_districte,Nom_districte,Codi_barri,Nom_barri,Clas_1,Clas_2,Codi_Contaminant",
                "4,Barcelona - Poblenou,I2,1,8019004,2.2045,41.4039,Poblenou,10,Sant Marti,68,el Poblenou,Urbana,Fons,8",
                "4,Barcelona - Poblenou,I2,1,8019004,2.2045,41.4039,Poblenou,10,Sant Marti,68,el Poblenou,Urbana,Fons,10",
                "42,Barcelona - Sants,ID,1,8019042,2.1331,41.3788,Sants,3,Sants-Montjuic,18,Sants,Urbana,Fons,8",
                "60,Barcelona - Navas,IA,1,8019060,2.1871,41.4159,Navas,9,Sant Andreu,63,Navas,Urbana,Transit,8",
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "station_catalog.csv"
    catalog = build_barcelona_station_catalog(
        raw_dir=raw_dir,
        stations_path=stations_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert catalog["station_code"].tolist() == ["4", "42", "60"]

    poblenou = catalog.loc[catalog["station_code"] == "4"].iloc[0]
    assert poblenou["station_name"] == "Barcelona - Poblenou"
    assert poblenou["years_present"] == "2023"
    assert poblenou["months_present"] == "2023-01,2023-02"
    assert bool(poblenou["is_common_all_files"]) is True
    assert bool(poblenou["use_for_experiments"]) is True
    assert pd.notna(poblenou["utm_x"])
    assert pd.notna(poblenou["utm_y"])

    sants = catalog.loc[catalog["station_code"] == "42"].iloc[0]
    assert bool(sants["is_common_all_files"]) is False
    assert bool(sants["use_for_experiments"]) is False
    assert "not_common_all_files" in str(sants["experiment_exclusion_reason"])

    navas = catalog.loc[catalog["station_code"] == "60"].iloc[0]
    assert bool(navas["is_in_measurement_history"]) is False
    assert bool(navas["use_for_experiments"]) is False
    assert "not_in_measurements" in str(navas["experiment_exclusion_reason"])


def test_build_barcelona_station_catalog_preserves_existing_ids(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    _write_barcelona_measurement_csv(
        raw_dir / "2023_01_Gener_qualitat_aire_BCN.csv",
        [
            "8,Barcelona,19,Barcelona,4,8,2023,1,1,10,V,11,V,12,V,13,V,14,V,15,V,16,V,17,V,18,V,19,V,20,V,21,V,22,V,23,V,24,V,25,V,26,V,27,V,28,V,29,V,30,V,31,V,32,V,33,V",
            "8,Barcelona,19,Barcelona,42,8,2023,1,1,5,V,6,V,7,V,8,V,9,V,10,V,11,V,12,V,13,V,14,V,15,V,16,V,17,V,18,V,19,V,20,V,21,V,22,V,23,V,24,V,25,V,26,V,27,V,28,V",
            "8,Barcelona,19,Barcelona,50,8,2023,1,1,5,V,6,V,7,V,8,V,9,V,10,V,11,V,12,V,13,V,14,V,15,V,16,V,17,V,18,V,19,V,20,V,21,V,22,V,23,V,24,V,25,V,26,V,27,V,28,V",
        ],
    )

    stations_path = raw_dir / "2026_qualitat_aire_estacions.csv"
    stations_path.write_text(
        "\n".join(
            [
                "Estacio,nom_cabina,codi_dtes,zqa,codi_eoi,Longitud,Latitud,ubicacio,Codi_districte,Nom_districte,Codi_barri,Nom_barri,Clas_1,Clas_2,Codi_Contaminant",
                "4,Barcelona - Poblenou,I2,1,8019004,2.2045,41.4039,Poblenou,10,Sant Marti,68,el Poblenou,Urbana,Fons,8",
                "42,Barcelona - Sants,ID,1,8019042,2.1331,41.3788,Sants,3,Sants-Montjuic,18,Sants,Urbana,Fons,8",
                "50,Barcelona - Ciutadella,IL,1,8019050,2.1874,41.3864,Ciutadella,1,Ciutat Vella,4,Ribera,Urbana,Fons,8",
            ]
        ),
        encoding="utf-8",
    )

    existing_catalog_path = tmp_path / "station_catalog.csv"
    existing_catalog_path.write_text(
        "\n".join(
            [
                "id,station_code",
                "9,4",
                "42,42",
            ]
        ),
        encoding="utf-8",
    )

    catalog = build_barcelona_station_catalog(
        raw_dir=raw_dir,
        stations_path=stations_path,
        output_path=existing_catalog_path,
        existing_catalog_path=existing_catalog_path,
    )

    ids = dict(zip(catalog["station_code"], catalog["id"], strict=False))
    assert ids["4"] == 9
    assert ids["42"] == 42
    assert ids["50"] == 43
