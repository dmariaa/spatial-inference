from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "BARCELONA" / "RAW"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "BARCELONA" / "station_catalog.csv"
DEFAULT_METRIC_CRS = "EPSG:25831"


@dataclass(frozen=True)
class BarcelonaRawFile:
    year: int
    month: int
    path: Path


def _iter_barcelona_raw_files(raw_dir: Path) -> list[BarcelonaRawFile]:
    files: list[BarcelonaRawFile] = []
    for path in sorted(raw_dir.glob("*.csv")):
        match = re.match(r"(?P<year>\d{4})_(?P<month>\d{2})_.+_qualitat_aire_BCN\.csv$", path.name)
        if match is None:
            continue

        files.append(
            BarcelonaRawFile(
                year=int(match.group("year")),
                month=int(match.group("month")),
                path=path,
            )
        )

    return sorted(files, key=lambda item: (item.year, item.month, item.path.name))


def _find_latest_station_snapshot(raw_dir: Path) -> Path:
    station_files: list[tuple[int, Path]] = []
    for path in sorted(raw_dir.glob("*_qualitat_aire_estacions.csv")):
        match = re.match(r"(?P<year>\d{4})_qualitat_aire_estacions\.csv$", path.name)
        if match is None:
            continue
        station_files.append((int(match.group("year")), path))

    if not station_files:
        raise FileNotFoundError(
            f"No Barcelona station snapshot found in {raw_dir}. "
            "Expected a file like 2026_qualitat_aire_estacions.csv."
        )

    return max(station_files, key=lambda item: (item[0], item[1].name))[1]


def _load_measurement_inventory(raw_dir: Path) -> tuple[pd.DataFrame, int]:
    raw_files = _iter_barcelona_raw_files(raw_dir)
    if not raw_files:
        raise FileNotFoundError(
            f"No Barcelona measurement CSV files found in {raw_dir}. "
            "Expected files like 2024_01_Gener_qualitat_aire_BCN.csv."
        )

    frames: list[pd.DataFrame] = []
    for raw_file in raw_files:
        frame = pd.read_csv(
            raw_file.path,
            encoding="utf-8",
            usecols=["ESTACIO", "CODI_CONTAMINANT", "ANY", "MES"],
        ).rename(
            columns={
                "ESTACIO": "station_code",
                "CODI_CONTAMINANT": "magnitude_code",
                "ANY": "year",
                "MES": "month",
            }
        )
        frame["station_code"] = frame["station_code"].astype("Int64").astype(str)
        frame["magnitude_code"] = frame["magnitude_code"].astype("Int64").astype(str)
        frame["month_key"] = (
            frame["year"].astype("Int64").astype(str) + "-" + frame["month"].astype("Int64").astype(str).str.zfill(2)
        )
        frames.append(frame)

    inventory = pd.concat(frames, ignore_index=True)
    inventory = inventory.sort_values(["station_code", "year", "month", "magnitude_code"]).reset_index(drop=True)
    return inventory, len(raw_files)


def _load_station_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Barcelona station snapshot not found: {path}")

    raw = pd.read_csv(path, encoding="utf-8")
    required = {
        "Estacio",
        "nom_cabina",
        "codi_dtes",
        "zqa",
        "codi_eoi",
        "Longitud",
        "Latitud",
        "ubicacio",
        "Codi_districte",
        "Nom_districte",
        "Codi_barri",
        "Nom_barri",
        "Clas_1",
        "Clas_2",
    }
    missing = required.difference(raw.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_text}")

    station_level = raw[
        [
            "Estacio",
            "nom_cabina",
            "codi_dtes",
            "zqa",
            "codi_eoi",
            "Longitud",
            "Latitud",
            "ubicacio",
            "Codi_districte",
            "Nom_districte",
            "Codi_barri",
            "Nom_barri",
            "Clas_1",
            "Clas_2",
        ]
    ].copy()

    station_level["station_code"] = station_level["Estacio"].astype("Int64").astype(str)
    station_level = (
        station_level.drop(columns=["Estacio"])
        .drop_duplicates(subset=["station_code"])
        .rename(
            columns={
                "nom_cabina": "station_name",
                "codi_dtes": "station_network_code",
                "zqa": "zone_code",
                "codi_eoi": "station_eoi_code",
                "Longitud": "longitude",
                "Latitud": "latitude",
                "ubicacio": "location",
                "Codi_districte": "district_code",
                "Nom_districte": "district_name",
                "Codi_barri": "neighborhood_code",
                "Nom_barri": "neighborhood_name",
                "Clas_1": "site_class_1",
                "Clas_2": "site_class_2",
            }
        )
        .sort_values("station_code")
        .reset_index(drop=True)
    )
    return station_level


def _assign_stable_ids(df: pd.DataFrame, existing_catalog_path: Path | None) -> pd.DataFrame:
    existing_ids: dict[str, int] = {}
    if existing_catalog_path is not None and existing_catalog_path.exists():
        existing = pd.read_csv(existing_catalog_path, encoding="utf-8")
        if {"station_code", "id"}.issubset(existing.columns):
            existing_ids = {
                str(row["station_code"]).strip(): int(row["id"])
                for _, row in existing.loc[:, ["station_code", "id"]].dropna().iterrows()
            }

    next_id = max(existing_ids.values(), default=0) + 1
    assigned_ids: list[int] = []
    for station_code in df["station_code"]:
        code = str(station_code).strip()
        station_id = existing_ids.get(code)
        if station_id is None:
            station_id = next_id
            existing_ids[code] = station_id
            next_id += 1
        assigned_ids.append(int(station_id))

    result = df.copy()
    result["id"] = assigned_ids
    return result


def _text_join(values: pd.Series) -> str:
    normalized = sorted({str(value).strip() for value in values if str(value).strip()})
    return ",".join(normalized)


def _build_experiment_exclusion_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row["is_in_measurement_history"]):
        reasons.append("not_in_measurements")
    if not bool(row["has_coordinates"]):
        reasons.append("missing_coordinates")
    if not bool(row["is_common_all_files"]):
        reasons.append("not_common_all_files")
    return ";".join(reasons)


def build_barcelona_station_catalog(
    *,
    raw_dir: Path | None = None,
    stations_path: Path | None = None,
    output_path: Path | None = None,
    existing_catalog_path: Path | None = None,
    metric_crs: str = DEFAULT_METRIC_CRS,
) -> pd.DataFrame:
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    stations_path = stations_path or _find_latest_station_snapshot(raw_dir)
    output_path = output_path or DEFAULT_CATALOG_PATH
    existing_catalog_path = existing_catalog_path or output_path

    inventory, n_raw_files = _load_measurement_inventory(raw_dir)
    station_snapshot = _load_station_snapshot(stations_path)
    transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)

    grouped = (
        inventory.groupby("station_code", sort=True)
        .agg(
            years_present=("year", lambda values: tuple(sorted({int(value) for value in values}))),
            months_present=("month_key", _text_join),
            pollutant_codes=("magnitude_code", _text_join),
        )
        .reset_index()
    )
    grouped["n_years_present"] = grouped["years_present"].apply(len).astype(int)
    grouped["years_present"] = grouped["years_present"].apply(lambda years: ",".join(str(year) for year in years))
    grouped["n_months_present"] = grouped["months_present"].apply(
        lambda value: len([item for item in str(value).split(",") if item])
    ).astype(int)
    grouped["first_month_present"] = grouped["months_present"].apply(lambda value: str(value).split(",")[0])
    grouped["last_month_present"] = grouped["months_present"].apply(lambda value: str(value).split(",")[-1])
    grouped["n_pollutants_present"] = grouped["pollutant_codes"].apply(
        lambda value: len([item for item in str(value).split(",") if item])
    ).astype(int)
    grouped["is_common_all_files"] = grouped["n_months_present"].eq(int(n_raw_files))
    grouped["is_in_measurement_history"] = True

    catalog = grouped.merge(station_snapshot, on="station_code", how="outer")
    catalog["station_name"] = catalog["station_name"].where(
        catalog["station_name"].astype("string").str.len().fillna(0) > 0,
        catalog["station_code"],
    )
    catalog["name"] = catalog["station_name"]
    catalog["source_dataset"] = "barcelona"
    catalog["has_station_metadata"] = catalog["latitude"].notna() & catalog["longitude"].notna()
    catalog["is_in_measurement_history"] = catalog["is_in_measurement_history"].eq(True)
    catalog["is_common_all_files"] = catalog["is_common_all_files"].eq(True)
    catalog["has_coordinates"] = catalog["latitude"].notna() & catalog["longitude"].notna()

    catalog["utm_x"] = np.nan
    catalog["utm_y"] = np.nan
    coords_mask = catalog["has_coordinates"]
    if coords_mask.any():
        utm_x, utm_y = transformer.transform(
            catalog.loc[coords_mask, "longitude"].to_numpy(dtype=float),
            catalog.loc[coords_mask, "latitude"].to_numpy(dtype=float),
        )
        catalog.loc[coords_mask, "utm_x"] = utm_x
        catalog.loc[coords_mask, "utm_y"] = utm_y

    catalog["metric_crs"] = metric_crs
    catalog["use_for_experiments"] = (
        catalog["is_in_measurement_history"]
        & catalog["has_coordinates"]
        & catalog["is_common_all_files"]
    )
    catalog["experiment_exclusion_reason"] = catalog.apply(_build_experiment_exclusion_reason, axis=1)

    fill_text_columns = [
        "years_present",
        "months_present",
        "first_month_present",
        "last_month_present",
        "pollutant_codes",
    ]
    for column in fill_text_columns:
        if column in catalog.columns:
            catalog[column] = catalog[column].fillna("")

    fill_count_columns = ["n_years_present", "n_months_present", "n_pollutants_present"]
    for column in fill_count_columns:
        if column in catalog.columns:
            catalog[column] = catalog[column].fillna(0).astype(int)

    catalog = _assign_stable_ids(catalog.sort_values("station_code").reset_index(drop=True), existing_catalog_path)
    catalog = catalog.sort_values("id").reset_index(drop=True)
    catalog = catalog[
        [
            "id",
            "station_code",
            "station_name",
            "name",
            "latitude",
            "longitude",
            "utm_x",
            "utm_y",
            "metric_crs",
            "source_dataset",
            "station_network_code",
            "zone_code",
            "station_eoi_code",
            "location",
            "district_code",
            "district_name",
            "neighborhood_code",
            "neighborhood_name",
            "site_class_1",
            "site_class_2",
            "years_present",
            "n_years_present",
            "months_present",
            "n_months_present",
            "first_month_present",
            "last_month_present",
            "pollutant_codes",
            "n_pollutants_present",
            "has_station_metadata",
            "has_coordinates",
            "is_in_measurement_history",
            "is_common_all_files",
            "use_for_experiments",
            "experiment_exclusion_reason",
        ]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False, encoding="utf-8")
    return catalog


if __name__ == "__main__":
    catalog = build_barcelona_station_catalog()
    selected = catalog[catalog["use_for_experiments"]]
    print(f"Wrote {len(catalog)} Barcelona stations to {DEFAULT_CATALOG_PATH}")
    print(f"Selected for experiments: {len(selected)} station(s)")
