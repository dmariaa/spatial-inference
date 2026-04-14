from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "AIRPARIF" / "raw"
DEFAULT_COORDS_PATH = DEFAULT_RAW_DIR / "common_no2_stations_2018_2024_coordinates.csv"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "AIRPARIF" / "station_catalog.csv"

DEFAULT_COMMON_YEARS = tuple(range(2018, 2025))
DEFAULT_EXCLUDED_STATION_CODES = ("EIFF3",)
DEFAULT_EXPERIMENT_RADIUS_KM = 15.0
DEFAULT_PARIS_CENTER_LAT = 48.8566
DEFAULT_PARIS_CENTER_LON = 2.3522
DEFAULT_METRIC_CRS = "EPSG:32631"


def _iter_raw_no2_files(raw_dir: Path) -> list[Path]:
    files: list[tuple[int, Path]] = []
    for path in sorted(raw_dir.glob("*-NO2.csv")):
        match = re.match(r"(?P<year>\d{4})-NO2\.csv$", path.name)
        if match is None:
            continue
        files.append((int(match.group("year")), path))

    return [path for _, path in sorted(files)]


def _read_header_rows(csv_path: Path, *, n_rows: int = 6) -> list[list[str]]:
    rows: list[list[str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(n_rows):
            try:
                rows.append(next(reader))
            except StopIteration as exc:
                raise ValueError(f"{csv_path} does not contain the expected {n_rows} AIRPARIF header rows.") from exc

    return rows


def _parse_header_inventory(csv_path: Path) -> pd.DataFrame:
    year_match = re.match(r"(?P<year>\d{4})-NO2\.csv$", csv_path.name)
    if year_match is None:
        raise ValueError(f"Unexpected AIRPARIF file name: {csv_path.name}")

    year = int(year_match.group("year"))
    header_rows = _read_header_rows(csv_path)
    width = max(len(row) for row in header_rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in header_rows]

    combined_codes = normalized_rows[0][1:]
    station_names = normalized_rows[1][1:]
    station_codes = normalized_rows[2][1:]
    pollutant_names = normalized_rows[3][1:]
    pollutant_codes = normalized_rows[4][1:]
    units = normalized_rows[5][1:]

    records: list[dict[str, object]] = []
    for combined_code, station_name, station_code, pollutant_name, pollutant_code, unit in zip(
        combined_codes,
        station_names,
        station_codes,
        pollutant_names,
        pollutant_codes,
        units,
        strict=False,
    ):
        code = station_code.strip()
        if not code and combined_code.strip():
            code = combined_code.split(":", 1)[0].strip()
        if not code:
            continue

        records.append(
            {
                "year": year,
                "station_code": code,
                "station_name": station_name.strip() or code,
                "combined_code": combined_code.strip(),
                "pollutant_code": pollutant_code.strip(),
                "pollutant_name": pollutant_name.strip(),
                "unit": unit.strip(),
            }
        )

    return pd.DataFrame.from_records(records)


def _load_inventory(raw_dir: Path) -> pd.DataFrame:
    frames = [_parse_header_inventory(path) for path in _iter_raw_no2_files(raw_dir)]
    if not frames:
        raise FileNotFoundError(
            f"No AIRPARIF NO2 files found in {raw_dir}. Expected files like 2018-NO2.csv."
        )

    inventory = pd.concat(frames, ignore_index=True)
    inventory = inventory.sort_values(["station_code", "year"]).reset_index(drop=True)
    return inventory


def _load_coordinates(coords_path: Path) -> pd.DataFrame:
    if not coords_path.exists():
        raise FileNotFoundError(f"AIRPARIF coordinates file not found: {coords_path}")

    coords = pd.read_csv(coords_path, encoding="utf-8")
    required = {"station_code", "station_name", "latitude", "longitude"}
    missing = required.difference(coords.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{coords_path} is missing required columns: {missing_text}")

    coords = coords.rename(columns={"station_name": "coords_station_name"})
    return coords


def _years_to_text(years: Iterable[int]) -> str:
    return ",".join(str(int(year)) for year in sorted(set(years)))


def _assign_stable_ids(df: pd.DataFrame, existing_catalog_path: Path | None) -> pd.DataFrame:
    existing_ids: dict[str, int] = {}
    if existing_catalog_path is not None and existing_catalog_path.exists():
        existing = pd.read_csv(existing_catalog_path)
        if {"station_code", "id"}.issubset(existing.columns):
            existing_ids = {
                str(row["station_code"]): int(row["id"])
                for _, row in existing.loc[:, ["station_code", "id"]].dropna().iterrows()
            }

    next_id = max(existing_ids.values(), default=0) + 1
    assigned_ids: list[int] = []
    for station_code in df["station_code"]:
        code = str(station_code)
        station_id = existing_ids.get(code)
        if station_id is None:
            station_id = next_id
            existing_ids[code] = station_id
            next_id += 1
        assigned_ids.append(int(station_id))

    result = df.copy()
    result["id"] = assigned_ids
    return result


def _haversine_km(
    *,
    latitude: pd.Series,
    longitude: pd.Series,
    center_lat: float,
    center_lon: float,
) -> pd.Series:
    lat = np.deg2rad(latitude.to_numpy(dtype=float))
    lon = np.deg2rad(longitude.to_numpy(dtype=float))
    center_lat_rad = math.radians(center_lat)
    center_lon_rad = math.radians(center_lon)

    dlat = lat - center_lat_rad
    dlon = lon - center_lon_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(center_lat_rad) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return pd.Series(6371.0 * c, index=latitude.index, dtype="float64")


def _build_experiment_exclusion_reason(
    row: pd.Series,
    *,
    experiment_radius_km: float,
) -> str:
    reasons: list[str] = []
    if not bool(row["has_coordinates"]):
        reasons.append("missing_coordinates")
    if not bool(row["is_common_2018_2024"]):
        reasons.append("not_common_2018_2024")

    distance_km = row["distance_to_paris_center_km"]
    if pd.notna(distance_km) and float(distance_km) > experiment_radius_km:
        reasons.append("outside_15km_radius")

    if bool(row["is_excluded_station"]):
        reasons.append("excluded_station")

    return ";".join(reasons)


def build_airparif_station_catalog(
    *,
    raw_dir: Path | None = None,
    coords_path: Path | None = None,
    output_path: Path | None = None,
    existing_catalog_path: Path | None = None,
    common_years: Sequence[int] = DEFAULT_COMMON_YEARS,
    experiment_radius_km: float = DEFAULT_EXPERIMENT_RADIUS_KM,
    excluded_station_codes: Sequence[str] = DEFAULT_EXCLUDED_STATION_CODES,
    paris_center_lat: float = DEFAULT_PARIS_CENTER_LAT,
    paris_center_lon: float = DEFAULT_PARIS_CENTER_LON,
    metric_crs: str = DEFAULT_METRIC_CRS,
) -> pd.DataFrame:
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    coords_path = coords_path or DEFAULT_COORDS_PATH
    output_path = output_path or DEFAULT_CATALOG_PATH
    existing_catalog_path = existing_catalog_path or output_path

    inventory = _load_inventory(raw_dir)
    coords = _load_coordinates(coords_path)
    transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)

    common_years_set = {int(year) for year in common_years}
    excluded_codes = {str(code).strip().upper() for code in excluded_station_codes}

    grouped = (
        inventory.groupby("station_code", sort=True)
        .agg(
            station_name=("station_name", "first"),
            pollutant_code=("pollutant_code", "first"),
            pollutant_name=("pollutant_name", "first"),
            unit=("unit", "first"),
            years_present=("year", lambda values: tuple(sorted({int(v) for v in values}))),
        )
        .reset_index()
    )

    grouped["n_years_present"] = grouped["years_present"].apply(len).astype(int)
    grouped["first_year_present"] = grouped["years_present"].apply(lambda years: int(years[0]))
    grouped["last_year_present"] = grouped["years_present"].apply(lambda years: int(years[-1]))
    grouped["years_present"] = grouped["years_present"].apply(_years_to_text)
    grouped["is_common_2018_2024"] = grouped["years_present"].apply(
        lambda years_text: common_years_set.issubset({int(year) for year in years_text.split(",") if year})
    )

    catalog = grouped.merge(coords, on="station_code", how="left")
    catalog["station_name"] = catalog["station_name"].where(
        catalog["station_name"].astype(str).str.len() > 0,
        catalog["coords_station_name"],
    )
    catalog["name"] = catalog["station_name"]
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
    catalog["distance_to_paris_center_km"] = np.nan
    if coords_mask.any():
        catalog.loc[coords_mask, "distance_to_paris_center_km"] = _haversine_km(
            latitude=catalog.loc[coords_mask, "latitude"],
            longitude=catalog.loc[coords_mask, "longitude"],
            center_lat=paris_center_lat,
            center_lon=paris_center_lon,
        )

    catalog["within_15km_paris_center"] = catalog["distance_to_paris_center_km"].le(experiment_radius_km).fillna(False)
    catalog["is_excluded_station"] = catalog["station_code"].str.upper().isin(excluded_codes)
    catalog["use_for_experiments"] = (
        catalog["has_coordinates"]
        & catalog["is_common_2018_2024"]
        & catalog["within_15km_paris_center"]
        & ~catalog["is_excluded_station"]
    )
    catalog["experiment_exclusion_reason"] = catalog.apply(
        _build_experiment_exclusion_reason,
        axis=1,
        experiment_radius_km=experiment_radius_km,
    )

    catalog = _assign_stable_ids(catalog.sort_values("station_code").reset_index(drop=True), existing_catalog_path)
    catalog = catalog.sort_values("id").reset_index(drop=True)
    catalog = catalog[
        [
            "id",
            "station_code",
            "station_name",
            "name",
            "pollutant_code",
            "pollutant_name",
            "unit",
            "latitude",
            "longitude",
            "utm_x",
            "utm_y",
            "metric_crs",
            "source",
            "years_present",
            "n_years_present",
            "first_year_present",
            "last_year_present",
            "is_common_2018_2024",
            "has_coordinates",
            "distance_to_paris_center_km",
            "within_15km_paris_center",
            "is_excluded_station",
            "use_for_experiments",
            "experiment_exclusion_reason",
        ]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False, encoding="utf-8")
    return catalog


if __name__ == "__main__":
    catalog = build_airparif_station_catalog()
    selected = catalog[catalog["use_for_experiments"]]
    print(f"Wrote {len(catalog)} AIRPARIF stations to {DEFAULT_CATALOG_PATH}")
    print(f"Selected for experiments: {len(selected)} station(s)")
