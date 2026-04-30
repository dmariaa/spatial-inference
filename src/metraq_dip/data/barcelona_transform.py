from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "BARCELONA" / "RAW"
DEFAULT_STATION_CATALOG_PATH = PROJECT_ROOT / "data" / "BARCELONA" / "station_catalog.csv"
DEFAULT_POLLUTANT_CATALOG_PATH = PROJECT_ROOT / "data" / "BARCELONA" / "qualitat_aire_contaminants.csv"
DEFAULT_MAGNITUDE_CATALOG_PATH = PROJECT_ROOT / "data" / "BARCELONA" / "magnitude_catalog.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "BARCELONA"


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


def _years_to_text(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in sorted(set(values)))


def _normalize_unit(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    fixed = (
        text.replace("Âµg/mÂ³", "µg/m³")
        .replace("Âµg/m³", "µg/m³")
        .replace("mg/mÂ³", "mg/m³")
    )
    normalized = fixed.replace("µg/m³", "microg/m3").replace("mg/m³", "mg/m3")
    return normalized.strip()


def _load_official_pollutant_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Barcelona pollutant catalog not found: {path}. "
            "Download the official 'qualitat_aire_contaminants.csv' file first."
        )

    catalog = pd.read_csv(path, encoding="utf-8")
    required = {"Codi_Contaminant", "Desc_Contaminant", "Unitats"}
    missing = required.difference(catalog.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required pollutant catalog columns: {missing_text}")

    catalog = catalog.rename(
        columns={
            "Codi_Contaminant": "magnitude_code",
            "Desc_Contaminant": "magnitude_name",
            "Unitats": "unit",
        }
    ).copy()
    catalog["magnitude_code"] = catalog["magnitude_code"].astype("Int64").astype(str)
    catalog["magnitude_name"] = catalog["magnitude_name"].astype(str).str.strip()
    catalog["unit"] = catalog["unit"].apply(_normalize_unit)
    catalog["metadata_source"] = "official_catalog"
    return catalog[["magnitude_code", "magnitude_name", "unit", "metadata_source"]]


def _load_station_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Barcelona station catalog not found: {path}")

    catalog = pd.read_csv(path, encoding="utf-8")
    required = {"id", "station_code", "station_name", "latitude", "longitude", "utm_x", "utm_y"}
    missing = required.difference(catalog.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required station catalog columns: {missing_text}")

    catalog["station_code"] = catalog["station_code"].astype(str)
    return catalog


def _build_observed_magnitude_inventory(raw_dir: Path) -> pd.DataFrame:
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
            usecols=["CODI_CONTAMINANT", "ANY", "ESTACIO"],
        ).rename(
            columns={
                "CODI_CONTAMINANT": "magnitude_code",
                "ANY": "year",
                "ESTACIO": "station_code",
            }
        )
        frame["magnitude_code"] = frame["magnitude_code"].astype("Int64").astype(str)
        frame["station_code"] = frame["station_code"].astype("Int64").astype(str)
        frames.append(frame)

    inventory = pd.concat(frames, ignore_index=True)
    grouped = (
        inventory.groupby("magnitude_code", sort=True)
        .agg(
            years_present=("year", lambda values: tuple(sorted({int(value) for value in values}))),
            station_codes=("station_code", lambda values: ",".join(sorted({str(value) for value in values}))),
        )
        .reset_index()
    )
    grouped["n_years_present"] = grouped["years_present"].apply(len).astype(int)
    grouped["years_present"] = grouped["years_present"].apply(_years_to_text)
    grouped["n_stations_present"] = grouped["station_codes"].apply(
        lambda value: len([item for item in str(value).split(",") if item])
    ).astype(int)
    return grouped


def _assign_stable_magnitude_ids(df: pd.DataFrame, existing_catalog_path: Path | None) -> pd.DataFrame:
    existing_ids: dict[str, int] = {}
    if existing_catalog_path is not None and existing_catalog_path.exists():
        existing = pd.read_csv(existing_catalog_path, encoding="utf-8")
        if {"magnitude_code", "id"}.issubset(existing.columns):
            existing_ids = {
                str(row["magnitude_code"]).strip(): int(row["id"])
                for _, row in existing.loc[:, ["magnitude_code", "id"]].dropna().iterrows()
            }

    next_id = max(existing_ids.values(), default=0) + 1
    assigned_ids: list[int] = []
    for magnitude_code in df["magnitude_code"]:
        code = str(magnitude_code).strip()
        magnitude_id = existing_ids.get(code)
        if magnitude_id is None:
            magnitude_id = next_id
            existing_ids[code] = magnitude_id
            next_id += 1
        assigned_ids.append(int(magnitude_id))

    result = df.copy()
    result["id"] = assigned_ids
    return result


def build_barcelona_magnitude_catalog(
    *,
    raw_dir: Path | None = None,
    pollutant_catalog_path: Path | None = None,
    output_path: Path | None = None,
    existing_catalog_path: Path | None = None,
    strict_missing_metadata: bool = False,
) -> pd.DataFrame:
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    pollutant_catalog_path = pollutant_catalog_path or DEFAULT_POLLUTANT_CATALOG_PATH
    output_path = output_path or DEFAULT_MAGNITUDE_CATALOG_PATH
    existing_catalog_path = existing_catalog_path or output_path

    observed = _build_observed_magnitude_inventory(raw_dir)
    official = _load_official_pollutant_catalog(pollutant_catalog_path)
    catalog = observed.merge(official, on="magnitude_code", how="left")
    missing_codes = sorted(catalog.loc[catalog["magnitude_name"].isna(), "magnitude_code"].unique().tolist())
    if missing_codes and strict_missing_metadata:
        missing_text = ", ".join(missing_codes)
        raise ValueError(
            f"Observed Barcelona pollutant codes are missing from {pollutant_catalog_path}: {missing_text}"
        )

    catalog["magnitude_name"] = catalog["magnitude_name"].where(
        catalog["magnitude_name"].astype("string").str.len().fillna(0) > 0,
        catalog["magnitude_code"].map(lambda code: f"unknown_code_{code}"),
    )
    catalog["unit"] = catalog["unit"].fillna("")
    catalog["metadata_source"] = catalog["metadata_source"].fillna("observed_only")
    catalog["is_missing_official_metadata"] = catalog["metadata_source"].eq("observed_only")
    catalog["source_dataset"] = "barcelona"

    catalog = _assign_stable_magnitude_ids(
        catalog.sort_values("magnitude_code").reset_index(drop=True),
        existing_catalog_path,
    )
    catalog = catalog.sort_values("id").reset_index(drop=True)
    catalog = catalog[
        [
            "id",
            "magnitude_code",
            "magnitude_name",
            "unit",
            "source_dataset",
            "years_present",
            "n_years_present",
            "station_codes",
            "n_stations_present",
            "metadata_source",
            "is_missing_official_metadata",
        ]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False, encoding="utf-8")
    return catalog


def _read_barcelona_measurements(raw_file: BarcelonaRawFile) -> pd.DataFrame:
    frame = pd.read_csv(raw_file.path, encoding="utf-8")
    base = frame[["ESTACIO", "CODI_CONTAMINANT", "ANY", "MES", "DIA"]].copy()
    base["sensor_code"] = base["ESTACIO"].astype("Int64").astype(str)
    base["magnitude_code"] = base["CODI_CONTAMINANT"].astype("Int64").astype(str)
    dates = pd.to_datetime(
        {
            "year": base["ANY"].astype(int),
            "month": base["MES"].astype(int),
            "day": base["DIA"].astype(int),
        }
    )

    parts: list[pd.DataFrame] = []
    for hour in range(1, 25):
        value_column = f"H{hour:02d}"
        flag_column = f"V{hour:02d}"

        part = base[["sensor_code", "magnitude_code"]].copy()
        part["entry_date"] = dates + pd.to_timedelta(hour, unit="h")
        part["value"] = pd.to_numeric(frame[value_column], errors="coerce")
        part["flag"] = frame[flag_column].astype(str).str.strip().str.upper()
        part = part[part["flag"] == "V"].drop(columns=["flag"])
        part = part.dropna(subset=["value"]).reset_index(drop=True)
        parts.append(part)

    measurements = pd.concat(parts, ignore_index=True)
    measurements["entry_date"] = pd.to_datetime(measurements["entry_date"])
    measurements["value"] = measurements["value"].astype("float64")
    return measurements.sort_values(["entry_date", "sensor_code", "magnitude_code"]).reset_index(drop=True)


def build_barcelona_measurement_store(
    *,
    raw_dir: Path | None = None,
    station_catalog_path: Path | None = None,
    pollutant_catalog_path: Path | None = None,
    magnitude_catalog_path: Path | None = None,
    output_dir: Path | None = None,
    strict_missing_metadata: bool = False,
) -> dict[int, Path]:
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    station_catalog_path = station_catalog_path or DEFAULT_STATION_CATALOG_PATH
    pollutant_catalog_path = pollutant_catalog_path or DEFAULT_POLLUTANT_CATALOG_PATH
    magnitude_catalog_path = magnitude_catalog_path or DEFAULT_MAGNITUDE_CATALOG_PATH
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    station_catalog = _load_station_catalog(station_catalog_path)
    magnitude_catalog = build_barcelona_magnitude_catalog(
        raw_dir=raw_dir,
        pollutant_catalog_path=pollutant_catalog_path,
        output_path=magnitude_catalog_path,
        existing_catalog_path=magnitude_catalog_path,
        strict_missing_metadata=strict_missing_metadata,
    )

    station_meta = station_catalog.rename(
        columns={
            "id": "sensor_id",
            "station_code": "sensor_code",
            "station_name": "sensor_name",
        }
    )[
        ["sensor_id", "sensor_code", "sensor_name", "latitude", "longitude", "utm_x", "utm_y"]
    ]
    magnitude_meta = magnitude_catalog.rename(columns={"id": "magnitude_id"})

    frames_by_year: dict[int, list[pd.DataFrame]] = {}
    for raw_file in _iter_barcelona_raw_files(raw_dir):
        long_frame = _read_barcelona_measurements(raw_file)

        measurements = long_frame.merge(station_meta, on="sensor_code", how="left")
        missing_station_codes = (
            measurements.loc[measurements["sensor_id"].isna(), "sensor_code"].dropna().unique().tolist()
        )
        if missing_station_codes:
            missing_text = ", ".join(sorted(map(str, missing_station_codes)))
            raise ValueError(
                f"{raw_file.path} contains station codes missing from {station_catalog_path}: {missing_text}"
            )

        measurements = measurements.merge(magnitude_meta, on="magnitude_code", how="left")
        if measurements["magnitude_id"].isna().any():
            missing_codes = (
                measurements.loc[measurements["magnitude_id"].isna(), "magnitude_code"].dropna().unique().tolist()
            )
            missing_text = ", ".join(sorted(map(str, missing_codes)))
            raise ValueError(
                f"{raw_file.path} contains pollutant codes missing from {magnitude_catalog_path}: {missing_text}"
            )

        measurements["sensor_id"] = measurements["sensor_id"].astype("int64")
        measurements["magnitude_id"] = measurements["magnitude_id"].astype("int32")
        measurements["entry_date"] = pd.to_datetime(measurements["entry_date"])

        measurements = measurements[
            [
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
        ].sort_values(["entry_date", "sensor_id", "magnitude_id"]).reset_index(drop=True)

        for year, year_frame in measurements.groupby(measurements["entry_date"].dt.year, sort=True):
            frames_by_year.setdefault(int(year), []).append(year_frame.reset_index(drop=True))

    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: dict[int, Path] = {}
    for year, frames in sorted(frames_by_year.items()):
        year_frame = pd.concat(frames, ignore_index=True)
        year_frame["entry_date"] = year_frame["entry_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        output_path = output_dir / f"barcelona_aq-{year}.csv"
        year_frame.to_csv(output_path, index=False, encoding="utf-8")
        written_files[year] = output_path

    return written_files


if __name__ == "__main__":
    written = build_barcelona_measurement_store()
    print(f"Wrote magnitude catalog to {DEFAULT_MAGNITUDE_CATALOG_PATH}")
    for year, path in written.items():
        print(f"{year}: {path}")
