from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "AIRPARIF" / "raw"
DEFAULT_STATION_CATALOG_PATH = PROJECT_ROOT / "data" / "AIRPARIF" / "station_catalog.csv"
DEFAULT_MAGNITUDE_CATALOG_PATH = PROJECT_ROOT / "data" / "AIRPARIF" / "magnitude_catalog.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "AIRPARIF"


@dataclass(frozen=True)
class AirparifRawFile:
    year: int
    pollutant_code: str
    path: Path


def _iter_airparif_raw_files(raw_dir: Path) -> list[AirparifRawFile]:
    files: list[AirparifRawFile] = []
    for path in sorted(raw_dir.glob("*.csv")):
        if path.name == "common_no2_stations_2018_2024_coordinates.csv":
            continue

        match = re.match(r"(?P<year>\d{4})-(?P<pollutant>[A-Za-z0-9_-]+)\.csv$", path.name)
        if match is None:
            continue

        files.append(
            AirparifRawFile(
                year=int(match.group("year")),
                pollutant_code=match.group("pollutant").strip().upper(),
                path=path,
            )
        )

    return sorted(files, key=lambda item: (item.year, item.pollutant_code, item.path.name))


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


def _parse_header_inventory(raw_file: AirparifRawFile) -> pd.DataFrame:
    header_rows = _read_header_rows(raw_file.path)
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

        parsed_pollutant_code = pollutant_code.strip().upper() or raw_file.pollutant_code
        records.append(
            {
                "year": raw_file.year,
                "pollutant_code": parsed_pollutant_code,
                "station_code": code,
                "station_name": station_name.strip() or code,
                "magnitude_name": pollutant_name.strip() or parsed_pollutant_code,
                "unit": unit.strip(),
            }
        )

    if not records:
        raise ValueError(f"{raw_file.path} does not contain any station columns.")

    return pd.DataFrame.from_records(records)


def _load_station_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"AIRPARIF station catalog not found: {path}")

    catalog = pd.read_csv(path, encoding="utf-8")
    required = {"id", "station_code", "station_name", "latitude", "longitude", "utm_x", "utm_y"}
    missing = required.difference(catalog.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required station catalog columns: {missing_text}")

    return catalog


def _assign_stable_magnitude_ids(df: pd.DataFrame, existing_catalog_path: Path | None) -> pd.DataFrame:
    existing_ids: dict[str, int] = {}
    if existing_catalog_path is not None and existing_catalog_path.exists():
        existing = pd.read_csv(existing_catalog_path, encoding="utf-8")
        if {"magnitude_code", "id"}.issubset(existing.columns):
            existing_ids = {
                str(row["magnitude_code"]).strip().upper(): int(row["id"])
                for _, row in existing.loc[:, ["magnitude_code", "id"]].dropna().iterrows()
            }

    next_id = max(existing_ids.values(), default=0) + 1
    ids: list[int] = []
    for magnitude_code in df["magnitude_code"]:
        code = str(magnitude_code).strip().upper()
        magnitude_id = existing_ids.get(code)
        if magnitude_id is None:
            magnitude_id = next_id
            existing_ids[code] = magnitude_id
            next_id += 1
        ids.append(int(magnitude_id))

    result = df.copy()
    result["id"] = ids
    return result


def _years_to_text(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in sorted(set(values)))


def build_airparif_magnitude_catalog(
    *,
    raw_dir: Path | None = None,
    output_path: Path | None = None,
    existing_catalog_path: Path | None = None,
) -> pd.DataFrame:
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    output_path = output_path or DEFAULT_MAGNITUDE_CATALOG_PATH
    existing_catalog_path = existing_catalog_path or output_path

    raw_files = _iter_airparif_raw_files(raw_dir)
    if not raw_files:
        raise FileNotFoundError(
            f"No AIRPARIF raw pollutant files found in {raw_dir}. Expected files like 2018-NO2.csv."
        )

    inventory = pd.concat([_parse_header_inventory(raw_file) for raw_file in raw_files], ignore_index=True)
    grouped = (
        inventory.groupby("pollutant_code", sort=True)
        .agg(
            magnitude_name=("magnitude_name", "first"),
            unit=("unit", "first"),
            years_present=("year", lambda values: tuple(sorted({int(v) for v in values}))),
        )
        .reset_index()
        .rename(columns={"pollutant_code": "magnitude_code"})
    )
    grouped["source_dataset"] = "airparif"
    grouped["n_years_present"] = grouped["years_present"].apply(len).astype(int)
    grouped["years_present"] = grouped["years_present"].apply(_years_to_text)

    catalog = _assign_stable_magnitude_ids(grouped.sort_values("magnitude_code").reset_index(drop=True), existing_catalog_path)
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
        ]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False, encoding="utf-8")
    return catalog


def _read_airparif_measurements(raw_file: AirparifRawFile, header_inventory: pd.DataFrame) -> pd.DataFrame:
    station_codes = header_inventory["station_code"].tolist()
    column_names = ["entry_date"] + station_codes
    frame = pd.read_csv(
        raw_file.path,
        skiprows=6,
        header=None,
        names=column_names,
        encoding="utf-8",
    )
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], utc=True)
    frame = frame.dropna(subset=["entry_date"])
    return frame


def build_airparif_measurement_store(
    *,
    raw_dir: Path | None = None,
    station_catalog_path: Path | None = None,
    magnitude_catalog_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[int, Path]:
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    station_catalog_path = station_catalog_path or DEFAULT_STATION_CATALOG_PATH
    magnitude_catalog_path = magnitude_catalog_path or DEFAULT_MAGNITUDE_CATALOG_PATH
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    station_catalog = _load_station_catalog(station_catalog_path)
    magnitude_catalog = build_airparif_magnitude_catalog(
        raw_dir=raw_dir,
        output_path=magnitude_catalog_path,
        existing_catalog_path=magnitude_catalog_path,
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
    for raw_file in _iter_airparif_raw_files(raw_dir):
        header_inventory = _parse_header_inventory(raw_file)
        wide = _read_airparif_measurements(raw_file, header_inventory)
        long_frame = wide.melt(id_vars="entry_date", var_name="sensor_code", value_name="value")
        long_frame = long_frame.dropna(subset=["value"]).reset_index(drop=True)
        long_frame["sensor_code"] = long_frame["sensor_code"].astype(str)
        long_frame["magnitude_code"] = raw_file.pollutant_code
        long_frame["value"] = pd.to_numeric(long_frame["value"], errors="coerce")
        long_frame = long_frame.dropna(subset=["value"]).reset_index(drop=True)

        measurements = long_frame.merge(station_meta, on="sensor_code", how="left")
        missing_station_codes = measurements.loc[measurements["sensor_id"].isna(), "sensor_code"].dropna().unique().tolist()
        if missing_station_codes:
            missing_text = ", ".join(sorted(map(str, missing_station_codes)))
            raise ValueError(
                f"{raw_file.path} contains station codes missing from {station_catalog_path}: {missing_text}"
            )

        measurements = measurements.merge(magnitude_meta, on="magnitude_code", how="left")
        if measurements["magnitude_id"].isna().any():
            missing_codes = measurements.loc[measurements["magnitude_id"].isna(), "magnitude_code"].dropna().unique().tolist()
            missing_text = ", ".join(sorted(map(str, missing_codes)))
            raise ValueError(
                f"{raw_file.path} contains pollutant codes missing from {magnitude_catalog_path}: {missing_text}"
            )

        measurements["sensor_id"] = measurements["sensor_id"].astype("int64")
        measurements["magnitude_id"] = measurements["magnitude_id"].astype("int32")
        measurements["entry_date"] = pd.to_datetime(measurements["entry_date"], utc=True)

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

        frames_by_year.setdefault(raw_file.year, []).append(measurements)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: dict[int, Path] = {}
    for year, frames in sorted(frames_by_year.items()):
        year_frame = pd.concat(frames, ignore_index=True)
        year_frame["entry_date"] = year_frame["entry_date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        output_path = output_dir / f"airparif_aq-{year}.csv"
        year_frame.to_csv(output_path, index=False, encoding="utf-8")
        written_files[year] = output_path

    return written_files


if __name__ == "__main__":
    written = build_airparif_measurement_store()
    print(f"Wrote magnitude catalog to {DEFAULT_MAGNITUDE_CATALOG_PATH}")
    for year, path in written.items():
        print(f"{year}: {path}")
