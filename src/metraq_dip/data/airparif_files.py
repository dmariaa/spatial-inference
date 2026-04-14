from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from threading import Lock

import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.dataset as ds


class AirparifFiles:
    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        station_catalog_path: Path | None = None,
        magnitude_catalog_path: Path | None = None,
        use_for_experiments_only: bool = True,
    ):
        project_root = Path(__file__).resolve().parents[3]
        env_dir = os.getenv("AIRPARIF_DATA_DIR")
        self.data_dir = Path(env_dir) if env_dir else (data_dir or project_root / "data" / "AIRPARIF")
        self.station_catalog_path = Path(
            os.getenv("AIRPARIF_STATION_CATALOG") or station_catalog_path or (self.data_dir / "station_catalog.csv")
        )
        self.magnitude_catalog_path = Path(
            os.getenv("AIRPARIF_MAGNITUDE_CATALOG") or magnitude_catalog_path or (self.data_dir / "magnitude_catalog.csv")
        )
        self.use_for_experiments_only = bool(use_for_experiments_only)
        self._dataset_lock = Lock()
        self._dataset_cache: dict[tuple[str, ...], ds.Dataset] = {}
        self._sensor_catalog_cache: pd.DataFrame | None = None
        self._sensor_coverage_cache: pd.DataFrame | None = None
        self._magnitude_bounds_cache: dict[tuple[int, ...], dict[int, tuple[float, float]]] = {}

    @property
    def _csv_format(self) -> ds.CsvFileFormat:
        convert_options = csv.ConvertOptions(
            column_types={
                "sensor_id": pa.int64(),
                "sensor_code": pa.string(),
                "sensor_name": pa.string(),
                "latitude": pa.float64(),
                "longitude": pa.float64(),
                "utm_x": pa.float64(),
                "utm_y": pa.float64(),
                "magnitude_id": pa.int32(),
                "magnitude_code": pa.string(),
                "magnitude_name": pa.string(),
                "unit": pa.string(),
                "entry_date": pa.timestamp("s", tz="UTC"),
                "value": pa.float64(),
            }
        )
        return ds.CsvFileFormat(convert_options=convert_options)

    @property
    def _all_files(self) -> tuple[Path, ...]:
        files = sorted(self.data_dir.glob("airparif_aq-*.csv"))
        if not files:
            raise FileNotFoundError(
                f"No AIRPARIF CSV files found in {self.data_dir}. "
                "Expected files like airparif_aq-2024.csv"
            )
        return tuple(files)

    @property
    def _year_to_file(self) -> dict[int, Path]:
        mapping: dict[int, Path] = {}
        for path in self._all_files:
            match = re.search(r"airparif_aq-(\d{4})\.csv$", path.name)
            if match is None:
                continue
            mapping[int(match.group(1))] = path

        if not mapping:
            raise FileNotFoundError(
                f"No yearly AIRPARIF files found in {self.data_dir}. "
                "Expected files like airparif_aq-2024.csv"
            )
        return mapping

    def _resolve_files(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> tuple[Path, ...]:
        if start_date is None or end_date is None:
            return tuple(self._year_to_file[year] for year in sorted(self._year_to_file))

        years = range(start_date.year, end_date.year + 1)
        missing_years = [year for year in years if year not in self._year_to_file]
        if missing_years:
            missing = ", ".join(map(str, missing_years))
            available = ", ".join(map(str, sorted(self._year_to_file)))
            raise FileNotFoundError(
                f"Missing AIRPARIF CSV files for years: {missing}. "
                f"Available years: {available}"
            )

        return tuple(self._year_to_file[year] for year in years)

    def _dataset_for_files(self, files: tuple[Path, ...]) -> ds.Dataset:
        key = tuple(str(path.resolve()) for path in files)
        with self._dataset_lock:
            dataset = self._dataset_cache.get(key)
            if dataset is None:
                dataset = ds.dataset(list(key), format=self._csv_format)
                self._dataset_cache[key] = dataset
        return dataset

    @staticmethod
    def _normalize_magnitudes(magnitudes: list[int] | tuple[int, ...] | None) -> list[int] | None:
        if magnitudes is None:
            return None
        return [int(magnitude) for magnitude in magnitudes]

    @staticmethod
    def _normalize_sensors(sensors: list[int] | tuple[int, ...] | None) -> list[int] | None:
        if sensors is None:
            return None
        return [int(sensor_id) for sensor_id in sensors]

    def _build_filter(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        magnitudes: list[int] | None = None,
        sensors: list[int] | None = None,
    ):
        expr = None
        timestamp_type = pa.timestamp("s", tz="UTC")

        if start_date is not None:
            start_expr = ds.field("entry_date") >= pa.scalar(pd.Timestamp(start_date, tz="UTC").to_pydatetime(), type=timestamp_type)
            expr = start_expr if expr is None else (expr & start_expr)

        if end_date is not None:
            end_expr = ds.field("entry_date") <= pa.scalar(pd.Timestamp(end_date, tz="UTC").to_pydatetime(), type=timestamp_type)
            expr = end_expr if expr is None else (expr & end_expr)

        if magnitudes:
            magnitude_expr = ds.field("magnitude_id").isin(magnitudes)
            expr = magnitude_expr if expr is None else (expr & magnitude_expr)

        if sensors:
            sensor_expr = ds.field("sensor_id").isin(sensors)
            expr = sensor_expr if expr is None else (expr & sensor_expr)

        return expr

    def get_measurements(
        self,
        *,
        start_date: datetime,
        end_date: datetime,
        magnitudes: list[int],
    ) -> pd.DataFrame:
        files = self._resolve_files(start_date=start_date, end_date=end_date)
        dataset = self._dataset_for_files(files)
        magnitudes = self._normalize_magnitudes(magnitudes)
        table = dataset.to_table(
            columns=["sensor_id", "entry_date", "magnitude_id", "value"],
            filter=self._build_filter(start_date=start_date, end_date=end_date, magnitudes=magnitudes),
        )
        df = table.to_pandas()
        if df.empty:
            return pd.DataFrame(
                {
                    "sensor_id": pd.Series(dtype="int64"),
                    "entry_date": pd.Series(dtype="datetime64[ns]"),
                    "magnitude_id": pd.Series(dtype="int32"),
                    "value": pd.Series(dtype="float64"),
                }
            )

        df["sensor_id"] = df["sensor_id"].astype("int64")
        df["magnitude_id"] = df["magnitude_id"].astype("int32")
        df["entry_date"] = (
            pd.to_datetime(df["entry_date"], utc=True)
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )
        return df.sort_values(["sensor_id", "entry_date", "magnitude_id"]).reset_index(drop=True)

    def get_sensors(
        self,
        *,
        magnitudes: list[int] | None = None,
        sensors: list[int] | None = None,
    ) -> pd.DataFrame:
        magnitudes = self._normalize_magnitudes(magnitudes)
        sensors = self._normalize_sensors(sensors)
        df = self._get_sensor_catalog()

        if sensors:
            df = df[df["id"].isin(sensors)]
        elif self.use_for_experiments_only and "use_for_experiments" in df.columns:
            df = df[df["use_for_experiments"].astype(bool)]

        if magnitudes:
            coverage = self._get_sensor_coverage()
            allowed_ids = coverage[coverage["magnitude_id"].isin(magnitudes)]["sensor_id"].unique()
            df = df[df["id"].isin(allowed_ids)]

        return df.sort_values("id").reset_index(drop=True)

    def get_magnitude_bounds(self, magnitudes: list[int]) -> dict[int, tuple[float, float]]:
        magnitudes = self._normalize_magnitudes(magnitudes)
        key = tuple(sorted(magnitudes))
        if key in self._magnitude_bounds_cache:
            return dict(self._magnitude_bounds_cache[key])

        files = self._resolve_files()
        dataset = self._dataset_for_files(files)
        table = dataset.to_table(
            columns=["magnitude_id", "value"],
            filter=self._build_filter(magnitudes=magnitudes),
        )
        if table.num_rows == 0:
            return {}

        grouped = table.group_by("magnitude_id").aggregate([("value", "min"), ("value", "max")]).to_pandas()
        bounds = {
            int(row["magnitude_id"]): (float(row["value_min"]), float(row["value_max"]))
            for _, row in grouped.iterrows()
        }
        self._magnitude_bounds_cache[key] = bounds
        return dict(bounds)

    def _get_sensor_catalog(self) -> pd.DataFrame:
        if self._sensor_catalog_cache is not None:
            return self._sensor_catalog_cache.copy()

        if not self.station_catalog_path.exists():
            raise FileNotFoundError(f"AIRPARIF station catalog not found: {self.station_catalog_path}")

        catalog = pd.read_csv(self.station_catalog_path, encoding="utf-8")
        if "name" not in catalog.columns and "station_name" in catalog.columns:
            catalog["name"] = catalog["station_name"]
        catalog["id"] = catalog["id"].astype(int)
        self._sensor_catalog_cache = catalog
        return catalog.copy()

    def _get_sensor_coverage(self) -> pd.DataFrame:
        if self._sensor_coverage_cache is not None:
            return self._sensor_coverage_cache.copy()

        files = self._resolve_files()
        dataset = self._dataset_for_files(files)
        table = dataset.to_table(columns=["sensor_id", "magnitude_id"])
        coverage = (
            table.to_pandas()
            .drop_duplicates(subset=["sensor_id", "magnitude_id"])
            .sort_values(["sensor_id", "magnitude_id"])
            .reset_index(drop=True)
        )
        self._sensor_coverage_cache = coverage
        return coverage.copy()


airparif_files = AirparifFiles()
