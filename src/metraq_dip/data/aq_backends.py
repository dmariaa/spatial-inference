from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol

import pandas as pd
from sqlalchemy import text

from metraq_dip.data.airparif_files import airparif_files
from metraq_dip.data.metraq_db_legacy import metraq_db as metraq_db_legacy
from metraq_dip.data.metraq_files import metraq_files


class AQBackend(Protocol):
    dataset_name: str
    backend_name: str

    def get_sensors(
        self,
        *,
        magnitudes: list[int] | None = None,
        sensors: list[int] | None = None,
    ) -> pd.DataFrame:
        ...

    def get_measurements(
        self,
        *,
        start_date: datetime,
        end_date: datetime,
        magnitudes: list[int],
    ) -> pd.DataFrame:
        ...

    def get_magnitude_bounds(self, magnitudes: list[int]) -> dict[int, tuple[float, float]]:
        ...


class MetraqFilesBackend:
    dataset_name = "metraq"
    backend_name = "files"

    def get_sensors(
        self,
        *,
        magnitudes: list[int] | None = None,
        sensors: list[int] | None = None,
    ) -> pd.DataFrame:
        return metraq_files.get_sensors(magnitudes=magnitudes, sensors=sensors)

    def get_measurements(
        self,
        *,
        start_date: datetime,
        end_date: datetime,
        magnitudes: list[int],
    ) -> pd.DataFrame:
        return metraq_files.get_measurements(
            start_date=start_date,
            end_date=end_date,
            magnitudes=magnitudes,
        )

    def get_magnitude_bounds(self, magnitudes: list[int]) -> dict[int, tuple[float, float]]:
        return metraq_files.get_magnitude_bounds(magnitudes)


class MetraqDbBackend:
    dataset_name = "metraq"
    backend_name = "db"

    def get_sensors(
        self,
        *,
        magnitudes: list[int] | None = None,
        sensors: list[int] | None = None,
    ) -> pd.DataFrame:
        query = [
            "SELECT id, name, utm_x, utm_y, latitude, longitude",
            "FROM merged_sensors",
            "WHERE 1 = 1",
        ]

        if magnitudes:
            magnitude_sql = ",".join(map(str, magnitudes))
            query.append(
                "AND id IN (SELECT DISTINCT sensor_id "
                f"FROM MAD_merged_aq_data WHERE magnitude_id IN ({magnitude_sql}))"
            )

        if sensors:
            sensor_sql = ",".join(map(str, sensors))
            query.append(f"AND id IN ({sensor_sql})")

        df = pd.read_sql_query(text("\n".join(query)), con=metraq_db_legacy.connection)
        if df.empty:
            return pd.DataFrame(
                {
                    "id": pd.Series(dtype="int64"),
                    "name": pd.Series(dtype="object"),
                    "utm_x": pd.Series(dtype="float64"),
                    "utm_y": pd.Series(dtype="float64"),
                    "latitude": pd.Series(dtype="float64"),
                    "longitude": pd.Series(dtype="float64"),
                }
            )

        df["id"] = df["id"].astype(int)
        return df.sort_values("id").reset_index(drop=True)

    def get_measurements(
        self,
        *,
        start_date: datetime,
        end_date: datetime,
        magnitudes: list[int],
    ) -> pd.DataFrame:
        magnitude_sql = ",".join(map(str, magnitudes))
        query = """
                SELECT md.sensor_id,
                       md.entry_date,
                       md.magnitude_id,
                       md.value
                FROM MAD_merged_aq_data md
                WHERE is_valid
                  AND entry_date >= :start_date
                  AND entry_date <= :end_date
        """
        query += f" AND magnitude_id IN ({magnitude_sql})"
        params = {"start_date": start_date, "end_date": end_date}
        df = pd.read_sql_query(
            text(query),
            con=metraq_db_legacy.connection,
            params=params,
            parse_dates=["entry_date"],
        )
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
        return df.sort_values(["sensor_id", "entry_date", "magnitude_id"]).reset_index(drop=True)

    def get_magnitude_bounds(self, magnitudes: list[int]) -> dict[int, tuple[float, float]]:
        magnitude_sql = ",".join(map(str, magnitudes))
        query = (
            "SELECT id, min_value, max_value "
            f"FROM aq_magnitudes WHERE id IN ({magnitude_sql})"
        )
        rows = metraq_db_legacy.execute(query)
        return {int(row[0]): (float(row[1]), float(row[2])) for row in rows}


class AirparifFilesBackend:
    dataset_name = "airparif"
    backend_name = "files"

    def get_sensors(
        self,
        *,
        magnitudes: list[int] | None = None,
        sensors: list[int] | None = None,
    ) -> pd.DataFrame:
        return airparif_files.get_sensors(magnitudes=magnitudes, sensors=sensors)

    def get_measurements(
        self,
        *,
        start_date: datetime,
        end_date: datetime,
        magnitudes: list[int],
    ) -> pd.DataFrame:
        return airparif_files.get_measurements(
            start_date=start_date,
            end_date=end_date,
            magnitudes=magnitudes,
        )

    def get_magnitude_bounds(self, magnitudes: list[int]) -> dict[int, tuple[float, float]]:
        return airparif_files.get_magnitude_bounds(magnitudes)


_BACKENDS: dict[tuple[str, str], AQBackend] = {
    ("airparif", "files"): AirparifFilesBackend(),
    ("metraq", "files"): MetraqFilesBackend(),
    ("metraq", "db"): MetraqDbBackend(),
}


def register_aq_backend(*, dataset: str, backend: str, implementation: AQBackend) -> None:
    _BACKENDS[(dataset.strip().lower(), backend.strip().lower())] = implementation


def get_aq_backend(
    *,
    dataset: str,
    backend: str,
) -> AQBackend:
    dataset_name = dataset.strip().lower()
    if not dataset_name:
        raise ValueError("dataset must be a non-empty string.")

    backend_name = backend.strip().lower()
    if not backend_name:
        raise ValueError("backend must be a non-empty string.")

    key = (dataset_name, backend_name)
    if key not in _BACKENDS:
        available = ", ".join(f"{dataset}:{backend_name}" for dataset, backend_name in sorted(_BACKENDS))
        raise ValueError(
            f"Unsupported AQ backend '{dataset_name}:{backend_name}'. "
            f"Available backends: {available}"
        )

    return _BACKENDS[key]


def _config_value(config: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def get_aq_backend_for_config(config: Mapping[str, Any] | Any) -> AQBackend:
    dataset = _config_value(config, "aq_dataset")
    backend = _config_value(config, "aq_backend")
    if dataset is None:
        raise ValueError("config must define aq_dataset.")
    if backend is None:
        raise ValueError("config must define aq_backend.")
    return get_aq_backend(dataset=str(dataset), backend=str(backend))
