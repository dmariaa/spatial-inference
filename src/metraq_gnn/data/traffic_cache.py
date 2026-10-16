from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from sqlalchemy import bindparam, text

from metraq_dip.data.aq_backends import AQBackend
from metraq_dip.tools.grid import map_sensor_ids_to_grid, prepare_grid_context
from metraq_gnn.data.cache import AQSensorCache


_PARTITION_PATTERN = re.compile(r"p\d{6}")


@dataclass(frozen=True)
class TrafficGridCache:
    """Read-only, memory-mapped hourly traffic intensity over the AQ grid."""

    path: Path
    metadata: dict
    timestamps_ns: np.ndarray
    values: np.ndarray
    availability: np.ndarray

    @property
    def time_index(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.timestamps_ns, unit="ns")

    @property
    def grid_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.metadata["grid_shape"])


def load_traffic_grid_cache(path: str | Path) -> TrafficGridCache:
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return TrafficGridCache(
        path=path,
        metadata=metadata,
        timestamps_ns=np.load(path / "timestamps.npy", mmap_mode="r"),
        values=np.load(path / "traffic_intensity.npy", mmap_mode="r"),
        availability=np.load(path / "traffic_availability.npy", mmap_mode="r"),
    )


def compute_traffic_training_normalization(
    cache: TrafficGridCache,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[float, float]:
    """Calculate traffic mean/std from available values in the TRAIN interval."""
    timestamps = cache.time_index
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    first = int(timestamps.searchsorted(start_ts))
    last = int(timestamps.searchsorted(end_ts, side="right"))
    if first >= last:
        raise ValueError("traffic normalization interval contains no cached timestamps")
    values = np.asarray(cache.values[first:last], dtype=np.float64)
    available = np.asarray(cache.availability[first:last], dtype=bool)
    selected = values[available]
    if selected.size == 0:
        raise ValueError("no traffic observations available for TRAIN normalization")
    mean = float(selected.mean())
    std = float(selected.std())
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return mean, std


def month_ranges(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Return disjoint monthly [start, end) ranges and their partition names."""
    first = pd.Timestamp(start)
    last_inclusive = pd.Timestamp(end)
    if first.floor("h") != first or last_inclusive.floor("h") != last_inclusive:
        raise ValueError("start and end must be hour-aligned")
    if first > last_inclusive:
        raise ValueError("start must not be after end")
    stop = last_inclusive + pd.Timedelta(hours=1)
    current = first
    result = []
    while current < stop:
        following_month = pd.Timestamp(
            year=current.year + (current.month == 12),
            month=current.month % 12 + 1,
            day=1,
        )
        chunk_end = min(stop, following_month)
        result.append((current, chunk_end, f"p{current.year}{current.month:02d}"))
        current = chunk_end
    return result


def build_traffic_grid_cache(
    *,
    path: str | Path,
    aq_cache: AQSensorCache,
    aq_backend: AQBackend,
    connection,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    sensor_batch_size: int = 64,
    statement_timeout_seconds: int = 120,
    progress_callback: Callable[[str, int, int, float], None] | None = None,
) -> TrafficGridCache:
    """Build or resume traffic intensity using bounded monthly partition queries."""
    if sensor_batch_size <= 0:
        raise ValueError("sensor_batch_size must be positive")
    if statement_timeout_seconds <= 0:
        raise ValueError("statement_timeout_seconds must be positive")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    ranges = month_ranges(start_ts, end_ts)
    aq_times = aq_cache.time_index
    if start_ts < aq_times[0] or end_ts > aq_times[-1]:
        raise ValueError("traffic period must be contained in the AQ cache period")

    grid_ctx = _rebuild_and_validate_grid(aq_cache, aq_backend)
    traffic_sensors = pd.read_sql_query(
        text("SELECT id, utm_x, utm_y FROM traffic_sensors"),
        con=connection,
    )
    traffic_sensors = traffic_sensors.dropna(subset=["utm_x", "utm_y"]).sort_values("id")
    traffic_ids = traffic_sensors["id"].to_numpy(dtype=np.int64)
    rows, cols, mapped = map_sensor_ids_to_grid(
        grid_ctx,
        traffic_sensors,
        traffic_ids,
        warn_prefix="Traffic cache",
    )
    mapped_ids = traffic_ids[mapped]
    mapped_nodes = rows[mapped] * aq_cache.grid_shape[1] + cols[mapped]
    if not mapped_ids.size:
        raise ValueError("no traffic sensors map to the AQ grid")
    sensor_nodes = {
        int(sensor_id): int(node)
        for sensor_id, node in zip(mapped_ids, mapped_nodes, strict=True)
    }

    missing = [partition for _, _, partition in ranges if not _partition_exists(connection, partition)]
    if missing:
        raise ValueError(f"missing traffic partitions: {missing}")

    path = Path(path)
    expected = {
        "version": 1,
        "source_table": "traffic_data",
        "value": "traffic_intensity",
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "grid_shape": list(aq_cache.grid_shape),
        "aq_cache": str(aq_cache.path.resolve()),
        "sensor_batch_size": int(sensor_batch_size),
        "aggregation": "mean within sensor/hour, then equal-weight mean across sensors in each grid cell",
        "query_scope": "explicit monthly partition and mapped traffic sensor IDs only",
    }
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"existing traffic cache has incompatible {key!r}")
    else:
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise ValueError("traffic cache directory exists but has no valid metadata.json")
        timestamps = pd.date_range(start_ts, end_ts, freq="h").as_unit("ns")
        np.save(path / "timestamps.npy", timestamps.asi8)
        values = open_memmap(
            path / "traffic_intensity.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(timestamps), *aq_cache.grid_shape),
        )
        values[:] = 0.0
        values.flush()
        availability = open_memmap(
            path / "traffic_availability.npy",
            mode="w+",
            dtype=np.bool_,
            shape=values.shape,
        )
        availability[:] = False
        availability.flush()
        metadata = expected | {
            "traffic_sensors_total": int(len(traffic_ids)),
            "traffic_sensors_mapped": int(len(mapped_ids)),
            "completed_months": [],
            "monthly_stats": {},
        }
        _write_metadata(metadata_path, metadata)

    cache = load_traffic_grid_cache(path)
    values = np.load(path / "traffic_intensity.npy", mmap_mode="r+")
    availability = np.load(path / "traffic_availability.npy", mmap_mode="r+")
    completed = set(metadata["completed_months"])
    connection.execute(text(f"SET SESSION max_statement_time={int(statement_timeout_seconds)}"))
    connection.rollback()

    total_months = len(ranges)
    for month_number, (chunk_start, chunk_end, partition) in enumerate(ranges, 1):
        month_key = chunk_start.strftime("%Y-%m")
        if month_key in completed:
            continue
        started = time.monotonic()
        first = int(cache.time_index.searchsorted(chunk_start))
        last = int(cache.time_index.searchsorted(chunk_end))
        sums = np.zeros((last - first, int(np.prod(cache.grid_shape))), dtype=np.float64)
        counts = np.zeros_like(sums, dtype=np.int32)
        connection.execute(text("START TRANSACTION READ ONLY"))
        try:
            for offset in range(0, len(mapped_ids), sensor_batch_size):
                batch_ids = mapped_ids[offset : offset + sensor_batch_size]
                frame = _query_partition_batch(
                    connection,
                    partition=partition,
                    start=chunk_start,
                    end=chunk_end,
                    sensor_ids=batch_ids,
                )
                _accumulate_frame(
                    frame=frame,
                    chunk_start=chunk_start,
                    sensor_nodes=sensor_nodes,
                    sums=sums,
                    counts=counts,
                )
        finally:
            connection.rollback()
        observed = counts > 0
        month_values = np.divide(sums, counts, out=np.zeros_like(sums), where=observed)
        values[first:last] = month_values.reshape(last - first, *cache.grid_shape)
        availability[first:last] = observed.reshape(last - first, *cache.grid_shape)
        values.flush()
        availability.flush()
        completed.add(month_key)
        metadata["completed_months"] = sorted(completed)
        elapsed_seconds = time.monotonic() - started
        metadata.setdefault("monthly_stats", {})[month_key] = {
            "available_cells": int(observed.sum()),
            "elapsed_seconds": round(elapsed_seconds, 3),
        }
        _write_metadata(metadata_path, metadata)
        if progress_callback is not None:
            progress_callback(month_key, month_number, total_months, elapsed_seconds)

    return load_traffic_grid_cache(path)


def _rebuild_and_validate_grid(aq_cache: AQSensorCache, aq_backend: AQBackend) -> dict:
    magnitudes = [int(value) for value in aq_cache.magnitude_ids]
    sensors = aq_backend.get_sensors(magnitudes=magnitudes).sort_values("id")
    grid_ctx = prepare_grid_context(
        sensors,
        cell_size_m=int(aq_cache.metadata["cell_size_m"]),
        margin_m_x=int(aq_cache.metadata["margin_m_x"]),
        margin_m_y=int(aq_cache.metadata["margin_m_y"]),
    )
    if tuple(grid_ctx["grid"].shape) != aq_cache.grid_shape:
        raise ValueError("reconstructed grid shape does not match AQ cache")
    sensor_ids = sensors["id"].to_numpy(dtype=np.int64)
    rows, cols, mapped = map_sensor_ids_to_grid(grid_ctx, sensors, sensor_ids)
    nodes = rows * aq_cache.grid_shape[1] + cols
    if (
        not mapped.all()
        or not np.array_equal(sensor_ids, np.asarray(aq_cache.sensor_ids))
        or not np.array_equal(nodes, np.asarray(aq_cache.sensor_node_indices))
    ):
        raise ValueError("reconstructed grid does not reproduce AQ sensor mapping")
    return grid_ctx


def _partition_exists(connection, partition: str) -> bool:
    if _PARTITION_PATTERN.fullmatch(partition) is None:
        raise ValueError(f"invalid partition name: {partition!r}")
    count = connection.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.partitions "
            "WHERE table_schema=DATABASE() AND table_name='traffic_data' "
            "AND partition_name=:partition"
        ),
        {"partition": partition},
    ).scalar()
    return bool(count)


def _query_partition_batch(
    connection,
    *,
    partition: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sensor_ids: np.ndarray,
) -> pd.DataFrame:
    if _PARTITION_PATTERN.fullmatch(partition) is None:
        raise ValueError(f"invalid partition name: {partition!r}")
    statement = text(
        f"""SELECT tr.sensor_id,
                   DATE_FORMAT(tr.entry_date, '%Y-%m-%d %H:00:00') AS hour,
                   AVG(tr.traffic_intensity) AS traffic_intensity
            FROM traffic_data PARTITION ({partition}) tr FORCE INDEX (PRIMARY)
            WHERE tr.sensor_id IN :sensor_ids
              AND tr.entry_date >= :start
              AND tr.entry_date < :end
              AND tr.traffic_intensity IS NOT NULL
            GROUP BY tr.sensor_id, hour"""
    ).bindparams(bindparam("sensor_ids", expanding=True))
    return pd.read_sql_query(
        statement,
        con=connection,
        params={"sensor_ids": [int(value) for value in sensor_ids], "start": start, "end": end},
        parse_dates=["hour"],
    )


def _accumulate_frame(
    *,
    frame: pd.DataFrame,
    chunk_start: pd.Timestamp,
    sensor_nodes: dict[int, int],
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    if frame.empty:
        return
    times = pd.DatetimeIndex(frame["hour"])
    time_positions = ((times - chunk_start) // pd.Timedelta(hours=1)).to_numpy(dtype=np.int64)
    nodes = frame["sensor_id"].map(sensor_nodes).to_numpy(dtype=np.int64)
    values = frame["traffic_intensity"].to_numpy(dtype=np.float64)
    valid = (
        (time_positions >= 0)
        & (time_positions < sums.shape[0])
        & np.isfinite(values)
    )
    np.add.at(sums, (time_positions[valid], nodes[valid]), values[valid])
    np.add.at(counts, (time_positions[valid], nodes[valid]), 1)


def _write_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "TrafficGridCache",
    "build_traffic_grid_cache",
    "compute_traffic_training_normalization",
    "load_traffic_grid_cache",
    "month_ranges",
]
