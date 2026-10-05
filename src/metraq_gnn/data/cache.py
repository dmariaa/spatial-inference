from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from torch_geometric.data import Data

from metraq_dip.data.aq_backends import AQBackend
from metraq_dip.tools.grid import map_sensor_ids_to_grid, prepare_grid_context
from metraq_gnn.data.graph import build_grid_graph


@dataclass(frozen=True)
class AQSensorCache:
    """Read-only, memory-mapped AQ observations in sensor space."""

    path: Path
    metadata: dict
    timestamps_ns: np.ndarray
    sensor_ids: np.ndarray
    sensor_node_indices: np.ndarray
    magnitude_ids: np.ndarray
    values: np.ndarray
    availability: np.ndarray
    graph: Data

    @property
    def time_index(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.timestamps_ns)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.metadata["grid_shape"])


def load_aq_sensor_cache(path: str | Path) -> AQSensorCache:
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    graph_arrays = np.load(path / "graph.npz")
    graph = Data(
        edge_index=np_to_torch(graph_arrays["edge_index"], "long"),
        edge_attr=np_to_torch(graph_arrays["edge_attr"], "float"),
        pos=np_to_torch(graph_arrays["pos"], "float"),
        num_nodes=int(np.prod(metadata["grid_shape"])),
        grid_shape=tuple(metadata["grid_shape"]),
    )
    return AQSensorCache(
        path=path,
        metadata=metadata,
        timestamps_ns=np.load(path / "timestamps.npy", mmap_mode="r"),
        sensor_ids=np.load(path / "sensor_ids.npy", mmap_mode="r"),
        sensor_node_indices=np.load(path / "sensor_node_indices.npy", mmap_mode="r"),
        magnitude_ids=np.load(path / "magnitude_ids.npy", mmap_mode="r"),
        values=np.load(path / "aq_values.npy", mmap_mode="r"),
        availability=np.load(path / "aq_availability.npy", mmap_mode="r"),
        graph=graph,
    )


def np_to_torch(array: np.ndarray, kind: str):
    import torch

    dtype = torch.long if kind == "long" else torch.float32
    return torch.as_tensor(np.asarray(array), dtype=dtype)


def build_aq_sensor_cache(
    *,
    path: str | Path,
    aq_backend: AQBackend,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    magnitudes: list[int],
    cell_size_m: int = 1000,
    margin_m_x: int = 3000,
    margin_m_y: int = 2000,
) -> AQSensorCache:
    """Build or resume a raw hourly cache, committing one calendar year at a time."""
    path = Path(path)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts.floor("h") != start_ts or end_ts.floor("h") != end_ts or start_ts > end_ts:
        raise ValueError("start and end must be ordered, hour-aligned timestamps")
    if not magnitudes or len(set(magnitudes)) != len(magnitudes):
        raise ValueError("magnitudes must contain unique values")

    expected = {
        "version": 1,
        "dataset": aq_backend.dataset_name,
        "backend": aq_backend.backend_name,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "magnitudes": [int(value) for value in magnitudes],
        "cell_size_m": int(cell_size_m),
        "margin_m_x": int(margin_m_x),
        "margin_m_y": int(margin_m_y),
    }
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"existing cache has incompatible {key!r}")
    else:
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise ValueError("cache directory exists but has no valid metadata.json")
        sensors = aq_backend.get_sensors(magnitudes=magnitudes).sort_values("id")
        if sensors.empty:
            raise ValueError("backend returned no sensors for the requested magnitudes")
        sensor_ids = sensors["id"].to_numpy(dtype=np.int64)
        grid_ctx = prepare_grid_context(
            sensors,
            cell_size_m=cell_size_m,
            margin_m_x=margin_m_x,
            margin_m_y=margin_m_y,
        )
        rows, cols, mapped = map_sensor_ids_to_grid(grid_ctx, sensors, sensor_ids)
        if not mapped.all():
            raise ValueError("all cached sensors must map to the generated grid")
        grid_shape = tuple(int(value) for value in grid_ctx["grid"].shape)
        node_indices = rows * grid_shape[1] + cols
        timestamps = pd.date_range(start_ts, end_ts, freq="h")

        np.save(path / "timestamps.npy", timestamps.asi8)
        np.save(path / "sensor_ids.npy", sensor_ids)
        np.save(path / "sensor_node_indices.npy", node_indices.astype(np.int64))
        np.save(path / "magnitude_ids.npy", np.asarray(magnitudes, dtype=np.int32))
        values = open_memmap(
            path / "aq_values.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(timestamps), len(sensor_ids), len(magnitudes)),
        )
        values[:] = 0.0
        values.flush()
        availability = open_memmap(
            path / "aq_availability.npy",
            mode="w+",
            dtype=np.bool_,
            shape=values.shape,
        )
        availability[:] = False
        availability.flush()
        graph = build_grid_graph(grid_ctx)
        np.savez(
            path / "graph.npz",
            edge_index=graph.edge_index.numpy(),
            edge_attr=graph.edge_attr.numpy(),
            pos=graph.pos.numpy(),
        )
        metadata = expected | {"grid_shape": list(grid_shape), "completed_years": []}
        _write_metadata(metadata_path, metadata)

    cache = load_aq_sensor_cache(path)
    completed = set(int(year) for year in metadata["completed_years"])
    writable_values = np.load(path / "aq_values.npy", mmap_mode="r+")
    writable_availability = np.load(path / "aq_availability.npy", mmap_mode="r+")
    time_index = cache.time_index
    sensor_lookup = {int(value): index for index, value in enumerate(cache.sensor_ids)}
    magnitude_lookup = {int(value): index for index, value in enumerate(cache.magnitude_ids)}

    for year in range(start_ts.year, end_ts.year + 1):
        if year in completed:
            continue
        chunk_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end_ts, pd.Timestamp(year=year, month=12, day=31, hour=23))
        frame = aq_backend.get_measurements(
            start_date=chunk_start.to_pydatetime(),
            end_date=chunk_end.to_pydatetime(),
            magnitudes=magnitudes,
        )
        _write_measurement_chunk(
            frame=frame,
            time_index=time_index,
            values=writable_values,
            availability=writable_availability,
            sensor_lookup=sensor_lookup,
            magnitude_lookup=magnitude_lookup,
            start=chunk_start,
            end=chunk_end,
        )
        writable_values.flush()
        writable_availability.flush()
        completed.add(year)
        metadata["completed_years"] = sorted(completed)
        _write_metadata(metadata_path, metadata)

    return load_aq_sensor_cache(path)


def _write_measurement_chunk(
    *,
    frame: pd.DataFrame,
    time_index: pd.DatetimeIndex,
    values: np.ndarray,
    availability: np.ndarray,
    sensor_lookup: dict[int, int],
    magnitude_lookup: dict[int, int],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    first = int(time_index.searchsorted(start))
    last = int(time_index.searchsorted(end, side="right"))
    values[first:last] = 0.0
    availability[first:last] = False
    if frame.empty:
        return
    grouped = frame.groupby(["entry_date", "sensor_id", "magnitude_id"], as_index=False)["value"].mean()
    for row in grouped.itertuples(index=False):
        timestamp = pd.Timestamp(row.entry_date)
        time_pos = int(time_index.searchsorted(timestamp))
        sensor_pos = sensor_lookup.get(int(row.sensor_id))
        magnitude_pos = magnitude_lookup.get(int(row.magnitude_id))
        if first <= time_pos < last and sensor_pos is not None and magnitude_pos is not None:
            values[time_pos, sensor_pos, magnitude_pos] = float(row.value)
            availability[time_pos, sensor_pos, magnitude_pos] = True


def _write_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


__all__ = ["AQSensorCache", "build_aq_sensor_cache", "load_aq_sensor_cache"]
