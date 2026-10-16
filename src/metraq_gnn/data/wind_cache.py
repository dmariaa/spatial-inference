"""Compact physical wind vectors at AQ cells, built from admitted u/v exports."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metraq_dip.tools.interpolator import IdwInterpolator
from metraq_gnn.data.cache import AQSensorCache, rebuild_and_validate_grid


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aq_grid_identity(aq):
    """Include absolute-grid metadata, sensor mapping and graph orientation."""
    return dict(grid_shape=list(aq.grid_shape),
                cell_size_m=float(aq.metadata["cell_size_m"]),
                margin_m_x=aq.metadata.get("margin_m_x"),
                margin_m_y=aq.metadata.get("margin_m_y"),
                sensor_ids=np.asarray(aq.sensor_ids).tolist(),
                sensor_nodes=np.asarray(aq.sensor_node_indices).tolist(),
                positions_sha256=hashlib.sha256(
                    aq.graph.pos.cpu().numpy().astype("<f4").tobytes()).hexdigest())


@dataclass(frozen=True)
class WindSourceCache:
    path: Path
    metadata: dict
    timestamps_ns: np.ndarray
    node_indices: np.ndarray
    values: np.ndarray
    availability: np.ndarray

    def align(self, aq, timestamps):
        if self.metadata["aq_grid"] != aq_grid_identity(aq):
            raise ValueError("wind cache grid or sensor mapping differs from AQ cache")
        wanted = pd.DatetimeIndex(timestamps).as_unit("ns").asi8
        indices = np.searchsorted(self.timestamps_ns, wanted)
        if (indices >= len(self.timestamps_ns)).any() or not np.array_equal(
                np.asarray(self.timestamps_ns)[indices], wanted):
            raise ValueError("wind cache does not cover the requested AQ timestamps")
        return (np.asarray(self.values[indices], dtype=np.float32),
                np.asarray(self.availability[indices], dtype=bool))


def load_wind_source_cache(path):
    """Verify content identity before using a cache for training or evaluation."""
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("version") != 1 or meta.get("units") != "m/s":
        raise ValueError("unsupported wind cache format or units")
    fingerprint = meta.pop("fingerprint")
    expected = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    meta["fingerprint"] = fingerprint
    if fingerprint != expected:
        raise ValueError("wind cache metadata fingerprint differs")
    for filename, digest in meta["files"].items():
        if file_sha256(path / filename) != digest:
            raise ValueError(f"wind cache checksum differs: {filename}")
    times = np.load(path / "timestamps.npy", mmap_mode="r")
    nodes = np.load(path / "node_indices.npy", mmap_mode="r")
    values = np.load(path / "wind_uv.npy", mmap_mode="r")
    available = np.load(path / "wind_available.npy", mmap_mode="r")
    expected_nodes = np.unique(meta["aq_grid"]["sensor_nodes"])
    if not np.array_equal(nodes, expected_nodes):
        raise ValueError("wind cache source cells differ from AQ sensor cells")
    if (times.ndim != 1 or not len(times) or (np.diff(times) <= 0).any()
            or values.shape != (len(times), len(nodes), 2)
            or available.shape != values.shape[:2] or available.dtype != np.bool_):
        raise ValueError("invalid wind cache axes or availability mask")
    if not np.isfinite(values[available]).all():
        raise ValueError("available wind vectors must be finite")
    return WindSourceCache(path, meta, times, nodes, values, available)


def build_wind_source_cache(*, path, aq_cache: AQSensorCache, meteorology_dirs,
                            sensors, start=None, end=None, observed_only=False,
                            allow_missing_months=False, power=2., neighbours=8,
                            max_radius_m=None, progress_callback=None, mapping="station",
                            source_interpolation=None):
    """Map existing admitted 811/812 components directly to their AQ cells.

    The station catalogue contains id, utm_x and utm_y in EPSG:25830, the
    project's Madrid grid CRS. No AQ values, split labels or TEST data are read.
    Missing vectors remain unavailable; calm zero vectors remain available.
    """
    if mapping not in {"station", "idw"}:
        raise ValueError("wind mapping must be station or idw")
    path = Path(path)
    if path.exists():
        raise ValueError("wind cache output already exists")
    if not {"id", "utm_x", "utm_y"} <= set(sensors.columns):
        raise ValueError("catalogue needs id, utm_x and utm_y (EPSG:25830)")
    sensors = sensors.copy()
    if sensors["id"].isna().any() or sensors["id"].duplicated().any():
        raise ValueError("catalogue needs unique nonmissing station IDs")
    if not np.isfinite(sensors[["utm_x", "utm_y"]].to_numpy(dtype=float)).all():
        raise ValueError("station coordinates must be finite")
    # Validate IDW options even when all requested months are missing.
    IdwInterpolator.from_points([0.], [0.], [0.], power=power, k=neighbours,
                                max_radius_m=max_radius_m)
    ctx = rebuild_and_validate_grid(aq_cache, sensors=sensors)
    if ctx["metric_crs"] != "EPSG:25830":
        raise ValueError("wind station catalogue must use the AQ grid CRS EPSG:25830")
    nodes = np.unique(aq_cache.sensor_node_indices)
    centers = np.array([(ctx["grid"].flat[int(node)].centroid.x,
                         ctx["grid"].flat[int(node)].centroid.y) for node in nodes])
    times = aq_cache.time_index.as_unit("ns")
    first = pd.Timestamp(start) if start is not None else times[0]
    last = pd.Timestamp(end) if end is not None else times[-1]
    if first > last or first.floor("h") != first or last.floor("h") != last:
        raise ValueError("wind start/end must be ordered hour-aligned timestamps")
    times = times[(times >= first) & (times <= last)]
    if not len(times):
        raise ValueError("wind range has no AQ timestamps")
    months = times.to_period("M")
    sources, scopes = {}, []
    for directory in map(Path, meteorology_dirs):
        scope_path = directory / "scope.json"
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        units = {int(item["id"]): item["units"] for item in scope.get("catalog", [])}
        if units.get(81) != "m/s" or units.get(82) != "Grd":
            raise ValueError("admitted wind must have speed in m/s and direction in degrees")
        if not scope.get("wind") or scope.get("clipping") is not False:
            raise ValueError("use admitted meteorology exports with documented u/v conversion")
        scopes.append(dict(path=str(scope_path.resolve()), sha256=file_sha256(scope_path)))
        for month in months.unique():
            candidate = directory / f"meteo_{month}.parquet"
            if candidate.exists():
                if month in sources:
                    raise ValueError(f"duplicate meteorology month: {month}")
                sources[month] = candidate
    missing = [str(month) for month in months.unique() if month not in sources]
    if missing and not allow_missing_months:
        raise ValueError("missing meteorology months: " + ", ".join(missing)
                         + "; explicitly allow missing months to use isotropic fallback")
    values = np.zeros((len(times), len(nodes), 2), dtype=np.float32)
    available = np.zeros(values.shape[:2], dtype=bool)
    station_count = np.zeros(len(times), dtype=np.int32)
    observed_count = np.zeros(len(times), dtype=np.int32)
    input_files = []
    catalogue = sensors.set_index("id")
    for month, source in sources.items():
        frame = pd.read_parquet(source)
        required = {"sensor_id", "entry_date", "magnitude_id", "value", "is_interpolated", "derived"}
        if not required <= set(frame.columns):
            raise ValueError(f"incomplete admitted meteorology export: {source}")
        frame = frame[frame.magnitude_id.isin([811, 812])].copy()
        frame["entry_date"] = pd.to_datetime(frame.entry_date)
        if ((frame.entry_date.dt.to_period("M") != month).any()
                or (frame.entry_date != frame.entry_date.dt.floor("h")).any()):
            raise ValueError("meteorology timestamps must match their month and hour")
        frame = frame[frame.entry_date.isin(times[months == month])]
        if not frame.derived.eq(True).all():
            raise ValueError("u/v must be derived from admitted speed/direction pairs")
        if frame.duplicated(["entry_date", "sensor_id", "magnitude_id"]).any():
            raise ValueError("duplicate wind components")
        unknown = set(frame.sensor_id) - set(catalogue.index)
        if unknown:
            raise ValueError(f"wind stations absent from catalogue: {sorted(unknown)}")
        if mapping == "station":
            # Values already estimated at AQ stations need no spatial interpolation.
            frame = frame[frame.sensor_id.isin(aq_cache.sensor_ids)]
            keys = ["entry_date", "sensor_id"]
            paired = frame.pivot(index=keys, columns="magnitude_id", values="value").reindex(columns=[811, 812])
            flags = frame.pivot(index=keys, columns="magnitude_id", values="is_interpolated").reindex(
                index=paired.index, columns=[811, 812])
            original = flags.eq(0).all(axis=1)
            valid = np.isfinite(paired.to_numpy()).all(axis=1)
            if observed_only:
                valid &= original.to_numpy()
            paired = paired.loc[valid].copy()
            paired["original"] = original.loc[paired.index]
            paired = paired.reset_index()
            if len(paired):
                station_nodes = dict(zip(map(int, aq_cache.sensor_ids),
                                         map(int, aq_cache.sensor_node_indices)))
                paired["node"] = paired.sensor_id.map(station_nodes)
                # Co-located stations share one cell; average complete vectors only.
                grouped = paired.groupby(["entry_date", "node"])[[811, 812]].mean().reset_index()
                rows = times.get_indexer(grouped.entry_date)
                columns = np.searchsorted(nodes, grouped.node.to_numpy())
                values[rows, columns] = grouped[[811, 812]].to_numpy(dtype=np.float32)
                available[rows, columns] = True
                support = paired.groupby("entry_date").agg(
                    stations=("sensor_id", "size"), original=("original", "sum"))
                rows = times.get_indexer(support.index)
                station_count[rows] = support.stations.to_numpy()
                observed_count[rows] = support.original.to_numpy()
        else:
            _interpolate_month_wind(frame, times, catalogue, centers, values, available,
                                    station_count, observed_count, observed_only=observed_only,
                                    power=power, neighbours=neighbours, max_radius_m=max_radius_m)
        input_files.append(dict(path=str(source.resolve()), sha256=file_sha256(source)))
        if progress_callback:
            progress_callback(str(month), int((station_count[months == month] > 0).sum()))
    metadata = dict(
        version=1, units="m/s", components=["east", "north"], metric_crs="EPSG:25830",
        conversion="Existing admitted 811/812: u=-VV*sin(DV), v=-VV*cos(DV)",
        aq_grid=aq_grid_identity(aq_cache), start=times[0].isoformat(), end=times[-1].isoformat(),
        mapping=mapping, source_interpolation=source_interpolation,
        interpolation=("none; direct AQ station-to-cell mapping; mean for co-located stations"
                       if mapping == "station" else "existing IdwInterpolator; same paired station mask for u and v"),
        power=float(power) if mapping == "idw" else None,
        neighbours=int(neighbours) if mapping == "idw" else None,
        max_radius_m=max_radius_m if mapping == "idw" else None,
        observed_only=bool(observed_only), allow_missing_months=bool(allow_missing_months),
        missing_months=missing, scope_files=scopes, source_files=input_files,
        catalogue_sha256=hashlib.sha256(sensors.sort_values("id").to_csv(index=False).encode()).hexdigest(),
        available_fraction=float(available.mean()), hours_with_wind=int(available.any(axis=1).sum()))
    path.mkdir(parents=True)
    arrays = {"timestamps.npy": times.asi8, "node_indices.npy": nodes,
              "wind_uv.npy": values, "wind_available.npy": available,
              "station_count.npy": station_count, "observed_station_count.npy": observed_count}
    for filename, array in arrays.items():
        np.save(path / filename, array, allow_pickle=False)
    metadata["files"] = {filename: file_sha256(path / filename) for filename in arrays}
    metadata["fingerprint"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return load_wind_source_cache(path)

def _interpolate_month_wind(frame, times, catalogue, centers, values, available,
                            station_count, observed_count, *, observed_only,
                            power, neighbours, max_radius_m):
    """Explicit opt-in for raw stations whose positions differ from AQ."""
    for timestamp, hourly in frame.groupby("entry_date", sort=False):
        pivot = hourly.pivot(index="sensor_id", columns="magnitude_id", values="value").reindex(columns=[811, 812])
        provenance = hourly.pivot(index="sensor_id", columns="magnitude_id", values="is_interpolated").reindex(
            index=pivot.index, columns=[811, 812])
        valid = np.isfinite(pivot.to_numpy()).all(axis=1)
        observed = provenance.eq(0).all(axis=1).to_numpy()
        if observed_only:
            valid &= observed
        position = int(times.searchsorted(timestamp))
        station_count[position] = int(valid.sum())
        observed_count[position] = int((valid & observed).sum())
        if not valid.any():
            continue
        paired = pivot.loc[valid]
        coordinates = catalogue.loc[paired.index, ["utm_x", "utm_y"]].to_numpy(dtype=float)
        for component, magnitude in enumerate((811, 812)):
            interpolator = IdwInterpolator.from_points(
                coordinates[:, 0], coordinates[:, 1], paired[magnitude].to_numpy(),
                power=power, k=neighbours, max_radius_m=max_radius_m)
            values[position, :, component] = interpolator(centers[:, 0], centers[:, 1], mode="points")
        available[position] = np.isfinite(values[position]).all(axis=-1)
        values[position, ~available[position]] = 0.
