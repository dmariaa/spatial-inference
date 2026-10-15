"""Observed-only hourly IDW fields on one common grid."""
import json
import re
import numpy as np
import pandas as pd
from metraq_dip.data.aq_backends import get_aq_backend
from metraq_app.data_source import POLLUTANTS
from metraq_app.grid import make_grid, aggregate_cells
from metraq_app.interpolation import reconstruct
DEFAULT_POLLUTANTS = (8, 7, 12, 9, 10, 14, 6, 30, 20)

def validate_cache(output, meta, times):
    """Reject incomplete or mismatched caches before rendering."""
    if [item['id'] for item in meta['pollutants']] != meta['pollutant_ids']:
        raise ValueError('Cached pollutant order is inconsistent.')
    if meta['window_hours'] != 24 or meta['cell_size_m'] != 1000 or meta['metric_crs'] != 'EPSG:25830':
        raise ValueError('Cached grid or time window is incompatible.')
    for pollutant in meta['pollutants']:
        path = output / f"pollutant_{pollutant['id']}.npz"
        if not path.exists():
            raise ValueError(f'Missing reconstruction cache: {path.name}')
        with np.load(path) as saved:
            expected = (len(times), *meta['grid_shape'])
            if saved['idw'].shape != expected or not np.isfinite(saved['idw']).all():
                raise ValueError(f'Invalid cached reconstruction: {path.name}')
            if not np.array_equal(saved['timestamps'], times.to_numpy()):
                raise ValueError(f'Cached timestamps differ: {path.name}')
            if not np.array_equal(saved['bounds'], meta['bounds_utm']):
                raise ValueError(f'Cached geographic extent differs: {path.name}')
            if not np.array_equal(saved['limits'], pollutant['limits']):
                raise ValueError(f'Cached concentration limits differ: {path.name}')
            if saved['counts'].shape != (len(times),) or saved['counts'].min() < 2:
                raise ValueError(f'Invalid cached observation coverage: {path.name}')

def prepare(output, month, pollutant_ids):
    """Load compatible cached fields or reconstruct each observed hour with IDW."""
    if re.fullmatch('\\d{4}-\\d{2}', month) is None:
        raise ValueError('Month must use YYYY-MM format.')
    if not 1 <= len(pollutant_ids) <= 9 or len(set(pollutant_ids)) != len(pollutant_ids):
        raise ValueError('Choose between one and nine distinct pollutant IDs.')
    if any((mid not in POLLUTANTS for mid in pollutant_ids)):
        raise ValueError('Unknown pollutant ID.')
    begin = pd.Timestamp(month + '-01')
    end = begin + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1)
    times = pd.date_range(begin - pd.Timedelta(hours=23), end, freq='h')
    manifest = output / 'data_manifest.json'
    if manifest.exists():
        meta = json.loads(manifest.read_text(encoding='utf-8'))
        if meta['month'] != month or meta['pollutant_ids'] != list(pollutant_ids):
            raise ValueError('Cache differs from requested period or pollutants; choose another output directory.')
        validate_cache(output, meta, times)
        return (meta, times)
    backend = get_aq_backend(dataset='metraq', backend='files')
    data = backend.get_measurements(start_date=times[0].to_pydatetime(), end_date=times[-1].to_pydatetime(), magnitudes=list(pollutant_ids))
    data = data.loc[data.is_interpolated.eq(False) & np.isfinite(data.value)].copy()
    sensors = backend.get_sensors(magnitudes=list(pollutant_ids))
    grid_sensors = sensors.loc[sensors.id.isin(data.sensor_id.unique())]
    if grid_sensors.empty:
        raise ValueError('No observed sensors in the requested month.')
    ctx = make_grid(grid_sensors, 1000)
    bounds = np.array([min((p.bounds[0] for p in ctx['grid_cells_m'])), min((p.bounds[1] for p in ctx['grid_cells_m'])), max((p.bounds[2] for p in ctx['grid_cells_m'])), max((p.bounds[3] for p in ctx['grid_cells_m']))])
    details = []
    for mid in pollutant_ids:
        group = data.loc[data.magnitude_id.eq(mid)]
        hourly = {pd.Timestamp(t): g for (t, g) in group.groupby('entry_date')}
        pred = []
        counts = []
        for t in times:
            if t not in hourly:
                raise ValueError(f'Missing observations for pollutant {mid} at {t}')
            (cells, stations) = aggregate_cells(ctx, sensors, hourly[t])
            if len(stations) != hourly[t].sensor_id.nunique():
                raise ValueError(f'Unmapped sensor for {mid} at {t}')
            if len(cells) < 2:
                raise ValueError(f'Insufficient spatial coverage for {mid} at {t}')
            surface = reconstruct(ctx['grid'].shape, cells, 'idw')
            known = surface[cells.row.to_numpy(int), cells.col.to_numpy(int)]
            if not np.allclose(known, cells.value):
                raise ValueError('Observed values changed')
            if not np.isfinite(surface).all():
                raise ValueError('Nonfinite reconstruction')
            if surface.min() < cells.value.min() - 0.0001 or surface.max() > cells.value.max() + 0.0001:
                raise ValueError('IDW outside observed range')
            pred.append(surface)
            counts.append(len(cells))
        pred = np.array(pred)
        counts = np.array(counts)
        limits = [float(group.value.min()), float(group.value.max())]
        np.savez_compressed(output / f'pollutant_{mid}.npz', idw=pred, timestamps=times.to_numpy(), counts=counts, bounds=bounds, limits=limits)
        (name, unit) = POLLUTANTS[mid]
        details.append(dict(id=mid, name=name, unit=unit, limits=limits, sensors=int(group.sensor_id.nunique()), min_observed_cells=int(counts.min()), max_observed_cells=int(counts.max()), observed_hours=len(hourly), records=len(group)))
        print(f'Pollutant {mid}: {len(pred)} fields; {counts.min()}-{counts.max()} observed cells/hour')
    meta = dict(month=month, pollutant_ids=list(pollutant_ids), pollutants=details, grid_shape=list(ctx['grid'].shape), bounds_utm=bounds.tolist(), metric_crs='EPSG:25830', cell_size_m=1000, window_hours=24, observed_only=True, known_values_preserved=True, full_hourly_coverage=True, idw_power=2, idw_neighbors=8, map_source='OpenStreetMap via Overpass; clipped to grid', map_copyright='https://www.openstreetmap.org/copyright')
    manifest.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return (meta, times)
