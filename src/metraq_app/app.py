"""Local METRAQ HTTP API and static frontend."""
from datetime import datetime
from functools import lru_cache
import logging
import numpy as np
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from metraq_app.data_source import POLLUTANTS, load_catalog, load_sensors, load_measurements
from metraq_app.grid import make_grid, aggregate_cells
from metraq_app.map import build_map
from metraq_app.interpolation import METHOD_NAMES, reconstruct

Source = Literal['files', 'db']
FRONTEND = Path(__file__).with_name('frontend')
app = FastAPI(title='METRAQ Madrid', docs_url=None, redoc_url=None)
app.mount('/static', StaticFiles(directory=FRONTEND), name='static')
logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def catalog_for(source):
    return load_catalog(source).sort_values('magnitude_id')


@lru_cache(maxsize=6)
def grid_for(source, cell_size):
    catalog = catalog_for(source)
    sensors = load_sensors(source, tuple(catalog.magnitude_id.astype(int)))
    if sensors.empty:
        raise ValueError('No sensors')
    return make_grid(sensors, cell_size), sensors


@lru_cache(maxsize=128)
def observations_for(source, magnitude, timestamp):
    return load_measurements(source, magnitude, timestamp)


@lru_cache(maxsize=32)
def surface_for(source, magnitude, timestamp, cell_size, method, dip_epochs):
    ctx, sensors = grid_for(source, cell_size)
    cells, _ = aggregate_cells(ctx, sensors, observations_for(source, magnitude, timestamp))
    return reconstruct(ctx['grid'].shape, cells, method, dip_epochs)


def clear_cache():
    surface_for.cache_clear()
    observations_for.cache_clear()
    grid_for.cache_clear()
    catalog_for.cache_clear()


@app.get('/')
def index():
    return FileResponse(FRONTEND / 'index.html')


@app.get('/api/config')
def configuration():
    source = os.getenv('METRAQ_APP_SOURCE', 'files')
    return {'default_source': source if source in ('files', 'db') else 'files'}


@app.get('/api/catalog')
def catalog(source: Source = 'files'):
    try:
        frame = catalog_for(source)
        result = []
        for item in frame.itertuples():
            magnitude = int(item.magnitude_id)
            if magnitude not in POLLUTANTS:
                continue
            name, unit = POLLUTANTS[magnitude]
            result.append({'id': magnitude, 'name': name, 'unit': unit,
                'first': item.first_date.isoformat(), 'last': item.last_date.isoformat(),
                'low': float(item.low), 'high': float(item.high)})
        return {'pollutants': result}
    except Exception as exc:
        logger.error('Catalog failed for %s (%s)', source, type(exc).__name__)
        raise HTTPException(503, 'No se pudo leer el origen. Comprueba la conexión o selecciona los CSV.') from None


@app.post('/api/refresh')
def refresh():
    clear_cache()
    return {'ok': True}


@app.get('/api/view')
def view(source: Source, magnitude: int, timestamp: datetime,
         cell_size: int = Query(1000), auto_scale: bool = True, show_sensors: bool = True,
         method: Literal['observed', 'dip-cnn', 'krg', 'idw'] = 'observed',
         dip_epochs: int = Query(250, ge=1, le=1000)):
    if cell_size not in (500, 1000, 2000):
        raise HTTPException(422, 'El tamaño de celda debe ser 500, 1000 o 2000 m.')
    if timestamp.tzinfo is not None or timestamp.minute or timestamp.second or timestamp.microsecond:
        raise HTTPException(422, 'Selecciona una hora completa, sin conversión de zona horaria.')
    try:
        available = catalog_for(source)
        selected = available.loc[available.magnitude_id.eq(magnitude)]
        if magnitude not in POLLUTANTS or selected.empty:
            raise HTTPException(404, 'Contaminante no disponible en este origen.')
        metadata = selected.iloc[0]
        if timestamp < metadata.first_date or timestamp > metadata.last_date:
            raise HTTPException(422, 'La hora seleccionada queda fuera del intervalo disponible.')
        ctx, sensors = grid_for(source, cell_size)
        measurements = observations_for(source, magnitude, timestamp)
        cells, stations = aggregate_cells(ctx, sensors, measurements)
        surface = surface_for(source, magnitude, timestamp, cell_size, method, dip_epochs)
        constant_field = method != 'observed' and not cells.empty and cells.value.min() == cells.value.max()
        name, unit = POLLUTANTS[magnitude]
        bounds = None if auto_scale else (float(metadata.low), float(metadata.high))
        station_records = [{'id': int(row.sensor_id), 'name': str(row.name), 'value': float(row.value),
                            'cell_id': f'{int(row.row)}:{int(row.col)}'}
                           for row in stations.itertuples()]
        return {'name': name, 'unit': unit, 'timestamp': timestamp.isoformat(),
                'map': build_map(ctx, cells, stations, unit, bounds, show_sensors, surface=surface, method=METHOD_NAMES[method]),
                'method': METHOD_NAMES[method],
                'interpolated_count': int(np.isfinite(surface).sum()) - len(cells),
                'dip_epochs': dip_epochs if method == 'dip-cnn' and not constant_field and not cells.empty else None,
                'constant_field': bool(constant_field),
                'stations': station_records, 'cell_count': len(cells),
                'mean': float(stations.value.mean()) if not stations.empty else None}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error('View failed for %s (%s)', source, type(exc).__name__)
        raise HTTPException(503, 'No se pudieron cargar las medidas. Comprueba el origen y la fecha.') from None
