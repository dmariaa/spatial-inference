from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from metraq_dip.data.data import get_grid
from metraq_dip.data.metraq_db import metraq_db
from metraq_dip.tools.grid import find_grid_cell


def to_grid(*, data: np.ndarray, sensor_ids: list, grid_ctx: dict):
    """
    TODO: This code is almost duplicated in data.py, just refactor
    Receives data in shape (channels, timestamps, sensors) and returns the same data in
    shape (channels, timestamps, rows, cols) where rows, cols are the coordinates of the sensor in the grid.
    Cells with no sensors have 0 value in all channels.

    The sensor_ids parameter must contain the list of sensors that matches the sensors dimension of the data.

    :param data:
    :param grid_ctx:
    :param sensor_ids:
    :return:
    """
    h, w = grid_ctx.get("grid").shape
    m, t, s = data.shape

    df_sensors = pd.read_sql("SELECT id, utm_x, utm_y FROM traffic_sensors", con=metraq_db.connection)

    if sensor_ids is None:
        sensor_ids = df_sensors["id"].to_numpy()

    rows = np.zeros(s, dtype=np.int64)
    cols = np.zeros(s, dtype=np.int64)

    for index, sensor_id in enumerate(sensor_ids):
        utm_x, utm_y = df_sensors.loc[df_sensors['id'] == sensor_id][['utm_x', 'utm_y']].values[0]

        try:
            r, c = find_grid_cell(grid_ctx, utm_x, utm_y, return_polygon=False)
        except IndexError:
            continue

        rows[index] = r
        cols[index] = c

    X_new = np.zeros((m, t, h, w), dtype=np.float32)
    X_new[:, :, rows, cols] = data

    return X_new


def get_traffic_data(*, start_date: datetime,
                     end_date: datetime):
    year = start_date.year
    month = start_date.month
    partition = f"p{year}{month}"

    data_query = f"""SELECT
                        tr.sensor_id,
                        DATE_FORMAT(tr.entry_date, '%Y-%m-%d %H:00:00') AS hour,
                        SUM(tr.traffic_intensity)  AS traffic_intensity,
                        AVG(tr.avg_speed)          AS avg_speed,
                        AVG(tr.sensor_occupancy)   AS sensor_occupancy
                    FROM traffic_data tr
                    WHERE tr.entry_date >= :start_date
                      AND tr.entry_date <= :end_date
                    GROUP BY tr.sensor_id, hour
                    ORDER BY tr.sensor_id, hour
    """

    params: dict = { 'start_date': start_date, 'end_date': end_date }
    df: pd.DataFrame = pd.read_sql_query(text(data_query), con=metraq_db.connection, params=params, parse_dates=['hour'])
    time_index = pd.date_range(start=start_date, end=end_date, freq='h')

    mat = df.pivot_table(
        index="hour",
        columns="sensor_id",
        values="traffic_intensity",
        aggfunc="mean",
    )

    sensor_ids = sorted(df['sensor_id'].unique().tolist())
    mat = mat.reindex(index=time_index, columns=sensor_ids)

    mask = (~mat.isna()).astype(np.float32).to_numpy()
    val = mat.fillna(0.0).astype(np.float32).to_numpy()
    d = np.concatenate([val[None, ...], mask[None, ...]], axis=0)

    return d, sensor_ids, time_index

if __name__ == "__main__":
    grid_ctx, _ = get_grid()
    data, sensor_ids, time_index = get_traffic_data(start_date=datetime.strptime('2024-03-12 09:00:00', '%Y-%m-%d %H:%M:%S'),
                        end_date=datetime.strptime('2024-03-13 08:00:00', '%Y-%m-%d %H:%M:%S'))

    gridded_data = to_grid(data=data, sensor_ids=sensor_ids, grid_ctx=grid_ctx)

    print(f"Length data: {len(data)}")
    print(data.head(n=10))