from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DatetimeIndex
from scipy.ndimage import distance_transform_edt
from sqlalchemy import text

from metraq_dip.data.metraq_db import metraq_db
from metraq_dip.tools.grid import find_grid_cell, prepare_grid_context
from metraq_dip.tools.random_tools import get_random_sensors


def get_max_min(magnitudes: list[int]):
    query = (
        "SELECT id, min_value, max_value "
        f"FROM aq_magnitudes WHERE id IN ({','.join(map(str, magnitudes))})"
    )

    rows = metraq_db.execute(query)
    minmax_map = {row[0]: (row[1], row[2]) for row in rows}

    # return min_values, max_values
    return minmax_map

class Normalizer:
    def __init__(self, pollutants: list [int]):
        self.pollutants = pollutants

        query = (
            "SELECT id, min_value, max_value "
            f"FROM aq_magnitudes WHERE id IN ({','.join(map(str, pollutants))})"
        )
        rows = metraq_db.execute(query)

        # Map by id so ordering matches `pollutants`
        min_map = {row[0]: row[1] for row in rows}
        max_map = {row[0]: row[2] for row in rows}

        self.min_values = np.array([min_map[p] for p in pollutants], dtype=np.float32)[:, None, None, None]
        self.max_values = np.array([max_map[p] for p in pollutants], dtype=np.float32)[:, None, None, None]

    def __call__(self, data: np.ndarray):
        return (data.astype(np.float32) - self.min_values) / (self.max_values - self.min_values)

    def inverse(self, data: np.ndarray):
        return data.astype(np.float32) * (self.max_values - self.min_values) + self.min_values


class MinMaxNormalizer:
    def __init__(self, data: torch.Tensor):
        self.min_values = data.min()
        self.max_values = data.max()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.min_values) / (self.max_values - self.min_values)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data * (self.max_values - self.min_values) + self.min_values


class TensorNormalizer:
    def __init__(self, pollutants: list[int], device: str = 'cpu'):
        self.pollutants = pollutants
        self.device = device
        
        query = f'SELECT id, min_value, max_value FROM aq_magnitudes WHERE id in ({",".join([str(p) for p in pollutants])})'
        rows = metraq_db.execute(query)
        
        min_map = {row[0]: row[1] for row in rows}
        max_map = {row[0]: row[2] for row in rows}
        
        min_vals = [min_map[p] for p in pollutants]
        max_vals = [max_map[p] for p in pollutants]
        
        # Reshape to (1, C, 1, 1, 1) to match (Batch, Pollutants, Time, Height, Width)
        self.min_values = torch.tensor(min_vals, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        self.max_values = torch.tensor(max_vals, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)

    def __call__(self, data: torch.Tensor):
        return (data - self.min_values) / (self.max_values - self.min_values)

    def inverse(self, data: torch.Tensor):
        return data * (self.max_values - self.min_values) + self.min_values

def get_grid():
    query = R"""
            SELECT id,
                   name,
                   latitude,
                   longitude
            FROM merged_sensors
            WHERE id IN (SELECT DISTINCT sensor_id FROM MAD_merged_aq_data)
            """
    df = pd.read_sql_query(text(query), con=metraq_db.connection)

    # Build a grid with 1000 m cells and 1000 m margin around sensors
    ctx = prepare_grid_context(df, cell_size_m=1000, margin_m_x=3000, margin_m_y=2000)

    sensor_ids = sorted(df["id"].tolist())
    return ctx, sensor_ids


def get_data(*, start_date: datetime,
             end_date: datetime,
             magnitudes: list):
    # returns data between (inclusive) both dates
    data_query = """
                 SELECT md.sensor_id,
                        md.entry_date,
                        md.magnitude_id,
                        md.value
                 FROM MAD_merged_aq_data md
                 WHERE is_valid
                   AND entry_date >= :start_date \
                   AND entry_date <= :end_date \
                 """

    data_query += f" AND magnitude_id IN ({','.join([str(p) for p in magnitudes])})"
    params: dict = { 'start_date': start_date, 'end_date': end_date }
    df: pd.DataFrame = pd.read_sql_query(text(data_query), con=metraq_db.connection, params=params, parse_dates=['entry_date'])
    time_index = pd.date_range(start=start_date, end=end_date, freq='h')

    return df, time_index


def get_magnitudes_data(*, start_date: datetime,
                        end_date: datetime,
                        magnitudes:list,
                        sensor_ids: list[int] = None,
                        normalize: bool = False) -> tuple[dict, dict, DatetimeIndex, list, dict]:
    """
    Returns values, masks, time_index where:
        values dict(mag_id: values(t, s)) where mag_id is the magnitude id (for all the magnitudes requested) and
             the values matrix contains the values per (timestamps, sensors) for that given magnitude,
             between the dates (inclusive) and for all the sensors that have any data if sensor_ids is None,
             or for the sensors included the list of sensor_ids if passed.

        masks dict(mag_id: mask(t, s)) where mag_id is the magnitude id and the maks matrix contains 0 for every
             (timestamp, sensor) combination that doesn't have a value and 1 elsewhere. It can be used to distinguish
             valid zero values (mask=1) from missing ones (mask=0).
    """
    df, time_index = get_data(start_date=start_date, end_date=end_date, magnitudes=magnitudes)
    values: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}

    minmax_map = {} if normalize else None

    if sensor_ids is None:
        sensor_ids = sorted(df['sensor_id'].unique().tolist())

    for idx, mag_id in enumerate(magnitudes):
        df_mag = df[df['magnitude_id'] == mag_id]

        mat = df_mag.pivot_table(
            index="entry_date",
            columns="sensor_id",
            values="value",
            aggfunc="mean",
        )

        mat = mat.reindex(index=time_index, columns=sensor_ids)

        mask = (~mat.isna()).astype(np.float32).to_numpy()

        # TODO: Refix normalization
        if normalize:
            # min_val, max_val = minmax_map[mag_id]
            # mat = (mat - min_val) / (max_val - min_val + 1e-6)
            mean = mat.values.mean()
            std = mat.values.std()
            mat = (mat - mean) / (std + 1e-6)
            minmax_map[mag_id] = (mean, std)

        val = mat.fillna(0.0).astype(np.float32).to_numpy()

        values[mag_id] = val
        masks[mag_id] = mask

    return values, masks, time_index, sensor_ids, minmax_map

def generate_pollutant_magnitudes(start_date: datetime,
                                  end_date: datetime,
                                  pollutants: list[int],
                                  grid_ctx: dict,
                                  sensor_ids: list[int],
                                  normalize: bool) -> tuple[np.ndarray,DatetimeIndex,list,dict]:
    values, masks, time_index, sensor_ids, minmax_map = get_magnitudes_data(start_date=start_date,
                                                    end_date=end_date,
                                                    magnitudes=pollutants,
                                                    sensor_ids=sensor_ids,
                                                    normalize=normalize)

    chans = []
    for mag_id in pollutants:
        v = values[mag_id]
        m = masks[mag_id]
        chans.append(v[None, ...])
        chans.append(m[None, ...])

    x = np.concatenate(chans, axis=0)
    x_grid = to_grid(data=x, sensor_ids=sensor_ids, grid_ctx=grid_ctx)

    return x_grid, time_index, sensor_ids, minmax_map


def generate_noise_channels(number_of_channels: int, hours: int, rows: int, cols: int) -> np.ndarray:
    noise = np.random.rand(number_of_channels, hours, rows, cols)
    return noise


def generate_meteo_magnitudes(*, start_date: datetime,
                              end_date: datetime,
                              grid_ctx: dict,
                              sensor_ids: list[int]) -> tuple[ndarray, DatetimeIndex, list]:
    wind_magnitudes = [81, 82]
    meteo_magnitudes = [83, 86, 87, 88, 89]

    values, masks, time_index, _, _ = get_magnitudes_data(start_date=start_date,
                                                       end_date=end_date,
                                                       magnitudes=meteo_magnitudes,
                                                       normalize=False
                                                       )

    # transform wind speed + direction to u, v vector
    df, _ = get_data(start_date=start_date, end_date=end_date, magnitudes=wind_magnitudes)
    df_wind = df[df['magnitude_id'].isin(wind_magnitudes)]
    df_wide = df_wind.pivot_table(
        index=["sensor_id", "entry_date"],
        columns="magnitude_id",
        values="value",
        aggfunc="mean",
    )
    wind_valid = ((~df_wide[81].isna()) & (~df_wide[82].isna()))
    wind_mask = wind_valid.astype("float32")
    rad = np.deg2rad(df_wide[82])
    u = (-df_wide[81] * np.sin(rad)).where(wind_valid).fillna(0.0).astype(np.float32)
    v = (-df_wide[81] * np.cos(rad)).where(wind_valid).fillna(0.0).astype(np.float32)

    u_val = u.unstack("sensor_id").reindex(index=time_index, columns=sensor_ids).to_numpy().astype(np.float32)
    v_val = v.unstack("sensor_id").reindex(index=time_index, columns=sensor_ids).to_numpy().astype(np.float32)
    u_mask = wind_mask.unstack("sensor_id").reindex(index=time_index, columns=sensor_ids).to_numpy().astype(np.float32)
    v_mask = u_mask.copy()

    values[811] = u_val
    masks[811] = u_mask

    values[812] = v_val
    masks[812] = v_mask

    meteo_mags = [811, 812] + meteo_magnitudes
    chans = []
    for mag_id in meteo_mags:
        v = values[mag_id]
        m = masks[mag_id]
        chans.append(v[None, ...])
        chans.append(m[None, ...])

    X = np.concatenate(chans, axis=0)
    X_grid = to_grid(data=X, sensor_ids=sensor_ids, grid_ctx=grid_ctx)

    return X_grid, time_index, meteo_mags


def to_grid(*, data: np.ndarray, sensor_ids: list, grid_ctx: dict):
    """
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

    df_sensors = pd.read_sql_query(text("SELECT id, utm_x, utm_y FROM merged_sensors"), con=metraq_db.connection)

    rows = np.zeros(s, dtype=np.int64)
    cols = np.zeros(s, dtype=np.int64)

    for index, sensor_id in enumerate(sensor_ids):
        utm_x, utm_y = df_sensors.loc[df_sensors['id'] == sensor_id][['utm_x', 'utm_y']].values[0]
        r, c = find_grid_cell(grid_ctx, utm_x, utm_y, return_polygon=False)
        rows[index] = r
        cols[index] = c

    X_new = np.zeros((m, t, h, w), dtype=np.float32)
    X_new[:, :, rows, cols] = data

    return X_new


def generate_distance_to_sensors(sensors_mask: np.ndarray, T: int, normalize: str = "max", eps: float = 1e-6) \
        -> np.ndarray:
    sensors_mask = sensors_mask.astype(bool)
    dist = distance_transform_edt((~sensors_mask)).astype(np.float32)

    if normalize == "max":
        dmax = float(dist.max()) + eps
        dist_n = dist / dmax  # [0,1]
    elif normalize == "log":
        dist_n = np.log1p(dist)
        dist_n = dist_n / (float(dist_n.max()) + eps)  # [0,1]
    elif normalize == "tanh":
        # escala suave; k controla “radio” efectivo en celdas
        k = max(1.0, float(dist.max()) / 3.0)
        dist_n = np.tanh(dist / k)  # [0,~1]
    else:
        raise ValueError("normalize must be 'max', 'log', or 'tanh'")

    dist_n = (dist_n * 2.0 - 1.0).astype(np.float32)  # [-1,1]

    dist_ch = np.tile(dist_n, (1, T, 1, 1)).astype(np.float32)
    return dist_ch

def generate_dimensions(grid_ctx: dict, T: int):
    H, W = grid_ctx.get("grid").shape
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    t = np.linspace(-1.0, 1.0, T, dtype=np.float32)

    xx = np.tile(x[None, :], (H, 1))
    yy = np.tile(y[:, None], (1, W))

    xx = np.tile(xx[None, :, :], (T, 1, 1))
    yy = np.tile(yy[None, :, :], (T, 1, 1))

    tt = t[:, None, None]
    tt = np.tile(tt, (1, H, W))

    coords = np.stack([xx, yy, tt], axis=0)  # (3,T,H,W)
    return coords


def generate_hour_of_day_coords(
    time_index: pd.DatetimeIndex,
    rows: int,
    cols: int,
    dtype=np.float32
) -> np.ndarray:
    # hour of day: 0..23
    hours = time_index.hour.values.astype(dtype)

    # cyclic encoding
    ang = 2.0 * np.pi * hours / 24.0
    sin_h = np.sin(ang).astype(dtype)
    cos_h = np.cos(ang).astype(dtype)

    T = len(time_index)

    # reshape to (T,1,1) then broadcast
    sin_h = sin_h[:, None, None]
    cos_h = cos_h[:, None, None]

    sin_h = np.tile(sin_h, (1, rows, cols))  # (T,H,W)
    cos_h = np.tile(cos_h, (1, rows, cols))  # (T,H,W)

    return np.stack([sin_h, cos_h], axis=0)  # (2,T,H,W)


def collect_data(*, start_date: datetime,
                 end_date: datetime,
                 number_of_noise_channels: int,
                 add_meteo: bool,
                 add_time_channels: bool,
                 add_coordinates: bool,
                 add_distance_to_sensors: bool,
                 number_of_val_sensors: int,
                 pollutants: list[int],
                 test_sensors: list[int] = None,
                 normalize: bool = False
                 ) -> dict:
    """
    Returns (data, val_data, sensors_mask, val_sensors_mask, sensor_ids, time_index)

    where data is (masked with sensors_mask):
        [C, T, H, W] where
        C:  - number_of_noise_channels [0-1] noise channels
            - (7x2) meteo channels and masks
            - (3) coordinates dimension, x, y, hour in time window
            - (2) temporal dimension, sin(hour of day), cos(hour of day)
        T: hours
        H: rows
        W: cols
    """

    grid_ctx, sensor_ids = get_grid()
    rows, cols = grid_ctx.get("grid").shape

    available_sensors = [sid for sid in sensor_ids if sid not in test_sensors]
    train_sensors, val_sensors, _ = get_random_sensors(val_number=number_of_val_sensors, test_number=0,
                                                       pollutants=pollutants, sensors=available_sensors)

    assert np.intersect1d(test_sensors, train_sensors).size == 0, "Sensors in test_sensors have leaked to train_sensors"
    assert np.intersect1d(val_sensors, train_sensors).size == 0, "Sensors in val_sensors have leaked to train_sensors"
    assert np.intersect1d(test_sensors, val_sensors).size == 0, "sensors in test_sensors have leaked to val_sensors"

    train_mask = to_grid(data=np.array(train_sensors)[None, None, :], sensor_ids=train_sensors, grid_ctx=grid_ctx).astype(int)
    val_mask = to_grid(data=np.array(val_sensors)[None, None, :], sensor_ids=val_sensors, grid_ctx=grid_ctx).astype(int)
    test_mask = to_grid(data=np.array(test_sensors)[None, None, :], sensor_ids=test_sensors, grid_ctx=grid_ctx).astype(int)

    parts = []

    # Get pollutants data
    pd_data, time_index, s, minmax_map = generate_pollutant_magnitudes(start_date=start_date,
                                                        end_date=end_date,
                                                        pollutants=pollutants,
                                                        grid_ctx=grid_ctx,
                                                        sensor_ids=sensor_ids,
                                                        normalize=False)

    # Generate noise channels
    hours = (end_date - start_date) // timedelta(hours=1) + 1
    noise = generate_noise_channels(number_of_channels=number_of_noise_channels, hours=hours, rows=rows, cols=cols)
    parts.append(noise)

    # Generate coordinates channels
    if add_coordinates:
        coords = generate_dimensions(grid_ctx, hours)
        parts.append(coords)

    # Generate time channels
    if add_time_channels:
        times = generate_hour_of_day_coords(time_index, rows, cols)
        parts.append(times)

    # Generate distance channels
    # TODO: There is a problem here if we dont have the train/val/test masks, this can't be calculated without the
    #  validation and test masks...
    if add_distance_to_sensors:
        distance = generate_distance_to_sensors(train_mask.astype(bool), hours, normalize="max")
        parts.append(distance)

    # Generate meteo channels
    if add_meteo:
        meteo, _, meteo_mags = generate_meteo_magnitudes(start_date=start_date, end_date=end_date, grid_ctx=grid_ctx, sensor_ids=sensor_ids)
        parts.append(meteo)

    # if normalize:
    #     norm_data = []
    #     for i, (mag_id, minmax) in enumerate(minmax_map.items()):
    #         mean_val, std_val = minmax
    #         mean_data = np.full_like(pd_data[i], mean_val, dtype=np.float32)
    #         std_data = np.full_like(pd_data[i], std_val, dtype=np.float32)
    #         norm_data.append(mean_data[None, ...])
    #         norm_data.append(std_data[None, ...])
    #
    #     norm_data = np.concatenate(norm_data, axis=0)
    #     parts.append(norm_data)

    # This is the data used to train the model
    train_data = pd_data[:1, :, :, :] * train_mask.astype(bool)
    val_data = pd_data[:1, :, :, :] * val_mask.astype(bool)
    test_data = pd_data[:1, :, :, :] * test_mask.astype(bool)

    if normalize:
        # Normalization for ALL data should be done using only train_data + val_data (known data)
        # to avoid information leakage into the test set.
        norm_data = []
        input_data = []
        minmax_map = {}
        n_mask = np.squeeze(train_mask.astype(bool) | val_mask.astype(bool))
        n_data = (train_data + val_data)
        T = train_data.shape[2]

        for idx, mag_id in enumerate(pollutants):
            # d = n_data[idx, :, n_mask]
            d = n_data[idx, -1, n_mask]         # normalize to current time values (mean, std)
            mean = d.mean()
            std = d.std()

            if (not np.isfinite(std)) or std < 1e-6:
                d_fb = n_data[idx, :, n_mask]   # fallback to full 24 h window
                mean = d_fb.mean()
                std = d_fb.std()

            minmax_map[mag_id] = (mean, std)
            mean_data = np.full_like(train_data[idx], mean, dtype=np.float32)
            std_data = np.full_like(train_data[idx], std, dtype=np.float32)
            norm_data.append(mean_data[None, ...])
            norm_data.append(std_data[None,  ...])

            train_data[idx, :, train_mask[idx, 0].astype(bool)] = (train_data[idx, :, train_mask[idx, 0].astype(bool)] - mean) / (std + 1e-6)
            val_data[idx, :, val_mask[idx, 0].astype(bool)] = (val_data[idx, :, val_mask[idx, 0].astype(bool)] - mean) / (std + 1e-6)
            test_data[idx, :, test_mask[idx, 0].astype(bool)] = (test_data[idx, :, test_mask[idx, 0].astype(bool)] - mean) / (std + 1e-6)

            input_data.append(train_data[idx][None, ...])
            input_data.append(train_mask[idx][None, ...].astype(bool).astype(float).repeat(T, 1))

        norm_data = np.concatenate(norm_data, axis=0)
        parts.append(norm_data)
        input_data = np.concatenate(input_data, axis=0)
        parts.append(input_data)
    else:
        # This is the input pollutant data to the model
        # it includes all the timestamps for the pollutant including the boolean mask
        input_data = pd_data * train_mask.astype(bool)
        parts.append(input_data)

    # Here we concatenate all input channels
    final_data =  np.concatenate(parts, axis=0)

    return {
        'input_data': final_data,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'time_index': time_index.tolist(),
        'train_mask': train_mask.astype(bool),
        'val_mask': val_mask.astype(bool),
        'test_mask': test_mask.astype(bool),
        'sensors': train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int),
        'minmax_map': minmax_map
    }




if __name__ == "__main__":
    d = collect_data(start_date=datetime.strptime('2024-03-12 09:00:00', '%Y-%m-%d %H:%M:%S'),
                        end_date=datetime.strptime('2024-03-13 08:00:00', '%Y-%m-%d %H:%M:%S'),
                        number_of_noise_channels=8,
                        add_meteo=True,
                        add_time_channels=True,
                        add_coordinates=True,
                        add_distance_to_sensors=True,
                        pollutants=[7, 8],
                        normalize=True
                        )

    pass
