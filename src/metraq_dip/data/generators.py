from datetime import datetime

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DatetimeIndex
from scipy.ndimage import distance_transform_edt

from metraq_dip.data.aq_backends import AQBackend


def generate_pollutant_magnitudes(start_date: datetime,
                                  end_date: datetime,
                                  pollutants: list[int],
                                  grid_ctx: dict,
                                  sensor_ids: list[int],
                                  normalize: bool,
                                  aq_backend: AQBackend) -> tuple[np.ndarray, DatetimeIndex, list, dict]:
    """
        Fetch pollutant measurements and map them from sensor space to grid space.

        The returned tensor has shape:

            (2 * n_pollutants, T, H, W)

        Channels are interleaved by pollutant:

            pollutant_1_value
            pollutant_1_availability_mask
            pollutant_2_value
            pollutant_2_availability_mask
            ...

        The availability mask is temporal because it represents whether a pollutant
        value exists for each timestamp and sensor. It is different from the
        train/validation/test sensor masks, which are spatial masks.

        Parameters
        ----------
        start_date:
            First timestamp to include.
        end_date:
            Last timestamp to include.
        pollutants:
            Pollutant magnitude ids to fetch.
        grid_ctx:
            Grid context returned by `prepare_grid_context`.
        sensor_ids:
            Sensor ids defining the sensor axis before gridding.
        normalize:
            Whether to normalize pollutant values inside `get_magnitudes_data`.
        aq_backend:
            Air-quality backend used to fetch measurements and sensor metadata.

        Returns
        -------
        pollutant_grid_data:
            Pollutant values and availability masks on the grid, with shape
            `(2 * n_pollutants, T, H, W)`.
        time_index:
            Hourly timestamps included between `start_date` and `end_date`.
        sensor_ids:
            Sensor ids used for the sensor axis before gridding.
        minmax_map:
            Normalization statistics returned by `get_magnitudes_data` when
            `normalize=True`; otherwise `None`.
    """
    from metraq_dip.data import data as data_module

    values, masks, time_index, sensor_ids, minmax_map = data_module.get_magnitudes_data(
        start_date=start_date,
        end_date=end_date,
        magnitudes=pollutants,
        sensor_ids=sensor_ids,
        normalize=normalize,
        aq_backend=aq_backend,
    )

    chans = []
    for mag_id in pollutants:
        v = values[mag_id]
        m = masks[mag_id]
        chans.append(v[None, ...])
        chans.append(m[None, ...])

    x = np.concatenate(chans, axis=0)
    x_grid = data_module.to_grid(data=x, sensor_ids=sensor_ids, grid_ctx=grid_ctx, aq_backend=aq_backend)

    return x_grid, time_index, sensor_ids, minmax_map


def generate_noise_channels(number_of_channels: int, hours: int, rows: int, cols: int) -> np.ndarray:
    noise = np.random.rand(number_of_channels, hours, rows, cols)
    return noise


def generate_meteo_magnitudes(*, start_date: datetime,
                              end_date: datetime,
                              grid_ctx: dict,
                              sensor_ids: list[int],
                              aq_backend: AQBackend) -> tuple[ndarray, DatetimeIndex, list]:
    from metraq_dip.data import data as data_module

    wind_magnitudes = [81, 82]
    meteo_magnitudes = [83, 86, 87, 88, 89]

    values, masks, time_index, _, _ = data_module.get_magnitudes_data(
        start_date=start_date,
        end_date=end_date,
        magnitudes=meteo_magnitudes,
        sensor_ids=sensor_ids,
        normalize=False,
        aq_backend=aq_backend,
    )

    # transform wind speed + direction to u, v vector
    df, _ = data_module.get_data(start_date=start_date, end_date=end_date, magnitudes=wind_magnitudes, aq_backend=aq_backend)
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
    X_grid = data_module.to_grid(data=X, sensor_ids=sensor_ids, grid_ctx=grid_ctx, aq_backend=aq_backend)

    return X_grid, time_index, meteo_mags


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
        # escala suave; k controla radio efectivo en celdas
        k = max(1.0, float(dist.max()) / 3.0)
        dist_n = np.tanh(dist / k)  # [0,~1]
    else:
        raise ValueError("normalize must be 'max', 'log', or 'tanh'")

    dist_n = (dist_n * 2.0 - 1.0).astype(np.float32)  # [-1,1]

    dist_ch = np.tile(dist_n, (1, T, 1, 1)).astype(np.float32)
    return dist_ch


def generate_spatial_dimensions(grid_ctx: dict) -> np.ndarray:
    """
        Generate normalized spatial coordinate channels for the grid.

        The returned tensor has shape:

            (2, H, W)

        Channel layout:

            0: x coordinate, normalized from -1 to 1 across columns
            1: y coordinate, normalized from -1 to 1 across rows

        Parameters
        ----------
        grid_ctx:
            Grid context returned by `prepare_grid_context`.

        Returns
        -------
        np.ndarray
            Spatial coordinate channels with shape `(2, H, W)`.
    """
    H, W = grid_ctx.get("grid").shape
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, H, dtype=np.float32)

    xx = np.tile(x[None, :], (H, 1))
    yy = np.tile(y[:, None], (1, W))

    return np.stack([xx, yy], axis=0)


def generate_temporal_dimensions(grid_ctx: dict, T: int) -> np.ndarray:
    """
       Generate a normalized temporal coordinate channel for the input window.

       The returned tensor has shape:

           (1, T, H, W)

       The same temporal value is repeated over all grid cells for each timestep.
       Values are normalized from -1 to 1 across the input window.

       Parameters
       ----------
       grid_ctx:
           Grid context returned by `prepare_grid_context`.
       T:
           Number of timesteps in the input window.

       Returns
       -------
       np.ndarray
           Temporal coordinate channel with shape `(1, T, H, W)`.
    """
    H, W = grid_ctx.get("grid").shape
    t = np.linspace(-1.0, 1.0, T, dtype=np.float32)

    tt = t[:, None, None]
    tt = np.tile(tt, (1, H, W))
    return tt[None, ...]


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

    return np.stack([sin_h, cos_h], axis=0).reshape(2 * T, rows, cols)  # (2*T,H,W)
