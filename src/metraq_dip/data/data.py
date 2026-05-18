from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DatetimeIndex
from scipy.ndimage import distance_transform_edt

from metraq_dip.data.aq_backends import AQBackend
from metraq_dip.data.traffic_data import get_traffic_grid
from metraq_dip.tools.grid import prepare_grid_context, to_grid
from metraq_dip.tools.random_tools import get_random_sensors


def get_max_min(magnitudes: list[int], *, aq_backend: AQBackend):
    return aq_backend.get_magnitude_bounds(magnitudes)

class Normalizer:
    def __init__(self, pollutants: list [int], *, aq_backend: AQBackend):
        self.pollutants = pollutants
        bounds = aq_backend.get_magnitude_bounds(pollutants)
        min_map = {magnitude_id: bounds[magnitude_id][0] for magnitude_id in pollutants}
        max_map = {magnitude_id: bounds[magnitude_id][1] for magnitude_id in pollutants}

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
    def __init__(self, pollutants: list[int], *, aq_backend: AQBackend, device: str = 'cpu'):
        self.pollutants = pollutants
        self.device = device
        bounds = aq_backend.get_magnitude_bounds(pollutants)
        min_map = {magnitude_id: bounds[magnitude_id][0] for magnitude_id in pollutants}
        max_map = {magnitude_id: bounds[magnitude_id][1] for magnitude_id in pollutants}
        
        min_vals = [min_map[p] for p in pollutants]
        max_vals = [max_map[p] for p in pollutants]
        
        # Reshape to (1, C, 1, 1, 1) to match (Batch, Pollutants, Time, Height, Width)
        self.min_values = torch.tensor(min_vals, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        self.max_values = torch.tensor(max_vals, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)

    def __call__(self, data: torch.Tensor):
        return (data - self.min_values) / (self.max_values - self.min_values)

    def inverse(self, data: torch.Tensor):
        return data * (self.max_values - self.min_values) + self.min_values

def get_grid(*, pollutants: list[int] | None = None, aq_backend: AQBackend):
    df = aq_backend.get_sensors(magnitudes=pollutants)

    # Build a grid with 1000 m cells and 1000 m margin around sensors
    ctx = prepare_grid_context(df, cell_size_m=1000, margin_m_x=3000, margin_m_y=2000)

    sensor_ids = sorted(df["id"].tolist())
    return ctx, sensor_ids


def get_data(*, start_date: datetime,
             end_date: datetime,
             magnitudes: list,
             aq_backend: AQBackend):
    # returns data between (inclusive) both dates
    df = aq_backend.get_measurements(
        start_date=start_date,
        end_date=end_date,
        magnitudes=magnitudes,
    )
    time_index = pd.date_range(start=start_date, end=end_date, freq='h')

    return df, time_index


def get_magnitudes_data(*, start_date: datetime,
                        end_date: datetime,
                        magnitudes:list,
                        sensor_ids: list[int] = None,
                        normalize: bool = False,
                        aq_backend: AQBackend) -> tuple[dict, dict, DatetimeIndex, list, dict]:
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
    df, time_index = get_data(start_date=start_date, end_date=end_date, magnitudes=magnitudes, aq_backend=aq_backend)
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
                                  normalize: bool,
                                  aq_backend: AQBackend) -> tuple[np.ndarray,DatetimeIndex,list,dict]:
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
    values, masks, time_index, sensor_ids, minmax_map = get_magnitudes_data(start_date=start_date,
                                                    end_date=end_date,
                                                    magnitudes=pollutants,
                                                    sensor_ids=sensor_ids,
                                                    normalize=normalize,
                                                    aq_backend=aq_backend)

    chans = []
    for mag_id in pollutants:
        v = values[mag_id]
        m = masks[mag_id]
        chans.append(v[None, ...])
        chans.append(m[None, ...])

    x = np.concatenate(chans, axis=0)
    x_grid = to_grid(data=x, sensor_ids=sensor_ids, grid_ctx=grid_ctx, aq_backend=aq_backend)

    return x_grid, time_index, sensor_ids, minmax_map


def generate_noise_channels(number_of_channels: int, hours: int, rows: int, cols: int) -> np.ndarray:
    noise = np.random.rand(number_of_channels, hours, rows, cols)
    return noise


def generate_meteo_magnitudes(*, start_date: datetime,
                              end_date: datetime,
                              grid_ctx: dict,
                              sensor_ids: list[int],
                              aq_backend: AQBackend) -> tuple[ndarray, DatetimeIndex, list]:
    wind_magnitudes = [81, 82]
    meteo_magnitudes = [83, 86, 87, 88, 89]

    values, masks, time_index, _, _ = get_magnitudes_data(start_date=start_date,
                                                       end_date=end_date,
                                                       magnitudes=meteo_magnitudes,
                                                       sensor_ids=sensor_ids,
                                                       normalize=False,
                                                       aq_backend=aq_backend
                                                       )

    # transform wind speed + direction to u, v vector
    df, _ = get_data(start_date=start_date, end_date=end_date, magnitudes=wind_magnitudes, aq_backend=aq_backend)
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
    X_grid = to_grid(data=X, sensor_ids=sensor_ids, grid_ctx=grid_ctx, aq_backend=aq_backend)

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
        # escala suave; k controla “radio” efectivo en celdas
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


def _concatenate_parts(parts: list[np.ndarray]) -> np.ndarray | None:
    if not parts:
        return None

    return np.concatenate(parts, axis=0)


def _build_sensor_mask(*, grid_ctx: dict, sensors: list[int] | np.ndarray, aq_backend: AQBackend) -> np.ndarray:
    sensors = np.asarray(sensors, dtype=int)
    rows, cols = grid_ctx.get("grid").shape

    if sensors.size == 0:
        return np.zeros((rows, cols), dtype=np.int32)

    return to_grid(
        data=sensors,
        sensor_ids=sensors.tolist(),
        grid_ctx=grid_ctx,
        aq_backend=aq_backend,
    ).astype(int)


def _compute_pollutant_normalization_stats(*,
                                           pollutant_data: np.ndarray,
                                           pollutants: list[int],
                                           test_mask: np.ndarray) -> dict[int, tuple[float, float]]:
    spatial_non_test_mask = ~np.squeeze(test_mask.astype(bool))
    stats: dict[int, tuple[float, float]] = {}

    for idx, mag_id in enumerate(pollutants):
        value_idx = idx * 2
        mask_idx = value_idx + 1

        values = pollutant_data[value_idx].astype(np.float32)
        availability = pollutant_data[mask_idx].astype(bool)
        valid_mask = availability & spatial_non_test_mask[None, ...]

        current_values = values[-1][valid_mask[-1]]
        if current_values.size:
            mean = current_values.mean()
            std = current_values.std()
        else:
            mean = np.nan
            std = np.nan

        if (not np.isfinite(mean)) or (not np.isfinite(std)) or std < 1e-6:
            fallback_values = values[valid_mask]
            if fallback_values.size:
                mean = fallback_values.mean()
                std = fallback_values.std()
            else:
                mean = 0.0
                std = 1.0

        if (not np.isfinite(std)) or std < 1e-6:
            std = 1.0
        if not np.isfinite(mean):
            mean = 0.0

        stats[mag_id] = (float(mean), float(std))

    return stats


def _apply_pollutant_normalization(*,
                                   pollutant_data: np.ndarray,
                                   pollutants: list[int],
                                   pollutant_norm_stats: dict[int, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """
        Normalize pollutant value channels and return static normalization channels.

        `pollutant_data` is expected to use pollutant value/mask channels on axis 0,
        with spatial dimensions on the last two axes.

        Returns
        -------
        normalized_data:
            Copy of `pollutant_data` with pollutant value channels normalized.
        norm_channels:
            Static mean/std channels with shape `(2 * n_pollutants, H, W)`.
    """
    if pollutant_data.ndim < 3:
        raise ValueError(
            "pollutant_data must have shape (channels, ..., rows, cols)"
        )

    normalized_data = np.array(pollutant_data, copy=True)
    norm_channels = []
    rows, cols = pollutant_data.shape[-2:]

    for idx, mag_id in enumerate(pollutants):
        value_idx = idx * 2
        mean, std = pollutant_norm_stats[mag_id]

        normalized_data[value_idx] = (normalized_data[value_idx] - mean) / (std + 1e-6)

        mean_data = np.full((rows, cols), mean, dtype=np.float32)
        std_data = np.full((rows, cols), std, dtype=np.float32)

        norm_channels.append(mean_data[None, ...])
        norm_channels.append(std_data[None, ...])

    return normalized_data, _concatenate_parts(norm_channels)


def collect_ensemble_data(*,
                          data: dict,
                          number_of_noise_channels: int,
                          number_of_val_sensors: int,
                          add_distance_to_sensors: bool,
                          normalize: bool = False,
                          aq_backend: AQBackend) -> dict:
    """
    Build the split-dependent data for a single ensemble member from the static data collected by `collect_data`.
    """
    grid_ctx = data['grid_ctx']
    sensor_ids = data['sensor_ids']
    test_sensors = data['test_sensors']
    pollutant_input_data = data['pollutant_data']
    pollutant_value_data = data['pollutant_value_data']

    available_sensors = [sid for sid in sensor_ids if sid not in test_sensors]
    train_sensors, val_sensors, _ = get_random_sensors(
        val_number=number_of_val_sensors,
        test_number=0,
        pollutants=data['pollutants'],
        sensors=available_sensors,
        aq_backend=aq_backend,
    )

    assert np.intersect1d(test_sensors, train_sensors).size == 0, "Sensors in test_sensors have leaked to train_sensors"
    assert np.intersect1d(val_sensors, train_sensors).size == 0, "Sensors in val_sensors have leaked to train_sensors"
    assert np.intersect1d(test_sensors, val_sensors).size == 0, "sensors in test_sensors have leaked to val_sensors"

    train_mask = _build_sensor_mask(grid_ctx=grid_ctx, sensors=train_sensors, aq_backend=aq_backend)
    val_mask = _build_sensor_mask(grid_ctx=grid_ctx, sensors=val_sensors, aq_backend=aq_backend)
    test_mask = np.array(data['test_mask'], copy=True).astype(int)

    parts = []

    noise = generate_noise_channels(
        number_of_channels=number_of_noise_channels,
        hours=data['hours'],
        rows=data['rows'],
        cols=data['cols'],
    )
    parts.append(noise)

    static_input_prefix = data.get('static_input_prefix')
    if static_input_prefix is not None:
        parts.append(static_input_prefix)

    if add_distance_to_sensors:
        distance = generate_distance_to_sensors(train_mask.astype(bool), data['hours'], normalize="max")
        parts.append(distance)

    static_input_suffix = data.get('static_input_suffix')
    if static_input_suffix is not None:
        parts.append(static_input_suffix)

    train_data = pollutant_value_data * train_mask.astype(bool)
    val_data = pollutant_value_data * val_mask.astype(bool)
    test_data = np.array(data['test_data'], copy=True)

    if normalize:
        pollutant_norm_channels = data.get('pollutant_norm_channels')
        if pollutant_norm_channels is not None:
            parts.append(pollutant_norm_channels[:, None, :, :].repeat(data["hours"], axis=1))

    parts.append(pollutant_input_data * train_mask.astype(bool))

    final_data = np.concatenate(parts, axis=0)

    return {
        'input_data': final_data,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'time_index': data['time_index'],
        'train_mask': train_mask.astype(bool),
        'val_mask': val_mask.astype(bool),
        'test_mask': test_mask.astype(bool),
        'sensors': train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int),
        'pollutants': list(data['pollutants']),
        'normalization_stats': dict(data.get('pollutant_norm_stats') or {}) if normalize else None,
        'minmax_map': dict(data.get('pollutant_norm_stats') or {}) if normalize else None
    }


def collect_data(*, start_date: datetime,
                 end_date: datetime,
                 add_meteo: bool,
                 add_time_channels: bool,
                 add_coordinates: bool,
                 add_traffic_data: bool,
                 pollutants: list[int],
                 test_sensors: list[int] = None,
                 normalize: bool = False,
                 aq_backend: AQBackend,
                 ) -> dict:
    """
    Collect the static data for a whole training run.

    This includes the raw pollutant grid, fixed test mask/data, and the input channels that do not depend on the
    train/validation split. Use `collect_ensemble_data` to materialize the per-ensemble tensors.
    """
    test_sensors = [] if test_sensors is None else list(test_sensors)
    grid_ctx, sensor_ids = get_grid(pollutants=pollutants, aq_backend=aq_backend)
    rows, cols = grid_ctx.get("grid").shape
    hours = (end_date - start_date) // timedelta(hours=1) + 1
    test_mask = _build_sensor_mask(grid_ctx=grid_ctx, sensors=test_sensors, aq_backend=aq_backend)

    # Get pollutants data
    pd_data, time_index, _, _ = generate_pollutant_magnitudes(start_date=start_date,
                                                              end_date=end_date,
                                                              pollutants=pollutants,
                                                              grid_ctx=grid_ctx,
                                                              sensor_ids=sensor_ids,
                                                              normalize=False,
                                                              aq_backend=aq_backend)

    pollutant_norm_stats = None
    pollutant_norm_channels = None
    if normalize:
        pollutant_norm_stats = _compute_pollutant_normalization_stats(
            pollutant_data=pd_data,
            pollutants=pollutants,
            test_mask=test_mask,
        )
        pd_data, pollutant_norm_channels = _apply_pollutant_normalization(
            pollutant_data=pd_data,
            pollutants=pollutants,
            pollutant_norm_stats=pollutant_norm_stats,
        )

    pollutant_value_data = np.array(pd_data[::2], copy=True)

    static_input_prefix = []
    static_input_suffix = []

    # Generate coordinates channels
    if add_coordinates:
        spatial_coords = generate_spatial_dimensions(grid_ctx)
        temporal_coord = generate_temporal_dimensions(grid_ctx, hours)
        static_input_prefix.append(spatial_coords[:, None, :, :].repeat(hours, axis=1))
        static_input_prefix.append(temporal_coord)

    # Generate time channels
    if add_time_channels:
        times = generate_hour_of_day_coords(time_index, rows, cols)
        times = times.reshape(2, hours, rows, cols)
        static_input_prefix.append(times)

    # Generate traffic channels
    if add_traffic_data:
        traffic_data, traffic_mask, _, _ = get_traffic_grid(start_date=start_date,
                                                            end_date=end_date,
                                                            grid_ctx=grid_ctx)
        if normalize and traffic_mask is not None:
            valid_traffic = traffic_data[traffic_mask.astype(bool)]
            if valid_traffic.size:
                traffic_mean = valid_traffic.mean()
                traffic_std = valid_traffic.std()

                if (not np.isfinite(traffic_std)) or traffic_std < 1e-6:
                    traffic_std = 1.0

                traffic_data = np.array(traffic_data, copy=True)
                traffic_data[traffic_mask.astype(bool)] = (
                    traffic_data[traffic_mask.astype(bool)] - traffic_mean
                ) / (traffic_std + 1e-6)

        static_input_prefix.append(traffic_data)

    # Generate meteo channels
    if add_meteo:
        meteo, _, meteo_mags = generate_meteo_magnitudes(
            start_date=start_date,
            end_date=end_date,
            grid_ctx=grid_ctx,
            sensor_ids=sensor_ids,
            aq_backend=aq_backend,
        )
        static_input_suffix.append(meteo)

    test_data = pollutant_value_data * test_mask.astype(bool)

    return {
        'grid_ctx': grid_ctx,
        'sensor_ids': sensor_ids,
        'pollutants': pollutants,
        'test_sensors': test_sensors,
        'pollutant_data': pd_data,
        'pollutant_value_data': pollutant_value_data,
        'pollutant_norm_channels': pollutant_norm_channels,
        'static_input_prefix': _concatenate_parts(static_input_prefix),
        'static_input_suffix': _concatenate_parts(static_input_suffix),
        'rows': rows,
        'cols': cols,
        'hours': hours,
        'test_data': test_data,
        'time_index': time_index.tolist(),
        'test_mask': test_mask.astype(bool),
        'pollutant_norm_stats': pollutant_norm_stats,
    }

