from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from pandas import DatetimeIndex

from metraq_dip.data.aq_backends import AQBackend
from metraq_dip.data.generators import (
    generate_distance_to_sensors,
    generate_hour_of_day_coords,
    generate_meteo_magnitudes,
    generate_noise_channels,
    generate_pollutant_magnitudes,
    generate_spatial_dimensions,
    generate_temporal_dimensions,
)
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
             aq_backend: AQBackend,
             observed_only: bool = False):
    # returns data between (inclusive) both dates
    df = aq_backend.get_measurements(
        start_date=start_date,
        end_date=end_date,
        magnitudes=magnitudes,
    )
    time_index = pd.date_range(start=start_date, end=end_date, freq='h')
    if observed_only:
        if "is_interpolated" not in df.columns:
            raise ValueError("Observed-only meteorology requires is_interpolated provenance (METRAQ files).")
        df = df.loc[df["is_interpolated"].eq(False)].copy()

    return df, time_index


def get_magnitudes_data(*, start_date: datetime,
                        end_date: datetime,
                        magnitudes:list,
                        sensor_ids: list[int] = None,
                        normalize: bool = False,
                        observed_only: bool = False,
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
    provenance_options = {"observed_only": True} if observed_only else {}
    df, time_index = get_data(start_date=start_date, end_date=end_date, magnitudes=magnitudes, aq_backend=aq_backend, **provenance_options)
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
    pollutant_observation_mask = np.asarray(data['pollutant_data'][1::2], dtype=bool)

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
    test_mask = _build_sensor_mask(grid_ctx=grid_ctx, sensors=test_sensors, aq_backend=aq_backend)

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
        'observation_mask': pollutant_observation_mask,
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
                 meteo_observed_only: bool = False,
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
            normalize=normalize,
            observed_only=meteo_observed_only,
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

