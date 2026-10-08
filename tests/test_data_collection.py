from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metraq_dip.data import data as data_module


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("no_original_wind", [False, True])
def test_observed_only_meteo_filters_before_normalization(monkeypatch, normalize, no_original_wind):
    times = pd.date_range("2024-01-01", periods=2, freq="h")
    records = []
    for mag in [81, 82, 83, 86, 87, 88, 89, 12]:
        for hour, timestamp in enumerate(times):
            for sensor in [10, 20]:
                interpolated = sensor == 20 or mag == 87 or (mag == 82 and hour == 1)
                if no_original_wind and mag in [81, 82]:
                    interpolated = True
                value = 10000.0 if interpolated else (90.0 if mag == 82 else 1.0 + 2 * hour)
                records.append((sensor, timestamp, mag, value, interpolated))
    frame = pd.DataFrame(records, columns=["sensor_id", "entry_date", "magnitude_id", "value", "is_interpolated"])

    class Backend:
        def get_measurements(self, *, start_date, end_date, magnitudes):
            return frame[frame.magnitude_id.isin(magnitudes)].copy()

    backend = Backend()
    monkeypatch.setattr(data_module, "to_grid", _fake_to_grid)
    grid, _, _ = data_module.generate_meteo_magnitudes(
        start_date=times[0], end_date=times[-1], sensor_ids=[10, 20],
        grid_ctx={"grid": np.zeros((2, 2))}, aq_backend=backend,
        normalize=normalize, observed_only=True,
    )
    assert np.isfinite(grid).all()
    np.testing.assert_array_equal(grid[:, :, 0, 1], 0.0)
    np.testing.assert_array_equal(grid[8:10], 0.0)  # No original pressure observations.
    np.testing.assert_array_equal(grid[4, :, 0, 0], [-1.0, 1.0] if normalize else [1.0, 3.0])
    np.testing.assert_array_equal(grid[5, :, 0, 0], 1.0)
    np.testing.assert_array_equal(grid[:4, 1], 0.0)  # Direction is interpolated at the second hour.
    if no_original_wind:
        np.testing.assert_array_equal(grid[:4], 0.0)
    else:
        assert grid[1, 0, 0, 0] == grid[3, 0, 0, 0] == 1.0
        np.testing.assert_allclose(grid[0, 0, 0, 0], 0.0 if normalize else -1.0, atol=1e-6)
    all_nox, _ = data_module.get_data(start_date=times[0], end_date=times[-1], magnitudes=[12], aq_backend=backend)
    assert len(all_nox) == 4  # The meteo filter does not remove NOX observations.


def test_observed_only_requires_provenance():
    class Backend:
        def get_measurements(self, **kwargs):
            return pd.DataFrame(columns=["sensor_id", "entry_date", "magnitude_id", "value"])

    with pytest.raises(ValueError, match="is_interpolated"):
        data_module.get_data(start_date=pd.Timestamp("2024-01-01"), end_date=pd.Timestamp("2024-01-01"),
                             magnitudes=[83], aq_backend=Backend(), observed_only=True)


def _fake_to_grid(*, data: np.ndarray, sensor_ids: list[int], grid_ctx: dict, aq_backend=None):
    mapping = {10: (0, 0), 20: (0, 1), 30: (1, 0), 40: (1, 1)}
    data = np.asarray(data)
    *prefix_shape, _ = data.shape
    rows, cols = grid_ctx["grid"].shape
    grid = np.zeros((*prefix_shape, rows, cols), dtype=np.float32)

    for sensor_idx, sensor_id in enumerate(sensor_ids):
        row, col = mapping[int(sensor_id)]
        grid[..., row, col] = data[..., sensor_idx]

    return grid


def _build_fake_pollutant_data():
    pollutant = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    availability = np.ones_like(pollutant, dtype=np.float32)
    return np.stack([pollutant, availability], axis=0)


def test_get_grid_filters_sensor_catalog_by_pollutants(monkeypatch):
    calls: list[tuple[list[int] | None, list[int] | None]] = []

    class FakeBackend:
        def get_sensors(self, *, magnitudes=None, sensors=None):
            calls.append((magnitudes, sensors))
            return pd.DataFrame(
                {
                    "id": [10, 20],
                    "latitude": [40.4168, 40.4300],
                    "longitude": [-3.7038, -3.7000],
                }
            )

    grid_ctx, sensor_ids = data_module.get_grid(pollutants=[7, 8], aq_backend=FakeBackend())

    assert sensor_ids == [10, 20]
    assert calls == [([7, 8], None)]
    assert grid_ctx["grid"].shape[0] > 0
    assert grid_ctx["grid"].shape[1] > 0


def test_collect_data_returns_only_static_components(monkeypatch):
    time_index = pd.date_range("2024-01-01 00:00:00", periods=2, freq="h")
    pollutant_data = _build_fake_pollutant_data()
    fake_backend = object()
    meteo_normalize_calls = []
    meteo_observed_calls = []

    def fake_meteo(**kwargs):
        meteo_normalize_calls.append(kwargs["normalize"])
        meteo_observed_calls.append(kwargs["observed_only"])
        return np.full((2, 2, 2, 2), 40.0, dtype=np.float32), time_index, [811, 812]

    monkeypatch.setattr(
        data_module,
        "get_grid",
        lambda **kwargs: ({"grid": np.zeros((2, 2), dtype=int)}, [10, 20, 30, 40]),
    )
    monkeypatch.setattr(data_module, "to_grid", _fake_to_grid)
    monkeypatch.setattr(
        data_module,
        "generate_pollutant_magnitudes",
        lambda **kwargs: (pollutant_data, time_index, [10, 20, 30, 40], None),
    )
    monkeypatch.setattr(
        data_module,
        "generate_spatial_dimensions",
        lambda grid_ctx: np.full((2, 2, 2), 10.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        data_module,
        "generate_temporal_dimensions",
        lambda grid_ctx, hours: np.full((1, hours, 2, 2), 10.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        data_module,
        "generate_hour_of_day_coords",
        lambda time_index, rows, cols: np.full((2 * len(time_index), rows, cols), 20.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        data_module,
        "get_traffic_grid",
        lambda **kwargs: (np.full((1, 2, 2, 2), 30.0, dtype=np.float32), None, None, None),
    )
    monkeypatch.setattr(
        data_module,
        "generate_meteo_magnitudes",
        fake_meteo,
    )
    monkeypatch.setattr(
        data_module,
        "get_random_sensors",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("collect_data should not sample train/val sensors")),
    )
    monkeypatch.setattr(
        data_module,
        "generate_noise_channels",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("collect_data should not build noise channels")),
    )
    monkeypatch.setattr(
        data_module,
        "generate_distance_to_sensors",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("collect_data should not build distance channels")),
    )

    result = data_module.collect_data(
        start_date=pd.Timestamp("2024-01-01 00:00:00"),
        end_date=pd.Timestamp("2024-01-01 01:00:00"),
        add_meteo=True,
        add_time_channels=True,
        add_coordinates=True,
        add_traffic_data=True,
        pollutants=[7],
        test_sensors=[40],
        aq_backend=fake_backend,
    )

    assert "input_data" not in result
    assert "train_data" not in result
    assert result["pollutant_data"].shape == (2, 2, 2, 2)
    assert result["static_input_prefix"].shape == (6, 2, 2, 2)
    assert result["static_input_suffix"].shape == (2, 2, 2, 2)
    assert result["test_mask"].dtype == np.bool_
    np.testing.assert_array_equal(result["test_mask"], np.array([[False, False], [False, True]]))
    np.testing.assert_array_equal(result["test_data"][0, :, 1, 1], np.array([4.0, 8.0], dtype=np.float32))
    data_module.collect_data(
        start_date=time_index[0],
        end_date=time_index[-1],
        add_meteo=True,
        add_time_channels=False,
        add_coordinates=False,
        add_traffic_data=False,
        pollutants=[7],
        test_sensors=[40],
        aq_backend=fake_backend,
        normalize=True,
        meteo_observed_only=True,
    )
    assert meteo_normalize_calls == [False, True]
    assert meteo_observed_calls == [False, True]


def test_collect_ensemble_data_builds_dynamic_channels_from_static_data(monkeypatch):
    time_index = pd.date_range("2024-01-01 00:00:00", periods=2, freq="h")
    pollutant_data = _build_fake_pollutant_data()
    fake_backend = object()

    monkeypatch.setattr(
        data_module,
        "get_grid",
        lambda **kwargs: ({"grid": np.zeros((2, 2), dtype=int)}, [10, 20, 30, 40]),
    )
    monkeypatch.setattr(data_module, "to_grid", _fake_to_grid)
    monkeypatch.setattr(
        data_module,
        "generate_pollutant_magnitudes",
        lambda **kwargs: (pollutant_data, time_index, [10, 20, 30, 40], None),
    )
    monkeypatch.setattr(
        data_module,
        "generate_spatial_dimensions",
        lambda grid_ctx: np.full((2, 2, 2), 10.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        data_module,
        "generate_temporal_dimensions",
        lambda grid_ctx, hours: np.full((1, hours, 2, 2), 10.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        data_module,
        "generate_hour_of_day_coords",
        lambda time_index, rows, cols: np.full((2 * len(time_index), rows, cols), 20.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        data_module,
        "get_traffic_grid",
        lambda **kwargs: (np.full((1, 2, 2, 2), 30.0, dtype=np.float32), None, None, None),
    )
    monkeypatch.setattr(
        data_module,
        "generate_meteo_magnitudes",
        lambda **kwargs: (np.full((2, 2, 2, 2), 40.0, dtype=np.float32), time_index, [811, 812]),
    )

    static_data = data_module.collect_data(
        start_date=pd.Timestamp("2024-01-01 00:00:00"),
        end_date=pd.Timestamp("2024-01-01 01:00:00"),
        add_meteo=True,
        add_time_channels=True,
        add_coordinates=True,
        add_traffic_data=True,
        pollutants=[7],
        test_sensors=[40],
        aq_backend=fake_backend,
    )

    monkeypatch.setattr(data_module, "get_random_sensors", lambda **kwargs: (np.array([10, 20]), np.array([30]), np.array([], dtype=int)))
    monkeypatch.setattr(
        data_module,
        "generate_noise_channels",
        lambda **kwargs: np.full((kwargs["number_of_channels"], kwargs["hours"], kwargs["rows"], kwargs["cols"]), 99.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        data_module,
        "generate_distance_to_sensors",
        lambda *args, **kwargs: np.full((1, 2, 2, 2), 77.0, dtype=np.float32),
    )

    result = data_module.collect_ensemble_data(
        data=static_data,
        number_of_noise_channels=2,
        number_of_val_sensors=1,
        add_distance_to_sensors=True,
        normalize=False,
        aq_backend=fake_backend,
    )

    assert result["input_data"].shape == (13, 2, 2, 2)
    np.testing.assert_array_equal(result["train_mask"], np.array([[True, True], [False, False]]))
    np.testing.assert_array_equal(result["val_mask"], np.array([[False, False], [True, False]]))
    np.testing.assert_array_equal(result["test_mask"], np.array([[False, False], [False, True]]))
    np.testing.assert_array_equal(result["sensors"], np.array([[10, 20], [30, 40]]))

    np.testing.assert_array_equal(result["input_data"][:2], np.full((2, 2, 2, 2), 99.0, dtype=np.float32))
    np.testing.assert_array_equal(result["input_data"][2:8], static_data["static_input_prefix"])
    np.testing.assert_array_equal(result["input_data"][8:9], np.full((1, 2, 2, 2), 77.0, dtype=np.float32))
    np.testing.assert_array_equal(result["input_data"][9:11], static_data["static_input_suffix"])

    expected_train_values = np.array(
        [
            [[1.0, 2.0], [0.0, 0.0]],
            [[5.0, 6.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    expected_train_availability = np.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(result["train_data"][0], expected_train_values)
    np.testing.assert_array_equal(result["val_data"][0, :, 1, 0], np.array([3.0, 7.0], dtype=np.float32))
    np.testing.assert_array_equal(result["test_data"][0, :, 1, 1], np.array([4.0, 8.0], dtype=np.float32))
    np.testing.assert_array_equal(result["input_data"][11], expected_train_values)
    np.testing.assert_array_equal(result["input_data"][12], expected_train_availability)


def test_collect_data_normalizes_traffic_over_valid_24h_values(monkeypatch):
    time_index = pd.date_range("2024-01-01 00:00:00", periods=2, freq="h")
    pollutant_data = _build_fake_pollutant_data()
    fake_backend = object()
    traffic_data = np.array(
        [
            [[10.0, 20.0], [0.0, 40.0]],
            [[30.0, 50.0], [0.0, 70.0]],
        ],
        dtype=np.float32,
    )[None, ...]
    traffic_mask = np.array(
        [
            [[True, True], [False, True]],
            [[True, True], [False, True]],
        ],
        dtype=bool,
    )[None, ...]

    monkeypatch.setattr(
        data_module,
        "get_grid",
        lambda **kwargs: ({"grid": np.zeros((2, 2), dtype=int)}, [10, 20, 30, 40]),
    )
    monkeypatch.setattr(data_module, "to_grid", _fake_to_grid)
    monkeypatch.setattr(
        data_module,
        "generate_pollutant_magnitudes",
        lambda **kwargs: (pollutant_data, time_index, [10, 20, 30, 40], None),
    )
    monkeypatch.setattr(
        data_module,
        "get_traffic_grid",
        lambda **kwargs: (traffic_data.copy(), traffic_mask.copy(), None, None),
    )

    result = data_module.collect_data(
        start_date=pd.Timestamp("2024-01-01 00:00:00"),
        end_date=pd.Timestamp("2024-01-01 01:00:00"),
        add_meteo=False,
        add_time_channels=False,
        add_coordinates=False,
        add_traffic_data=True,
        pollutants=[7],
        test_sensors=[],
        normalize=True,
        aq_backend=fake_backend,
    )

    expected = traffic_data.copy()
    valid = traffic_mask.astype(bool)
    mean = expected[valid].mean()
    std = expected[valid].std()
    expected[valid] = (expected[valid] - mean) / (std + 1e-6)

    np.testing.assert_allclose(result["static_input_prefix"], expected)


def test_collect_ensemble_data_reuses_static_pollutant_normalization_stats(monkeypatch):
    time_index = pd.date_range("2024-01-01 00:00:00", periods=2, freq="h")
    pollutant_data = _build_fake_pollutant_data()
    fake_backend = object()

    monkeypatch.setattr(
        data_module,
        "get_grid",
        lambda **kwargs: ({"grid": np.zeros((2, 2), dtype=int)}, [10, 20, 30, 40]),
    )
    monkeypatch.setattr(data_module, "to_grid", _fake_to_grid)
    monkeypatch.setattr(
        data_module,
        "generate_pollutant_magnitudes",
        lambda **kwargs: (pollutant_data, time_index, [10, 20, 30, 40], None),
    )
    monkeypatch.setattr(
        data_module,
        "generate_noise_channels",
        lambda **kwargs: np.zeros((kwargs["number_of_channels"], kwargs["hours"], kwargs["rows"], kwargs["cols"]), dtype=np.float32),
    )

    static_data = data_module.collect_data(
        start_date=pd.Timestamp("2024-01-01 00:00:00"),
        end_date=pd.Timestamp("2024-01-01 01:00:00"),
        add_meteo=False,
        add_time_channels=False,
        add_coordinates=False,
        add_traffic_data=False,
        pollutants=[7],
        test_sensors=[40],
        normalize=True,
        aq_backend=fake_backend,
    )

    assert static_data["pollutant_norm_stats"] is not None
    expected_stats = static_data["pollutant_norm_stats"][7]
    expected_mean = np.array([5.0, 6.0, 7.0], dtype=np.float32).mean()
    expected_std = np.array([5.0, 6.0, 7.0], dtype=np.float32).std()
    expected_test_data = np.zeros((1, 2, 2, 2), dtype=np.float32)
    expected_test_data[0, :, 1, 1] = (np.array([4.0, 8.0], dtype=np.float32) - expected_mean) / (expected_std + 1e-6)

    assert static_data["pollutant_norm_channels"] is not None
    assert static_data["pollutant_norm_channels"].shape == (2, 2, 2)
    np.testing.assert_allclose(expected_stats, (expected_mean, expected_std))
    np.testing.assert_allclose(static_data["test_data"], expected_test_data)

    monkeypatch.setattr(data_module, "get_random_sensors", lambda **kwargs: (np.array([10, 20]), np.array([30]), np.array([], dtype=int)))
    result_one = data_module.collect_ensemble_data(
        data=static_data,
        number_of_noise_channels=1,
        number_of_val_sensors=1,
        add_distance_to_sensors=False,
        normalize=True,
        aq_backend=fake_backend,
    )

    monkeypatch.setattr(data_module, "get_random_sensors", lambda **kwargs: (np.array([10, 30]), np.array([20]), np.array([], dtype=int)))
    result_two = data_module.collect_ensemble_data(
        data=static_data,
        number_of_noise_channels=1,
        number_of_val_sensors=1,
        add_distance_to_sensors=False,
        normalize=True,
        aq_backend=fake_backend,
    )

    assert result_one["normalization_stats"][7] == expected_stats
    assert result_two["normalization_stats"][7] == expected_stats
    assert result_one["input_data"].shape == (5, 2, 2, 2)
    np.testing.assert_allclose(result_one["input_data"][1], np.full((2, 2, 2), expected_mean, dtype=np.float32))
    np.testing.assert_allclose(result_one["input_data"][2], np.full((2, 2, 2), expected_std, dtype=np.float32))
    np.testing.assert_allclose(result_one["test_data"], static_data["test_data"])
    np.testing.assert_allclose(result_two["test_data"], static_data["test_data"])


@pytest.mark.parametrize("normalize,edge_cases", [(False, False), (True, False), (True, True)])
def test_generate_meteo_magnitudes_aligns_all_channels_to_requested_sensor_ids(monkeypatch, normalize, edge_cases):
    time_index = pd.date_range("2024-01-01 00:00:00", periods=2, freq="h")
    sensor_ids = [10, 20]

    values = {
        83: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        86: np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        87: np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
        88: np.array([[13.0, 14.0], [15.0, 16.0]], dtype=np.float32),
        89: np.array([[17.0, 18.0], [19.0, 20.0]], dtype=np.float32),
    }
    masks = {mag_id: np.ones_like(data, dtype=np.float32) for mag_id, data in values.items()}
    if edge_cases:
        values[83] = np.array([[1.0, 9999.0], [3.0, np.nan]], dtype=np.float32)
        masks[83] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        values[86][:] = 50.0
        masks[87][:] = 0.0
        values[88][0, 0] = np.inf

    monkeypatch.setattr(
        data_module,
        "get_magnitudes_data",
        lambda **kwargs: (values, masks, time_index, sensor_ids, None),
    )
    monkeypatch.setattr(
        data_module,
        "get_data",
        lambda **kwargs: (
            pd.DataFrame(
                {
                    "sensor_id": [10, 10, 20, 20, 10, 10, 20, 20],
                    "entry_date": list(time_index) * 4,
                    "magnitude_id": [81, 81, 81, 81, 82, 82, 82, 82],
                    "value": [1.0, 2.0, 3.0, 4.0, 90.0, 180.0, 270.0, 0.0],
                }
            ),
            time_index,
        ),
    )
    monkeypatch.setattr(data_module, "to_grid", _fake_to_grid)

    grid, returned_time_index, meteo_mags = data_module.generate_meteo_magnitudes(
        start_date=pd.Timestamp("2024-01-01 00:00:00"),
        end_date=pd.Timestamp("2024-01-01 01:00:00"),
        grid_ctx={"grid": np.zeros((2, 2), dtype=int)},
        sensor_ids=sensor_ids,
        aq_backend=object(),
        normalize=normalize,
    )

    assert list(returned_time_index) == list(time_index)
    assert meteo_mags == [811, 812, 83, 86, 87, 88, 89]
    assert grid.shape == (14, 2, 2, 2)
    expected_values = {
        811: np.array([[-1.0, 3.0], [0.0, 0.0]]),
        812: np.array([[0.0, 0.0], [2.0, -4.0]]),
        **values,
    }
    for channel, mag_id in enumerate(meteo_mags):
        expected = expected_values[mag_id].copy()
        mask = masks.get(mag_id, np.ones((2, 2)))
        if normalize:
            valid = np.isfinite(expected) & np.isfinite(mask) & (mask > 0)
            observed = expected[valid]
            expected = np.zeros((2, 2))
            if observed.size:
                scale = observed.std() if observed.std() >= 1e-6 else 1.0
                expected[valid] = (observed - observed.mean()) / scale
            mask = valid.astype(np.float32)
        np.testing.assert_allclose(grid[2 * channel, :, 0, :], expected, atol=1e-6)
        np.testing.assert_array_equal(grid[2 * channel + 1, :, 0, :], mask)
    assert np.isfinite(grid).all()
