from __future__ import annotations

import numpy as np
import pandas as pd

from metraq_dip.data import data as data_module


def _fake_to_grid(*, data: np.ndarray, sensor_ids: list[int], grid_ctx: dict):
    mapping = {10: (0, 0), 20: (0, 1), 30: (1, 0), 40: (1, 1)}
    channels, timestamps, _ = data.shape
    rows, cols = grid_ctx["grid"].shape
    grid = np.zeros((channels, timestamps, rows, cols), dtype=np.float32)

    for sensor_idx, sensor_id in enumerate(sensor_ids):
        row, col = mapping[int(sensor_id)]
        grid[:, :, row, col] = data[:, :, sensor_idx]

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


def test_collect_data_returns_only_static_components(monkeypatch):
    time_index = pd.date_range("2024-01-01 00:00:00", periods=2, freq="h")
    pollutant_data = _build_fake_pollutant_data()

    monkeypatch.setattr(data_module, "get_grid", lambda: ({"grid": np.zeros((2, 2), dtype=int)}, [10, 20, 30, 40]))
    monkeypatch.setattr(data_module, "to_grid", _fake_to_grid)
    monkeypatch.setattr(
        data_module,
        "generate_pollutant_magnitudes",
        lambda **kwargs: (pollutant_data, time_index, [10, 20, 30, 40], None),
    )
    monkeypatch.setattr(data_module, "generate_dimensions", lambda grid_ctx, hours: np.full((3, hours, 2, 2), 10.0, dtype=np.float32))
    monkeypatch.setattr(
        data_module,
        "generate_hour_of_day_coords",
        lambda time_index, rows, cols: np.full((2, len(time_index), rows, cols), 20.0, dtype=np.float32),
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
    )

    assert "input_data" not in result
    assert "train_data" not in result
    assert result["pollutant_data"].shape == (2, 2, 2, 2)
    assert result["static_input_prefix"].shape == (6, 2, 2, 2)
    assert result["static_input_suffix"].shape == (2, 2, 2, 2)
    assert result["test_mask"].dtype == np.bool_
    np.testing.assert_array_equal(result["test_mask"][0, 0], np.array([[False, False], [False, True]]))
    np.testing.assert_array_equal(result["test_data"][0, :, 1, 1], np.array([4.0, 8.0], dtype=np.float32))


def test_collect_ensemble_data_builds_dynamic_channels_from_static_data(monkeypatch):
    time_index = pd.date_range("2024-01-01 00:00:00", periods=2, freq="h")
    pollutant_data = _build_fake_pollutant_data()

    monkeypatch.setattr(data_module, "get_grid", lambda: ({"grid": np.zeros((2, 2), dtype=int)}, [10, 20, 30, 40]))
    monkeypatch.setattr(data_module, "to_grid", _fake_to_grid)
    monkeypatch.setattr(
        data_module,
        "generate_pollutant_magnitudes",
        lambda **kwargs: (pollutant_data, time_index, [10, 20, 30, 40], None),
    )
    monkeypatch.setattr(data_module, "generate_dimensions", lambda grid_ctx, hours: np.full((3, hours, 2, 2), 10.0, dtype=np.float32))
    monkeypatch.setattr(
        data_module,
        "generate_hour_of_day_coords",
        lambda time_index, rows, cols: np.full((2, len(time_index), rows, cols), 20.0, dtype=np.float32),
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
    )

    assert result["input_data"].shape == (13, 2, 2, 2)
    np.testing.assert_array_equal(result["train_mask"][0, 0], np.array([[True, True], [False, False]]))
    np.testing.assert_array_equal(result["val_mask"][0, 0], np.array([[False, False], [True, False]]))
    np.testing.assert_array_equal(result["test_mask"][0, 0], np.array([[False, False], [False, True]]))

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

    monkeypatch.setattr(data_module, "get_grid", lambda: ({"grid": np.zeros((2, 2), dtype=int)}, [10, 20, 30, 40]))
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

    monkeypatch.setattr(data_module, "get_grid", lambda: ({"grid": np.zeros((2, 2), dtype=int)}, [10, 20, 30, 40]))
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
    )

    assert static_data["pollutant_norm_stats"] is not None
    expected_stats = static_data["pollutant_norm_stats"][7]
    expected_mean = np.array([5.0, 6.0, 7.0], dtype=np.float32).mean()
    expected_std = np.array([5.0, 6.0, 7.0], dtype=np.float32).std()
    expected_test_data = np.zeros((1, 2, 2, 2), dtype=np.float32)
    expected_test_data[0, :, 1, 1] = (np.array([4.0, 8.0], dtype=np.float32) - expected_mean) / (expected_std + 1e-6)

    assert static_data["pollutant_norm_channels"] is not None
    np.testing.assert_allclose(expected_stats, (expected_mean, expected_std))
    np.testing.assert_allclose(static_data["test_data"], expected_test_data)

    monkeypatch.setattr(data_module, "get_random_sensors", lambda **kwargs: (np.array([10, 20]), np.array([30]), np.array([], dtype=int)))
    result_one = data_module.collect_ensemble_data(
        data=static_data,
        number_of_noise_channels=1,
        number_of_val_sensors=1,
        add_distance_to_sensors=False,
        normalize=True,
    )

    monkeypatch.setattr(data_module, "get_random_sensors", lambda **kwargs: (np.array([10, 30]), np.array([20]), np.array([], dtype=int)))
    result_two = data_module.collect_ensemble_data(
        data=static_data,
        number_of_noise_channels=1,
        number_of_val_sensors=1,
        add_distance_to_sensors=False,
        normalize=True,
    )

    assert result_one["normalization_stats"][7] == expected_stats
    assert result_two["normalization_stats"][7] == expected_stats
    np.testing.assert_allclose(result_one["test_data"], static_data["test_data"])
    np.testing.assert_allclose(result_two["test_data"], static_data["test_data"])
