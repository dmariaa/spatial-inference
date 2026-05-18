from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metraq_dip.tools import grid as grid_module


def _patch_simple_sensor_mapping(monkeypatch):
    class FakeBackend:
        def get_sensors(self, *, sensors):
            return pd.DataFrame({"id": sensors})

    rows = np.array([0, 0, 1], dtype=np.int64)
    cols = np.array([0, 1, 1], dtype=np.int64)
    mapped = np.array([True, False, True])
    monkeypatch.setattr(
        grid_module,
        "map_sensor_ids_to_grid",
        lambda *args, **kwargs: (rows, cols, mapped),
    )
    return FakeBackend()


def test_to_grid_maps_sensor_vector_to_spatial_grid(monkeypatch):
    backend = _patch_simple_sensor_mapping(monkeypatch)

    grid = grid_module.to_grid(
        data=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        sensor_ids=[10, 20, 30],
        grid_ctx={"grid": np.zeros((2, 2), dtype=int)},
        aq_backend=backend,
    )

    expected = np.array([[10.0, 0.0], [0.0, 30.0]], dtype=np.float32)
    assert grid.shape == (2, 2)
    np.testing.assert_array_equal(grid, expected)


def test_to_grid_preserves_leading_dimensions(monkeypatch):
    backend = _patch_simple_sensor_mapping(monkeypatch)
    data = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)

    grid = grid_module.to_grid(
        data=data,
        sensor_ids=[10, 20, 30],
        grid_ctx={"grid": np.zeros((2, 2), dtype=int)},
        aq_backend=backend,
    )

    assert grid.shape == (2, 4, 2, 2)
    np.testing.assert_array_equal(grid[..., 0, 0], data[..., 0])
    np.testing.assert_array_equal(grid[..., 1, 1], data[..., 2])
    np.testing.assert_array_equal(grid[..., 0, 1], np.zeros((2, 4), dtype=np.float32))


def test_to_grid_maps_channel_sensor_matrix(monkeypatch):
    backend = _patch_simple_sensor_mapping(monkeypatch)
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    grid = grid_module.to_grid(
        data=data,
        sensor_ids=[10, 20, 30],
        grid_ctx={"grid": np.zeros((2, 2), dtype=int)},
        aq_backend=backend,
    )

    assert grid.shape == (2, 2, 2)
    np.testing.assert_array_equal(grid[:, 0, 0], data[:, 0])
    np.testing.assert_array_equal(grid[:, 1, 1], data[:, 2])


def test_to_grid_rejects_sensor_id_length_mismatch(monkeypatch):
    backend = _patch_simple_sensor_mapping(monkeypatch)

    with pytest.raises(ValueError, match="sensor_ids length"):
        grid_module.to_grid(
            data=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            sensor_ids=[10, 20],
            grid_ctx={"grid": np.zeros((2, 2), dtype=int)},
            aq_backend=backend,
        )
