from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metraq_dip.data import traffic_data as traffic_module


def _patch_mapping(monkeypatch, *, rows, cols, mapped):
    monkeypatch.setattr(
        traffic_module.pd,
        "read_sql_query",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "id": [10, 20, 30],
                "utm_x": [0.0, 0.0, 0.0],
                "utm_y": [0.0, 0.0, 0.0],
            }
        ),
    )
    monkeypatch.setattr(
        traffic_module,
        "map_sensor_ids_to_grid",
        lambda *args, **kwargs: (
            np.asarray(rows),
            np.asarray(cols),
            np.asarray(mapped),
        ),
    )


def test_to_grid_averages_collocated_available_sensors(monkeypatch):
    _patch_mapping(
        monkeypatch,
        rows=[0, 0, 1],
        cols=[0, 0, 1],
        mapped=[True, True, True],
    )
    data = np.array(
        [
            [[10.0, 30.0, 50.0], [20.0, 999.0, 60.0]],
            [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    result = traffic_module.to_grid(
        data=data,
        sensor_ids=[10, 20, 30],
        grid_ctx={"grid": np.zeros((2, 2), dtype=int)},
    )

    expected_values = np.array(
        [[[20.0, 0.0], [0.0, 50.0]], [[20.0, 0.0], [0.0, 60.0]]],
        dtype=np.float32,
    )
    expected_mask = np.array(
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result[0], expected_values)
    np.testing.assert_array_equal(result[1], expected_mask)


def test_to_grid_ignores_unmapped_sensors(monkeypatch):
    _patch_mapping(
        monkeypatch,
        rows=[0, 0, 1],
        cols=[0, 1, 1],
        mapped=[True, False, True],
    )
    data = np.array(
        [[[10.0, 20.0, 30.0]], [[1.0, 1.0, 1.0]]],
        dtype=np.float32,
    )

    result = traffic_module.to_grid(
        data=data,
        sensor_ids=[10, 20, 30],
        grid_ctx={"grid": np.zeros((2, 2), dtype=int)},
    )

    np.testing.assert_array_equal(result[0, 0], [[10.0, 0.0], [0.0, 30.0]])
    np.testing.assert_array_equal(result[1, 0], [[1.0, 0.0], [0.0, 1.0]])


def test_to_grid_rejects_invalid_channel_shape(monkeypatch):
    _patch_mapping(
        monkeypatch,
        rows=[0, 0, 1],
        cols=[0, 1, 1],
        mapped=[True, True, True],
    )

    with pytest.raises(ValueError, match="shape"):
        traffic_module.to_grid(
            data=np.ones((1, 2, 3), dtype=np.float32),
            sensor_ids=[10, 20, 30],
            grid_ctx={"grid": np.zeros((2, 2), dtype=int)},
        )
