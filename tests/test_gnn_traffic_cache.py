from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from metraq_gnn.data.traffic_cache import (
    _accumulate_frame,
    _query_partition_batch,
    compute_traffic_training_normalization,
    load_traffic_grid_cache,
    month_ranges,
)


def test_month_ranges_use_zero_padded_monthly_partitions() -> None:
    ranges = month_ranges("2015-01-15 00:00", "2015-03-02 23:00")

    assert ranges == [
        (pd.Timestamp("2015-01-15"), pd.Timestamp("2015-02-01"), "p201501"),
        (pd.Timestamp("2015-02-01"), pd.Timestamp("2015-03-01"), "p201502"),
        (pd.Timestamp("2015-03-01"), pd.Timestamp("2015-03-03"), "p201503"),
    ]


def test_accumulate_frame_averages_traffic_sensors_within_cell() -> None:
    frame = pd.DataFrame(
        {
            "sensor_id": [10, 20, 30, 10],
            "hour": pd.to_datetime(
                ["2015-01-01 00:00", "2015-01-01 00:00", "2015-01-01 00:00", "2015-01-01 01:00"]
            ),
            "traffic_intensity": [10.0, 30.0, 50.0, 20.0],
        }
    )
    sums = np.zeros((2, 4), dtype=np.float64)
    counts = np.zeros((2, 4), dtype=np.int32)

    _accumulate_frame(
        frame=frame,
        chunk_start=pd.Timestamp("2015-01-01 00:00"),
        sensor_nodes={10: 0, 20: 0, 30: 3},
        sums=sums,
        counts=counts,
    )

    observed = counts > 0
    values = np.divide(sums, counts, out=np.zeros_like(sums), where=observed)
    np.testing.assert_array_equal(values, [[20.0, 0.0, 0.0, 50.0], [20.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(observed, [[True, False, False, True], [True, False, False, False]])


def test_query_rejects_untrusted_partition_name() -> None:
    with pytest.raises(ValueError, match="invalid partition"):
        _query_partition_batch(
            object(),
            partition="p201501; DROP TABLE traffic_data",
            start=pd.Timestamp("2015-01-01"),
            end=pd.Timestamp("2015-02-01"),
            sensor_ids=np.array([10]),
        )


def test_load_traffic_grid_cache_uses_memory_mapped_arrays(tmp_path) -> None:
    metadata = {"grid_shape": [2, 2]}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(tmp_path / "timestamps.npy", np.array([pd.Timestamp("2015-01-01").value]))
    np.save(tmp_path / "traffic_intensity.npy", np.ones((1, 2, 2), dtype=np.float32))
    np.save(tmp_path / "traffic_availability.npy", np.ones((1, 2, 2), dtype=bool))

    cache = load_traffic_grid_cache(tmp_path)

    assert cache.grid_shape == (2, 2)
    assert cache.time_index.equals(pd.DatetimeIndex(["2015-01-01"]))
    assert isinstance(cache.values, np.memmap)
    assert isinstance(cache.availability, np.memmap)


def test_traffic_normalization_uses_only_available_train_values(tmp_path) -> None:
    metadata = {"grid_shape": [1, 2]}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    timestamps = pd.date_range("2022-12-31 22:00", periods=4, freq="h")
    np.save(tmp_path / "timestamps.npy", timestamps.asi8)
    np.save(
        tmp_path / "traffic_intensity.npy",
        np.array([[[10.0, 999.0]], [[20.0, 999.0]], [[1000.0, 1000.0]], [[1000.0, 1000.0]]], dtype=np.float32),
    )
    np.save(
        tmp_path / "traffic_availability.npy",
        np.array([[[True, False]], [[True, False]], [[True, True]], [[True, True]]]),
    )
    cache = load_traffic_grid_cache(tmp_path)

    mean, std = compute_traffic_training_normalization(
        cache, start="2022-12-31 22:00", end="2022-12-31 23:00"
    )

    assert mean == 15.0
    assert std == 5.0
