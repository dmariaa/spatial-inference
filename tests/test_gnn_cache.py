from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from metraq_gnn.data import (
    SensorGraphWindowDataset,
    build_aq_sensor_cache,
    compute_training_normalization,
    load_aq_sensor_cache,
    split_sensor_nodes,
)


class FakeAQBackend:
    dataset_name = "fake"
    backend_name = "memory"

    def __init__(self) -> None:
        self.measurement_calls = 0

    def get_sensors(self, *, magnitudes=None, sensors=None):
        frame = pd.DataFrame(
            {
                "id": [10, 20, 30, 40],
                "utm_x": [440100.0, 440200.0, 441100.0, 442100.0],
                "utm_y": [4470100.0, 4470200.0, 4470100.0, 4470100.0],
            }
        )
        return frame if sensors is None else frame[frame["id"].isin(sensors)]

    def get_measurements(self, *, start_date, end_date, magnitudes):
        self.measurement_calls += 1
        timestamps = pd.date_range(start_date, end_date, freq="h")
        rows = []
        for offset, timestamp in enumerate(timestamps):
            rows.extend(
                [
                    (10, timestamp, 8, float(offset + 1)),
                    (20, timestamp, 8, float(offset + 3)),
                    (30, timestamp, 8, float(offset + 5)),
                    (40, timestamp, 8, float(offset + 7)),
                ]
            )
        return pd.DataFrame(rows, columns=["sensor_id", "entry_date", "magnitude_id", "value"])


def test_sensor_cache_builds_resumes_and_memory_maps(tmp_path):
    backend = FakeAQBackend()
    path = tmp_path / "aq-cache"
    cache = build_aq_sensor_cache(
        path=path,
        aq_backend=backend,
        start="2023-12-31 22:00",
        end="2024-01-01 02:00",
        magnitudes=[8],
        cell_size_m=1000,
        margin_m_x=0,
        margin_m_y=0,
    )

    assert backend.measurement_calls == 2
    assert cache.values.shape == (5, 4, 1)
    assert cache.availability.all()
    assert isinstance(cache.values, np.memmap)
    assert cache.metadata["completed_years"] == [2023, 2024]

    resumed = build_aq_sensor_cache(
        path=path,
        aq_backend=backend,
        start="2023-12-31 22:00",
        end="2024-01-01 02:00",
        magnitudes=[8],
        cell_size_m=1000,
        margin_m_x=0,
        margin_m_y=0,
    )
    assert backend.measurement_calls == 2
    assert load_aq_sensor_cache(path).values.shape == resumed.values.shape


def test_sensor_dataset_aggregates_colocated_sensors_only_for_window(tmp_path):
    backend = FakeAQBackend()
    cache = build_aq_sensor_cache(
        path=tmp_path / "aq-cache",
        aq_backend=backend,
        start="2024-01-01 00:00",
        end="2024-01-01 04:00",
        magnitudes=[8],
        cell_size_m=1000,
        margin_m_x=0,
        margin_m_y=0,
    )
    assert cache.sensor_node_indices[0] == cache.sensor_node_indices[1]
    node_mask = np.zeros(cache.grid_shape, dtype=bool)
    node_mask.reshape(-1)[np.unique(cache.sensor_node_indices)] = True
    dataset = SensorGraphWindowDataset(
        cache=cache,
        context_sensor_mask=node_mask,
        forbidden_context_mask=np.zeros(cache.grid_shape, dtype=bool),
        target_fraction=0.5,
        hours=3,
        split_start="2024-01-01 00:00",
        split_end="2024-01-01 04:00",
        seed=7,
    )

    sample = dataset[0]
    colocated_node = int(cache.sensor_node_indices[0])
    # At t=0 the two colocated values are 1 and 3, hence their cell mean is 2.
    if sample.context_sensor_mask[colocated_node]:
        assert torch.isclose(sample.x[colocated_node, 0, 0], torch.tensor(2.0))
    assert sample.x.shape[1] == 3
    assert sample.target_mask.any()


def test_spatial_splits_and_normalization_use_only_train_nodes(tmp_path):
    backend = FakeAQBackend()
    cache = build_aq_sensor_cache(
        path=tmp_path / "aq-cache",
        aq_backend=backend,
        start="2024-01-01 00:00",
        end="2024-01-01 04:00",
        magnitudes=[8],
        cell_size_m=1000,
        margin_m_x=0,
        margin_m_y=0,
    )
    train, validation, test = split_sensor_nodes(
        cache, validation_nodes=1, test_nodes=1, seed=3
    )
    assert train.sum() == validation.sum() == test.sum() == 1
    assert not np.any(train & validation)
    assert not np.any(train & test)
    mean, std = compute_training_normalization(
        cache,
        train_node_mask=train,
        start="2024-01-01 00:00",
        end="2024-01-01 04:00",
    )
    assert mean.shape == std.shape == (1,)
    assert std[0] > 0


def test_spatial_split_accepts_explicit_test_sensor_ids(tmp_path):
    cache = build_aq_sensor_cache(
        path=tmp_path / "aq-cache",
        aq_backend=FakeAQBackend(),
        start="2024-01-01 00:00",
        end="2024-01-01 04:00",
        magnitudes=[8],
        cell_size_m=1000,
        margin_m_x=0,
        margin_m_y=0,
    )
    requested = [int(cache.sensor_ids[-1])]

    train, validation, test = split_sensor_nodes(
        cache,
        validation_nodes=1,
        test_nodes=99,
        test_sensor_ids=requested,
        seed=42,
    )

    expected_nodes = {int(cache.sensor_node_indices[-1])}
    assert set(np.flatnonzero(test)) == expected_nodes
    assert not np.any(train & validation)
    assert not np.any(train & test)
    assert not np.any(validation & test)


def test_sensor_dataset_appends_cyclical_time_features(tmp_path):
    cache = build_aq_sensor_cache(
        path=tmp_path / "aq-cache",
        aq_backend=FakeAQBackend(),
        start="2024-01-01 00:00",
        end="2024-01-01 04:00",
        magnitudes=[8],
        cell_size_m=1000,
        margin_m_x=0,
        margin_m_y=0,
    )
    node_mask = np.zeros(cache.grid_shape, dtype=bool)
    node_mask.reshape(-1)[np.unique(cache.sensor_node_indices)] = True
    dataset = SensorGraphWindowDataset(
        cache=cache,
        context_sensor_mask=node_mask,
        forbidden_context_mask=np.zeros(cache.grid_shape, dtype=bool),
        target_fraction=0.5,
        hours=3,
        split_start="2024-01-01 00:00",
        split_end="2024-01-01 04:00",
        include_time_features=True,
        seed=7,
    )

    sample = dataset[0]

    assert sample.x.shape[-1] == 6  # AQ value, availability, and four time channels.
    torch.testing.assert_close(sample.x[:, 0, 2], torch.zeros(sample.num_nodes))
    torch.testing.assert_close(sample.x[:, 0, 3], torch.ones(sample.num_nodes))
    expected_hour_sin = torch.full((sample.num_nodes,), 0.5)
    torch.testing.assert_close(sample.x[:, 2, 2], expected_hour_sin)
