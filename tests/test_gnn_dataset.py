from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from metraq_gnn.data import GraphWindowDataset, build_grid_graph


def _series(*, periods: int = 8):
    time_index = pd.date_range("2023-01-01", periods=periods, freq="h")
    values = np.arange(periods * 6, dtype=np.float32).reshape(periods, 2, 3)
    availability = np.ones_like(values)
    pollution = np.stack((values, availability))
    extras = np.ones((1, periods, 2, 3), dtype=np.float32)
    graph = build_grid_graph({"grid": np.zeros((2, 3), dtype=object)})
    return graph, pollution, extras, time_index


def test_dataset_indexes_only_complete_windows_inside_split():
    graph, pollution, extras, time_index = _series()
    dataset = GraphWindowDataset(
        graph=graph,
        pollutant_data=pollution,
        extra_features=extras,
        time_index=time_index,
        context_sensor_mask=np.array([[True, True, True], [True, False, False]]),
        forbidden_context_mask=np.array([[False, False, False], [False, True, True]]),
        target_sensor_mask=np.array([[False, False, False], [False, True, False]]),
        hours=3,
        split_start="2023-01-01 02:00",
        split_end="2023-01-01 06:00",
    )

    assert len(dataset) == 3
    assert dataset.window_end_times.tolist() == time_index[4:7].tolist()
    sample = dataset[0]
    assert sample.x.shape == (6, 3, 3)
    assert pd.Timestamp(sample.window_start_ns.item()) == time_index[2]
    assert pd.Timestamp(sample.window_end_ns.item()) == time_index[4]


def test_dataset_excludes_windows_crossing_a_time_gap():
    graph, pollution, _, time_index = _series(periods=6)
    gapped_index = time_index.delete(3)
    pollution = np.delete(pollution, 3, axis=1)
    dataset = GraphWindowDataset(
        graph=graph,
        pollutant_data=pollution,
        time_index=gapped_index,
        context_sensor_mask=np.array([[True, True, True], [True, False, False]]),
        forbidden_context_mask=np.array([[False, False, False], [False, True, True]]),
        target_sensor_mask=np.array([[False, False, False], [False, True, False]]),
        hours=2,
        split_start=gapped_index[0],
        split_end=gapped_index[-1],
    )

    assert dataset.window_end_times.tolist() == [time_index[1], time_index[2], time_index[5]]


def test_training_masks_are_reproducible_and_change_by_epoch():
    graph, pollution, _, time_index = _series()
    dataset = GraphWindowDataset(
        graph=graph,
        pollutant_data=pollution,
        time_index=time_index,
        context_sensor_mask=np.array([[True, True, True], [True, True, True]]),
        forbidden_context_mask=np.zeros((2, 3), dtype=bool),
        target_fraction=0.5,
        hours=3,
        split_start=time_index[0],
        split_end=time_index[-1],
        seed=9,
    )

    first = dataset[0].target_sensor_mask.clone()
    assert torch.equal(first, dataset[0].target_sensor_mask)
    dataset.set_epoch(1)
    second = dataset[0].target_sensor_mask

    assert first.sum() == 3
    assert second.sum() == 3
    assert not torch.equal(first, second)


def test_fixed_target_dataset_filters_unobserved_target_windows():
    graph, pollution, _, time_index = _series(periods=5)
    pollution[1, -1, 1, 1] = 0.0
    dataset = GraphWindowDataset(
        graph=graph,
        pollutant_data=pollution,
        time_index=time_index,
        context_sensor_mask=np.array([[True, True, True], [True, False, False]]),
        forbidden_context_mask=np.array([[False, False, False], [False, True, True]]),
        target_sensor_mask=np.array([[False, False, False], [False, True, False]]),
        hours=2,
        split_start=time_index[0],
        split_end=time_index[-1],
    )

    assert dataset.window_end_times.tolist() == time_index[1:-1].tolist()


def test_dataset_batches_windows_and_offsets_edges():
    graph, pollution, _, time_index = _series(periods=5)
    dataset = GraphWindowDataset(
        graph=graph,
        pollutant_data=pollution,
        time_index=time_index,
        context_sensor_mask=np.array([[True, True, True], [True, False, False]]),
        forbidden_context_mask=np.array([[False, False, False], [False, True, True]]),
        target_sensor_mask=np.array([[False, False, False], [False, True, False]]),
        hours=2,
        split_start=time_index[0],
        split_end=time_index[-1],
    )

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert batch.x.shape == (12, 2, 2)
    assert batch.edge_index[:, : graph.edge_index.shape[1]].max() == 5
    assert batch.edge_index[:, graph.edge_index.shape[1] :].min() == 6


def test_dataset_rejects_context_from_forbidden_cells():
    graph, pollution, _, time_index = _series()

    with pytest.raises(ValueError, match="must not appear in context"):
        GraphWindowDataset(
            graph=graph,
            pollutant_data=pollution,
            time_index=time_index,
            context_sensor_mask=np.array([[True, False, False], [False, False, False]]),
            forbidden_context_mask=np.array([[True, False, False], [False, False, False]]),
            target_fraction=0.5,
            hours=3,
            split_start=time_index[0],
            split_end=time_index[-1],
        )
