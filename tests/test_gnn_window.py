from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from metraq_gnn.data.graph import build_grid_graph
from metraq_gnn.data.window import build_graph_window, split_training_sensor_masks


def _pollutant_window() -> np.ndarray:
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    availability = np.ones_like(values)
    return np.stack((values, availability))


def test_split_training_masks_hides_only_eligible_train_cells():
    train = torch.tensor([[True, True], [True, False]])
    final_observations = torch.tensor([[True, False], [True, True]])
    generator = torch.Generator().manual_seed(7)

    context, target = split_training_sensor_masks(
        train,
        final_observations,
        target_fraction=0.5,
        generator=generator,
    )

    assert torch.equal(context | target, train)
    assert not torch.any(context & target)
    assert target.sum() == 1
    assert not target[0, 1]
    assert not target[1, 1]


def test_split_training_masks_keeps_at_least_one_context_sensor():
    train = torch.tensor([[True, True]])

    context, target = split_training_sensor_masks(
        train,
        torch.ones_like(train),
        target_fraction=0.99,
        generator=torch.Generator().manual_seed(1),
    )

    assert context.sum() == 1
    assert target.sum() == 1


def test_build_graph_window_exposes_only_context_air_quality():
    graph = build_grid_graph({"grid": np.zeros((2, 2), dtype=object)})
    context = np.array([[True, True], [False, False]])
    target = np.array([[False, False], [True, False]])
    extras = np.full((1, 2, 2, 2), 9.0, dtype=np.float32)

    sample = build_graph_window(
        graph=graph,
        pollutant_data=_pollutant_window(),
        context_sensor_mask=context,
        target_sensor_mask=target,
        extra_features=extras,
    )

    assert sample.x.shape == (4, 2, 3)
    assert sample.y.shape == (4, 1)
    assert sample.target_mask.shape == (4, 1)
    torch.testing.assert_close(sample.x[0, :, 0], torch.tensor([1.0, 5.0]))
    torch.testing.assert_close(sample.x[1, :, 0], torch.tensor([2.0, 6.0]))
    torch.testing.assert_close(sample.x[2:, :, 0], torch.zeros((2, 2)))
    torch.testing.assert_close(sample.x[:, :, 1], sample.context_mask[:, :, 0].float())
    torch.testing.assert_close(sample.x[:, :, 2], torch.full((4, 2), 9.0))
    assert sample.target_mask[:, 0].tolist() == [False, False, True, False]
    assert sample.y[:, 0].tolist() == [5.0, 6.0, 7.0, 8.0]
    assert torch.equal(sample.edge_index, graph.edge_index)
    assert torch.equal(sample.edge_attr, graph.edge_attr)


def test_target_and_unselected_nodes_have_no_aq_history_in_features():
    graph = build_grid_graph({"grid": np.zeros((2, 2), dtype=object)})
    sample = build_graph_window(
        graph=graph,
        pollutant_data=_pollutant_window(),
        context_sensor_mask=np.array([[True, False], [False, False]]),
        target_sensor_mask=np.array([[False, True], [False, False]]),
    )

    torch.testing.assert_close(sample.x[1:, :, :2], torch.zeros((3, 2, 2)))


def test_build_graph_window_rejects_context_target_overlap():
    graph = build_grid_graph({"grid": np.zeros((2, 2), dtype=object)})
    shared = np.array([[True, False], [False, False]])

    with pytest.raises(ValueError, match="must be disjoint"):
        build_graph_window(
            graph=graph,
            pollutant_data=_pollutant_window(),
            context_sensor_mask=shared,
            target_sensor_mask=shared,
        )


def test_build_graph_window_rejects_forbidden_context_cells():
    graph = build_grid_graph({"grid": np.zeros((2, 2), dtype=object)})

    with pytest.raises(ValueError, match="must not appear in context"):
        build_graph_window(
            graph=graph,
            pollutant_data=_pollutant_window(),
            context_sensor_mask=np.array([[True, False], [False, False]]),
            target_sensor_mask=np.array([[False, True], [False, False]]),
            forbidden_context_mask=np.array([[True, False], [False, True]]),
        )


def test_build_graph_window_requires_available_target_value():
    graph = build_grid_graph({"grid": np.zeros((2, 2), dtype=object)})
    pollution = _pollutant_window()
    pollution[1, -1, 1, 0] = 0.0

    with pytest.raises(ValueError, match="no valid final-timestep"):
        build_graph_window(
            graph=graph,
            pollutant_data=pollution,
            context_sensor_mask=np.array([[True, False], [False, False]]),
            target_sensor_mask=np.array([[False, False], [True, False]]),
        )


def test_graph_windows_batch_with_offset_edges():
    graph = build_grid_graph({"grid": np.zeros((2, 2), dtype=object)})
    kwargs = {
        "graph": graph,
        "pollutant_data": _pollutant_window(),
        "context_sensor_mask": np.array([[True, False], [False, False]]),
        "target_sensor_mask": np.array([[False, True], [False, False]]),
        "forbidden_context_mask": np.array([[False, True], [False, True]]),
    }
    samples = [build_graph_window(**kwargs), build_graph_window(**kwargs)]

    batch = next(iter(DataLoader(samples, batch_size=2, shuffle=False)))

    assert batch.x.shape == (8, 2, 2)
    assert batch.y.shape == (8, 1)
    assert batch.edge_index.shape == (2, 32)
    assert batch.edge_index[:, :16].max() == 3
    assert batch.edge_index[:, 16:].min() == 4
    assert batch.batch.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
