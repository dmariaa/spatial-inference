from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from metraq_gnn.data.graph import build_grid_graph
from metraq_gnn.data.window import build_graph_window
from metraq_gnn.model import SensorToGridGNN, SpatioTemporalGNN


def _sample():
    graph = build_grid_graph({"grid": np.zeros((2, 2), dtype=object)})
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    pollution = np.stack((values, np.ones_like(values)))
    extras = np.full((1, 3, 2, 2), 0.5, dtype=np.float32)
    return build_graph_window(
        graph=graph,
        pollutant_data=pollution,
        context_sensor_mask=np.array([[True, True], [False, False]]),
        target_sensor_mask=np.array([[False, False], [True, False]]),
        forbidden_context_mask=np.array([[False, False], [True, True]]),
        extra_features=extras,
    )


def _model(*, dropout: float = 0.0) -> SpatioTemporalGNN:
    return SpatioTemporalGNN(
        input_channels=3,
        output_channels=1,
        edge_channels=4,
        temporal_hidden_channels=8,
        graph_hidden_channels=8,
        graph_layers=2,
        attention_heads=2,
        dropout=dropout,
    )


def _sensor_to_grid_model(*, local_refinement_layers: int = 1) -> SensorToGridGNN:
    return SensorToGridGNN(
        input_channels=3,
        output_channels=1,
        edge_channels=4,
        temporal_hidden_channels=8,
        graph_hidden_channels=8,
        attention_heads=2,
        local_refinement_layers=local_refinement_layers,
        dropout=0.0,
    )


def test_model_predicts_one_value_per_node():
    sample = _sample()

    prediction = _model()(sample)

    assert prediction.shape == (4, 1)
    assert torch.isfinite(prediction).all()


def test_model_accepts_batched_graph_windows():
    sample = _sample()
    batch = next(iter(DataLoader([sample, sample], batch_size=2, shuffle=False)))

    prediction = _model()(batch)

    assert prediction.shape == (8, 1)
    assert torch.isfinite(prediction).all()


def test_attention_is_normalized_over_incoming_edges():
    sample = _sample()
    model = _model()
    model.eval()

    _, attention_history = model(sample, return_attention_weights=True)

    assert len(attention_history) == 2
    for attention in attention_history:
        assert attention.edge_index.shape == sample.edge_index.shape
        assert attention.alpha.shape == (sample.edge_index.shape[1], 2)
        targets = attention.edge_index[1]
        for target in range(sample.num_nodes):
            incoming = attention.alpha[targets == target]
            torch.testing.assert_close(incoming.sum(dim=0), torch.ones(2))


def test_masked_node_loss_trains_attention_parameters():
    torch.manual_seed(3)
    sample = _sample()
    model = _model()

    prediction = model(sample)
    loss = torch.abs(prediction[sample.target_mask] - sample.y[sample.target_mask]).mean()
    loss.backward()

    attention_gradient = model.graph_convolutions[0].att.grad
    assert attention_gradient is not None
    assert torch.isfinite(attention_gradient).all()
    assert torch.count_nonzero(attention_gradient) > 0


def test_model_rejects_wrong_feature_count():
    sample = _sample()
    sample.x = sample.x[..., :2]

    with pytest.raises(ValueError, match="features, expected 3"):
        _model()(sample)


def test_model_requires_hidden_channels_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        SpatioTemporalGNN(
            input_channels=3,
            output_channels=1,
            edge_channels=4,
            graph_hidden_channels=7,
            attention_heads=2,
        )


def test_sensor_to_grid_model_predicts_and_normalizes_global_attention():
    sample = _sample()
    model = _sensor_to_grid_model()
    model.eval()

    prediction, attention = model(sample, return_attention_weights=True)

    assert prediction.shape == (4, 1)
    assert attention.edge_index.shape == sample.sensor_edge_index.shape
    targets = attention.edge_index[1]
    for target in range(sample.num_nodes):
        incoming = attention.alpha[targets == target]
        torch.testing.assert_close(incoming.sum(dim=0), torch.ones(2))


def test_sensor_to_grid_model_accepts_batched_dynamic_edges():
    sample = _sample()
    batch = next(iter(DataLoader([sample, sample], batch_size=2, shuffle=False)))

    prediction = _sensor_to_grid_model()(batch)

    assert prediction.shape == (8, 1)
    first_edges = sample.sensor_edge_index.shape[1]
    assert batch.sensor_edge_index[:, first_edges:].min() >= sample.num_nodes


def test_sensor_to_grid_model_can_disable_local_refinement():
    sample = _sample()
    model = _sensor_to_grid_model(local_refinement_layers=0)

    prediction, attention = model(sample, return_attention_weights=True)

    assert prediction.shape == (4, 1)
    assert attention.alpha.shape == (sample.sensor_edge_index.shape[1], 2)
    assert len(model.local_convolutions) == 0
