from __future__ import annotations

import numpy as np
import pytest
import torch

from metraq_gnn.data.graph import (
    build_grid_graph,
    build_knn_sensor_to_grid_edges,
    build_sensor_to_grid_edges,
    grid_to_nodes,
    nodes_to_grid,
)


def test_knn_sensor_edges_restrict_candidates_per_destination():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[2, 2] = True

    edge_index, edge_attr = build_knn_sensor_to_grid_edges(mask, k=1)

    assert edge_index.shape == (2, 9)
    assert edge_attr.shape == (9, 4)
    assert torch.equal(edge_index[1], torch.arange(9))
    assert edge_index[0, 0].item() == 0
    assert edge_index[0, -1].item() == 8


@pytest.mark.parametrize("array_type", [np.asarray, torch.as_tensor])
def test_grid_node_round_trip_preserves_leading_dimensions(array_type):
    original = array_type(np.arange(2 * 3 * 4).reshape(2, 3, 4))

    nodes = grid_to_nodes(original)
    restored = nodes_to_grid(nodes, (3, 4))

    assert nodes.shape == (2, 12)
    if isinstance(original, torch.Tensor):
        torch.testing.assert_close(restored, original)
    else:
        np.testing.assert_array_equal(restored, original)


def test_nodes_to_grid_rejects_wrong_number_of_nodes():
    with pytest.raises(ValueError, match="expected 6"):
        nodes_to_grid(np.zeros(5), (2, 3))


def test_two_by_two_grid_connects_every_node_to_every_node():
    graph = build_grid_graph({"grid": np.zeros((2, 2), dtype=object)})

    assert graph.num_nodes == 4
    assert graph.grid_shape == (2, 2)
    assert graph.edge_index.shape == (2, 16)
    assert graph.edge_attr.shape == (16, 4)
    assert set(map(tuple, graph.edge_index.T.tolist())) == {
        (source, target) for source in range(4) for target in range(4)
    }


def test_three_by_three_grid_has_local_edges_and_self_loops():
    graph = build_grid_graph({"grid": np.zeros((3, 3), dtype=object)})
    edges = set(map(tuple, graph.edge_index.T.tolist()))

    assert graph.edge_index.shape[1] == 49
    assert all((node, node) in edges for node in range(9))
    assert (0, 8) not in edges
    assert (8, 0) not in edges


def test_edge_attributes_follow_physical_message_direction():
    graph = build_grid_graph({"grid": np.zeros((3, 3), dtype=object)})

    edge_lookup = {
        tuple(edge): attributes
        for edge, attributes in zip(graph.edge_index.T.tolist(), graph.edge_attr)
    }

    # Node 0 is north-west of node 4. For the message 0 -> 4, travel is
    # one cell east and one cell south; physical y therefore decreases.
    diagonal = edge_lookup[(0, 4)]
    torch.testing.assert_close(
        diagonal,
        torch.tensor(
            [1.0, -1.0, np.sqrt(2.0), np.exp(-1.0)],
            dtype=torch.float32,
        ),
    )

    self_loop = edge_lookup[(4, 4)]
    torch.testing.assert_close(self_loop, torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_node_positions_use_row_major_grid_coordinates():
    graph = build_grid_graph({"grid": np.zeros((2, 3), dtype=object)})

    torch.testing.assert_close(
        graph.pos,
        torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, -1.0],
                [1.0, -1.0],
                [2.0, -1.0],
            ]
        ),
    )


def test_sensor_to_grid_edges_connect_each_source_to_every_node():
    edge_index, edge_attr = build_sensor_to_grid_edges(
        np.array([[True, False, False], [False, True, False]])
    )

    assert edge_index.shape == (2, 12)
    assert edge_attr.shape == (12, 4)
    assert set(edge_index[0].tolist()) == {0, 4}
    for target in range(6):
        assert set(edge_index[0, edge_index[1] == target].tolist()) == {0, 4}
    assert torch.isfinite(edge_attr).all()
