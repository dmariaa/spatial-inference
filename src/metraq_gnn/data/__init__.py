"""Data structures and transformations for graph-based spatial inference."""

from metraq_gnn.data.graph import (
    build_grid_graph,
    build_knn_sensor_to_grid_edges,
    build_sensor_to_grid_edges,
    grid_to_nodes,
    nodes_to_grid,
)
from metraq_gnn.data.cache import AQSensorCache, build_aq_sensor_cache, load_aq_sensor_cache
from metraq_gnn.data.dataset import GraphWindowDataset, SensorGraphWindowDataset
from metraq_gnn.data.splits import compute_training_normalization, split_sensor_nodes
from metraq_gnn.data.window import build_graph_window, split_training_sensor_masks

__all__ = [
    "build_graph_window",
    "build_aq_sensor_cache",
    "build_grid_graph",
    "build_knn_sensor_to_grid_edges",
    "build_sensor_to_grid_edges",
    "AQSensorCache",
    "GraphWindowDataset",
    "SensorGraphWindowDataset",
    "compute_training_normalization",
    "grid_to_nodes",
    "nodes_to_grid",
    "load_aq_sensor_cache",
    "split_training_sensor_masks",
    "split_sensor_nodes",
]
