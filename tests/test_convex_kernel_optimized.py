from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

from metraq_gnn.data.cache import AQSensorCache
from metraq_gnn.data.dataset import SensorGraphWindowDataset
from metraq_gnn.data.graph import build_grid_graph
from metraq_gnn.data.kernel_window import (
    CachedKernelDataset,
)
from metraq_gnn.model.convex_kernel import ConvexGeometricKernel


def make_cache():
    times = pd.date_range("2015-01-01", periods=54, freq="h").delete(27).as_unit("ns")
    rng = np.random.default_rng(51)
    values = rng.normal(20., 5., (len(times), 8, 2)).astype(np.float32)
    available = rng.random(values.shape) > .15
    # Colocated sensors, two pollutants, true zeros and missing data.
    available[:, :2] = True
    values[0:25, 0, 0] = 0.
    values[~available] = np.nan
    return AQSensorCache(
        path=Path("."), metadata={"grid_shape": [2, 4], "cell_size_m": 1000},
        timestamps_ns=times.asi8, sensor_ids=np.arange(8),
        sensor_node_indices=np.array([0, 0, 1, 2, 3, 4, 5, 6]),
        magnitude_ids=np.array([8, 12]), values=values, availability=available,
        graph=build_grid_graph({"grid": np.zeros((2, 4))}),
    )


def datasets(training):
    cache = make_cache()
    context = np.array([[True, True, True, True], [False, False, False, False]])
    target = np.array([[False, False, False, False], [True, True, False, False]])
    forbidden = ~context
    options = dict(
        cache=cache, context_sensor_mask=context, forbidden_context_mask=forbidden,
        hours=24, split_start="2015-01-01", split_end="2015-01-03 05:00",
        normalization_mean=np.array([10., 15.]), normalization_std=np.array([5., 7.]),
        seed=17,
    )
    options.update(dict(target_fraction=.25) if training else dict(target_sensor_mask=target))
    return cache, context, SensorGraphWindowDataset(**options), CachedKernelDataset(**options)


@pytest.mark.parametrize("training", [False, True])
def test_snapshot_matches_original_windows_masks_and_values(training):
    cache, context, old, new = datasets(training)
    assert old.end_indices == new.end_indices
    for epoch in (0, 3):
        old.set_epoch(epoch)
        new.set_epoch(epoch)
        for index in range(len(new)):
            previous, snapshot = old[index], new[index]
            torch.testing.assert_close(previous.x[:, -1, :cache.magnitude_ids.size], snapshot.x[:, -1, :cache.magnitude_ids.size])
            torch.testing.assert_close(previous.context_mask[:, -1], snapshot.context_mask[:, -1])
            torch.testing.assert_close(previous.target_mask, snapshot.target_mask)
            # Non-target labels deliberately omit forbidden sensors in the new path.
            torch.testing.assert_close(previous.y[previous.target_mask], snapshot.y[snapshot.target_mask])
            assert previous.window_end_ns == snapshot.window_end_ns


@pytest.mark.parametrize("mode", ["fixed", "power", "neural"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_snapshot_model_predictions_weights_and_gradients_match(mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cache, context, old, new = datasets(False)
    indices = [0, 1, len(new)-1]
    sparse = Batch.from_data_list([old[index] for index in indices]).to(device)
    dense = Batch.from_data_list([new[index] for index in indices]).to(device)
    history_model = ConvexGeometricKernel(mode=mode).to(device).prepare_geometry(
        cache.graph.pos, np.flatnonzero(context))
    # Verify beyond the special zero-correction initialization.
    with torch.no_grad():
        history_model.correction[-1].weight.normal_(std=.2)
        history_model.correction[-1].bias.fill_(.3)
    snapshot_model = deepcopy(history_model).prepare_geometry(
        cache.graph.pos, np.flatnonzero(context))
    expected, expected_weights = history_model(sparse, return_weights=True)
    actual, actual_weights = snapshot_model(dense, return_weights=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for a, b in zip(expected_weights, actual_weights, strict=True):
        for aa, bb in zip(a, b, strict=True):
            torch.testing.assert_close(aa, bb, rtol=2e-5, atol=2e-6)
    if mode != "fixed":
        ((expected[sparse.target_mask] - sparse.y[sparse.target_mask])**2).mean().backward()
        ((actual[dense.target_mask] - dense.y[dense.target_mask])**2).mean().backward()
        for a, b in zip(history_model.parameters(), snapshot_model.parameters(), strict=True):
            if a.requires_grad:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=3e-6)
    # Geometry is a runtime cache, not part of saved learned parameters.
    assert history_model.state_dict().keys() == snapshot_model.state_dict().keys()


def test_forbidden_test_changes_cannot_change_snapshot():
    cache, context, old, new = datasets(True)
    before = new[0]
    # Sensor 7 belongs to the held-out TEST cell 6.
    cache.values[:, 7, :] = 1e10
    cache.availability[:, 7, :] = True
    options = dict(cache=cache, context_sensor_mask=context,
                   forbidden_context_mask=~context, hours=24,
                   split_start="2015-01-01", split_end="2015-01-03 05:00",
                   target_fraction=.25, seed=17,
                   normalization_mean=np.array([10., 15.]),
                   normalization_std=np.array([5., 7.]))
    after = CachedKernelDataset(**options)[0]
    torch.testing.assert_close(before.x, after.x)
    torch.testing.assert_close(before.y, after.y)
    torch.testing.assert_close(before.target_mask, after.target_mask)
