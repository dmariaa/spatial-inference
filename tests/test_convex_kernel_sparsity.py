"""Station removal must be nested, paired and independent of hidden measurements."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from metraq_gnn.data.graph import build_grid_graph
from metraq_gnn.model import ConvexGeometricKernel
from metraq_gnn.sparsity import RestrictedContext, canonicalize, classical_predictions, subset_plans


def test_fixed_nested_plans_do_not_duplicate_the_dense_baseline():
    context = np.zeros(30, bool)
    context[:20] = True
    a = subset_plans(context, seed=20261019, group_index=1)
    assert a == subset_plans(context, seed=20261019, group_index=1)
    assert len(a) == 16 and sum(p["source_count"] == 20 for p in a) == 1
    for repeat in range(5):
        subsets = [set(p["nodes"]) for p in a if p["repeat"] == repeat]
        assert len(subsets) == 3
        assert subsets[2] < subsets[1] < subsets[0] <= set(range(20))
    assert a != subset_plans(context, seed=20261019, group_index=2)


def sample():
    graph = build_grid_graph({"grid": np.zeros((2, 4))})
    visible = torch.tensor([1]*6+[0, 0], dtype=torch.bool)
    targets = ~visible
    values = torch.arange(8, dtype=torch.float32)
    return Data(x=torch.stack([torch.where(visible, values, 0.), visible.float()], -1)[:, None],
        y=values[:, None], context_mask=visible[:, None, None], context_sensor_mask=visible,
        target_mask=targets[:, None], target_sensor_mask=targets, forbidden_context_mask=targets,
        wind_uv=torch.ones((8, 2)), wind_available=visible.clone(), pos=graph.pos, num_nodes=8)


def test_removal_preserves_targets_and_valid_zero_without_mutating_original():
    original = sample()
    restricted = RestrictedContext([original], [0, 1, 4])[0]
    assert restricted.context_mask[0].item() and restricted.x[0, 0, 0] == 0.
    assert restricted.context_mask.sum() == 3 and original.context_mask.sum() == 6
    assert torch.equal(original.target_mask, restricted.target_mask)
    assert torch.equal(original.y[original.target_mask], restricted.y[original.target_mask])
    assert not restricted.wind_available[[2, 3, 5]].any()
    assert not restricted.wind_uv[[2, 3, 5]].any()
    assert restricted.forbidden_context_mask[[2, 3, 5, 6, 7]].all()


def test_all_methods_ignore_removed_aq_wind_and_target_labels():
    original = sample()
    changed = original.clone()
    changed.x[[2, 3, 5], :, 0] = 1e6
    changed.wind_uv[[2, 3, 5]] = -1e6
    changed.y[[2, 3, 5, 6, 7]] = -1e8
    nodes = [0, 1, 4]
    a, b = (RestrictedContext([item], nodes)[0] for item in (original, changed))
    model = ConvexGeometricKernel(mode="power", wind_mode="directional").prepare_geometry(original.pos, range(6))
    torch.testing.assert_close(model(a), model(b), rtol=0, atol=0)
    ids_a, predictions_a, count = classical_predictions(a, (2, 4), [10.], [2.])
    ids_b, predictions_b, _ = classical_predictions(b, (2, 4), [10.], [2.])
    assert count == 3
    np.testing.assert_array_equal(ids_a, ids_b)
    for method in ("KRG", "IDW"):
        np.testing.assert_array_equal(predictions_a[method], predictions_b[method])


def test_rejects_target_as_source_and_does_not_drop_source_failures():
    with pytest.raises(ValueError, match="forbidden or target"):
        RestrictedContext([sample()], [0, 6])[0]
    with pytest.raises(ValueError, match="fewer than two"):
        RestrictedContext([sample()], [0])[0]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "target", "nonfinite"])
def test_canonical_targets_are_never_silently_removed(defect):
    truth = pd.DataFrame([dict(sensor_group="a", window_end=pd.Timestamp("2024-01-01"), sensor_id=1, target=10.)])
    prediction = truth.assign(prediction=11.)
    if defect == "missing":
        prediction = prediction.iloc[:0]
    elif defect == "duplicate":
        prediction = pd.concat([prediction, prediction])
    elif defect == "target":
        prediction.loc[0, "target"] = 12.
    else:
        prediction.loc[0, "prediction"] = np.nan
    with pytest.raises(ValueError):
        canonicalize(prediction, truth)
