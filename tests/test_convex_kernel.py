import pytest
import torch
from torch_geometric.data import Data, Batch
from metraq_gnn.model.convex_kernel import ConvexGeometricKernel


def sample(offset=0.):
    values = torch.tensor([[10.+offset], [0.], [30.+offset]])
    available = torch.tensor([[True], [False], [True]])
    return Data(pos=torch.tensor([[0., 0.], [1., 0.], [3., 0.]]),
                x=torch.cat((values, available.float()), dim=-1).unsqueeze(1),
                context_mask=available.unsqueeze(1), num_nodes=3)


def model(mode="neural"):
    return ConvexGeometricKernel(mode=mode).prepare_geometry(sample().pos, [0, 2])


def test_weights_range_and_idw_initialization():
    prediction, weights = model()(sample(), return_weights=True)
    alpha = weights[0][-1]
    assert torch.all(alpha > 0)
    torch.testing.assert_close(alpha.sum(1), torch.ones(3))
    assert torch.all((prediction >= 10) & (prediction <= 30))
    torch.testing.assert_close(prediction, model("fixed")(sample()))
    assert prediction[0] == 10 and prediction[2] == 30
    d = torch.tensor([1001., 2001.])
    expected = ((d**-2) * torch.tensor([10., 30.])).sum() / (d**-2).sum()
    torch.testing.assert_close(prediction[1, 0], expected)


def test_constant_and_batch_isolation_and_labels():
    kernel = model()
    first, second = sample(), sample(100)
    batch = Batch.from_data_list([first, second])
    expected = torch.cat([kernel(first), kernel(second)])
    batch.y = torch.full((6, 1), float("nan"))
    torch.testing.assert_close(kernel(batch), expected)
    first.x[:, -1, 0][first.context_mask[:, -1, 0]] = 7.
    torch.testing.assert_close(kernel(first), torch.full((3, 1), 7.))


def test_missing_sources_are_error():
    data = sample()
    data.context_mask[:] = False
    with pytest.raises(ValueError, match="no current"):
        model()(data)


def test_final_availability_and_leakage_checks():
    data = sample()
    data.x = data.x.repeat(1, 2, 1)
    data.x[:, 0, 0] = 999.
    data.context_mask = data.context_mask.repeat(1, 2, 1)
    data.context_mask[1, 0] = True
    data.target_mask = torch.tensor([[False], [True], [False]])
    data.forbidden_context_mask = torch.tensor([False, True, False])
    torch.testing.assert_close(model()(data), model()(sample()))
    data.context_mask[1, -1] = True
    with pytest.raises(ValueError, match="target AQ"):
        model()(data)
    data.target_mask[:] = False
    with pytest.raises(ValueError, match="forbidden AQ"):
        model()(data)


def test_learning_has_gradients():
    kernel = model()
    loss = (kernel(sample())[1, 0] - 25).square()
    loss.backward()
    assert kernel.raw_power.grad.abs() > 0
    assert kernel.correction[-1].weight.grad.abs().sum() > 0


def test_reject_different_grid_and_unprepared_geometry():
    data = sample()
    with pytest.raises(ValueError, match="prepare_geometry"):
        ConvexGeometricKernel()(data)
    different = sample()
    different.pos[1, 0] += .5
    with pytest.raises(ValueError, match="positions differ"):
        model()(Batch.from_data_list([data, different]))
    different = Data(pos=data.pos[:2], x=data.x[:2], context_mask=data.context_mask[:2], num_nodes=2)
    with pytest.raises(ValueError, match="same prepared grid"):
        model()(Batch.from_data_list([data, different]))
