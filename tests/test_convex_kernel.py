import pytest
import torch
from torch_geometric.data import Data, Batch
from metraq_gnn.model.convex_kernel import ConvexGeometricKernel
from metraq_gnn.data.kernel_window import adapt_kernel_window


def sample(offset=0.):
    return Data(pos=torch.tensor([[0., 0.], [1., 0.], [3., 0.]]),
                kernel_values=torch.tensor([[10.+offset], [0.], [30.+offset]]),
                kernel_available=torch.tensor([[True], [False], [True]]), num_nodes=3)


def test_weights_range_and_idw_initialization():
    model = ConvexGeometricKernel()
    prediction, weights = model(sample(), return_weights=True)
    alpha = weights[0][-1]
    assert torch.all(alpha > 0)
    torch.testing.assert_close(alpha.sum(1), torch.ones(3))
    assert torch.all((prediction >= 10) & (prediction <= 30))
    torch.testing.assert_close(prediction, ConvexGeometricKernel(mode="fixed")(sample()))
    assert prediction[0] == 10 and prediction[2] == 30
    d = torch.tensor([1001., 2001.])
    expected = ((d**-2) * torch.tensor([10., 30.])).sum() / (d**-2).sum()
    torch.testing.assert_close(prediction[1, 0], expected)


def test_constant_and_batch_isolation_and_labels():
    model = ConvexGeometricKernel()
    first, second = sample(), sample(100)
    batch = Batch.from_data_list([first, second])
    expected = torch.cat([model(first), model(second)])
    batch.y = torch.full((6, 1), float("nan"))
    torch.testing.assert_close(model(batch), expected)
    first.kernel_values[first.kernel_available] = 7
    torch.testing.assert_close(model(first), torch.full((3, 1), 7.))


def test_missing_sources_are_error():
    data = sample()
    data.kernel_available[:] = False
    with pytest.raises(ValueError, match="no current"):
        ConvexGeometricKernel()(data)


def test_adapter_uses_final_availability_and_rejects_leakage():
    data = sample()
    data.x = torch.tensor([[[99.], [10.]], [[88.], [0.]], [[77.], [30.]]])
    data.context_mask = torch.tensor([[[True], [True]], [[True], [False]],
                                      [[True], [True]]])
    data.target_mask = torch.tensor([[False], [True], [False]])
    data.forbidden_context_mask = torch.tensor([False, True, False])
    adapted = adapt_kernel_window(data)
    assert not adapted.kernel_available[1, 0]
    data.context_mask[1, -1] = True
    with pytest.raises(ValueError, match="target AQ"):
        adapt_kernel_window(data)


def test_learning_has_gradients():
    model = ConvexGeometricKernel()
    loss = (model(sample())[1, 0] - 25).square()
    loss.backward()
    assert model.raw_power.grad.abs() > 0
    assert model.correction[-1].weight.grad.abs().sum() > 0
