import torch
from torch_geometric.data import Data, Batch
from metraq_gnn.model import AdvectionDiffusionGNN


def sample():
    return Data(x=torch.tensor([[[10., 1.]], [[0., 0.]], [[30., 1.]]]),
        pos=torch.tensor([[0., 0.], [1., 0.], [2., 0.]]),
        context_mask=torch.tensor([[[True]], [[False]], [[True]]]),
        target_mask=torch.tensor([[False], [True], [False]]),
        forbidden_context_mask=torch.tensor([False, True, False]),
        wind_uv=torch.tensor([[1., 0.], [0., 0.], [1., 0.]]),
        wind_available=torch.tensor([True, False, True]), num_nodes=3)


def test_batch_masking_anchors_and_gradients():
    model = AdvectionDiffusionGNN(propagation_steps=2)
    data = sample()
    first = model(data)
    data.x[1, 0, 0] = 9999.
    torch.testing.assert_close(model(data), first)
    torch.testing.assert_close(model(Batch.from_data_list([data, data])), first.repeat(2, 1))
    torch.testing.assert_close(first[[0, 2]], torch.tensor([[10.], [30.]]))
    first[1].sum().backward()
    assert model.raw_diffusion.grad is not None
    assert model.initializer.output.weight.grad.abs().sum() > 0


def test_flux_conservation_positivity_and_wind_direction():
    model = AdvectionDiffusionGNN()
    edges = torch.tensor([[0, 1], [1, 0]])
    delta = torch.tensor([[1000., 0.], [-1000., 0.]])
    field = torch.tensor([[10.], [10.]])
    east = model.local_step(field, edges, delta, torch.tensor([[2., 0.], [2., 0.]]))
    west = model.local_step(field, edges, delta, torch.tensor([[-2., 0.], [-2., 0.]]))
    torch.testing.assert_close(east.sum(), field.sum())
    torch.testing.assert_close(east.flip(0), west)
    assert east[1] > east[0]
    assert east.min() >= 0
    calm = model.local_step(field, edges, delta, torch.zeros(2, 2))
    torch.testing.assert_close(calm, field)
    peak = torch.tensor([[20.], [0.]])
    smooth = model.local_step(peak, edges, delta, torch.zeros(2, 2))
    assert smooth.max() < peak.max()
    torch.testing.assert_close(smooth.sum(), peak.sum())
