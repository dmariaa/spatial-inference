import pytest
import torch

from metraq_dip.trainer.dip_optimizer import DipOptimizer
from metraq_dip.trainer.graph_dip_optimizer import GraphDipOptimizer
from test_graph_dip_optimizer import graph_config, graph_split


@pytest.mark.parametrize("optimizer_class", [DipOptimizer, GraphDipOptimizer])
def test_mixed_loss_gradients_and_validation_selection(optimizer_class):
    optimizer = optimizer_class(configuration=graph_config(optimization_loss="mae_mse"),
                                split_data=graph_split(), device="cpu", disable_tqdm=True)
    error = torch.tensor(2.0, requires_grad=True)
    loss = optimizer._get_optimization_loss({"L1Loss": error.abs(), "MSELoss": error.square()})
    loss.backward()
    assert loss.item() == pytest.approx(2.4)
    assert error.grad.item() == pytest.approx(1.4)
    optimizer.artifacts = {"val_l1_history": torch.tensor([1.0, 1.1]),
                           "val_mse_history": torch.tensor([4.0, 1.0])}
    assert optimizer._get_selection_loss_history().argmin().item() == 1


def test_graph_mixed_loss_runs():
    optimizer = GraphDipOptimizer(configuration=graph_config(optimization_loss="mae_mse"),
                                  split_data=graph_split(), device="cpu", disable_tqdm=True)
    assert torch.isfinite(torch.as_tensor(optimizer.optimize())).all()


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_mse_weight(weight):
    with pytest.raises(ValueError, match="optimization_mse_weight"):
        DipOptimizer(configuration={"optimization_mse_weight": weight}, split_data={})
