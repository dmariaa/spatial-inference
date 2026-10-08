import numpy as np
import pytest
import torch

from metraq_dip.trainer.graph_dip_optimizer import GraphDipOptimizer
from test_graph_dip_optimizer import graph_config, graph_split
from test_config_tools import _base_config
from metraq_dip.tools.config_tools import SessionConfig


def optimizer(**overrides):
    return GraphDipOptimizer(configuration=graph_config(optimization_loss="huber", **overrides),
                             split_data=graph_split(), device="cpu", disable_tqdm=True)


def test_huber_formula_gradient_and_mask():
    opt = optimizer()
    prediction = torch.tensor([0.5, 2.0, float("nan")], requires_grad=True)
    loss = opt._masked_huber_loss(prediction, torch.zeros(3), torch.tensor([True, True, False]))
    assert loss.item() == pytest.approx((0.125 + 1.5) / 2)
    loss.backward()
    assert prediction.grad.tolist() == pytest.approx([0.25, 0.5, 0.0])
    with pytest.raises(ValueError, match="valid target"):
        opt._masked_huber_loss(prediction, torch.zeros(3), torch.zeros(3, dtype=torch.bool))


@pytest.mark.parametrize("delta", [0, -1, float("nan"), float("inf")])
def test_invalid_delta(delta):
    with pytest.raises(ValueError, match="huber_delta"):
        optimizer(huber_delta=delta)
    config = _base_config()
    config.update(optimization_loss="huber", huber_delta=delta, surface_optimizer="graph_dip")
    with pytest.raises(ValueError):
        SessionConfig.model_validate(config)


def test_huber_early_stop_selection_and_export():
    opt = optimizer(epochs=6)
    opt.patience = 1
    opt.min_delta = 1e6
    surface = opt.optimize()
    assert np.isfinite(surface).all()
    assert opt.artifacts["epochs_completed"].item() == 2
    history = opt.artifacts["val_huber_history"]
    assert torch.isfinite(history).all()
    assert torch.equal(history[2:], history[1].expand(4))
    assert opt.get_selected_epoch_indices().tolist() == [int(history[:2].argmin())]
    member = opt.get_artifacts()["member_artifacts"][0]
    assert "train_huber_history" in member and "val_huber_history" in member
    from metraq_dip.experiments import _build_experiment_artifacts
    saved = _build_experiment_artifacts(
        static_data={"test_data": np.zeros((1, 2, 3, 3), dtype=np.float32),
                     "test_mask": np.ones((3, 3), dtype=bool)},
        optimizer_artifacts=opt.get_artifacts(),
    )
    assert saved["val_huber_history"].shape == (1, 6)
    np.testing.assert_array_equal(saved["val_huber_history"][0], member["val_huber_history"])


def test_huber_config():
    config = _base_config()
    config.update(optimization_loss="huber", surface_optimizer="graph_dip")
    assert SessionConfig.model_validate(config).huber_delta == 1.0
