from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

from metraq_gnn.data.graph import build_grid_graph
from metraq_gnn.model import SpatioTemporalGNN
from metraq_gnn.trainer import GNNTrainer, masked_regression_losses


class ScalarNodeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, data: Data) -> torch.Tensor:
        return self.value.expand_as(data.y)


class RecordingLogger:
    def __init__(self) -> None:
        self.epochs = []
        self.summary = None
        self.checkpoint = None

    def log_epoch(self, metrics, *, step: int) -> None:
        self.epochs.append((step, dict(metrics)))

    def log_summary(self, summary) -> None:
        self.summary = dict(summary)

    def log_checkpoint(self, path) -> None:
        self.checkpoint = path


def _loader(target: float) -> DataLoader:
    sample = Data(
        x=torch.zeros((2, 1, 1)),
        y=torch.tensor([[target], [1000.0]]),
        target_mask=torch.tensor([[True], [False]]),
        edge_index=torch.tensor([[0, 1], [0, 1]]),
        edge_attr=torch.zeros((2, 1)),
        num_nodes=2,
    )
    return DataLoader([sample], batch_size=1, shuffle=False)


def test_masked_losses_ignore_unselected_nodes():
    losses = masked_regression_losses(
        prediction=torch.tensor([[2.0], [-500.0]]),
        target=torch.tensor([[5.0], [1000.0]]),
        mask=torch.tensor([[True], [False]]),
    )

    torch.testing.assert_close(losses["mae"], torch.tensor(3.0))
    torch.testing.assert_close(losses["mse"], torch.tensor(9.0))
    torch.testing.assert_close(losses["rmse"], torch.tensor(3.0))


def test_trainer_learns_masked_target_and_writes_best_checkpoint(tmp_path):
    torch.manual_seed(1)
    model = ScalarNodeModel()
    trainer = GNNTrainer(
        model=model,
        learning_rate=0.1,
        optimization_loss="mse",
        patience=20,
        gradient_clip_norm=None,
        device="cpu",
    )
    checkpoint = tmp_path / "best-model.pt"

    result = trainer.fit(
        _loader(3.0),
        _loader(3.0),
        epochs=100,
        checkpoint_path=checkpoint,
    )

    assert checkpoint.exists()
    assert result.best_epoch >= 0
    assert result.epochs_completed <= 100
    assert result.best_validation_loss < 0.01
    assert abs(model.value.item() - 3.0) < 0.1
    assert result.history[0]["validation_mse"] > result.best_validation_loss

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["epoch"] == result.best_epoch
    assert saved["validation_loss"] == result.best_validation_loss
    torch.testing.assert_close(saved["model_state_dict"]["value"], model.value.detach())


def test_trainer_reports_epochs_summary_and_checkpoint_to_logger(tmp_path):
    logger = RecordingLogger()
    checkpoint = tmp_path / "best-model.pt"
    trainer = GNNTrainer(
        model=ScalarNodeModel(),
        learning_rate=0.1,
        optimization_loss="mse",
        patience=2,
        gradient_clip_norm=None,
        device="cpu",
        logger=logger,
    )

    result = trainer.fit(_loader(1.0), _loader(1.0), epochs=3, checkpoint_path=checkpoint)

    assert len(logger.epochs) == result.epochs_completed
    assert logger.epochs[0][0] == 0
    assert logger.epochs[0][1]["epoch"] == 0.0
    assert logger.summary == {
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "epochs_completed": result.epochs_completed,
    }
    assert logger.checkpoint == checkpoint


def test_evaluate_reports_all_regression_metrics():
    model = ScalarNodeModel()
    model.value.data.fill_(1.0)
    trainer = GNNTrainer(model=model, device="cpu")

    metrics = trainer.evaluate(_loader(3.0))

    assert metrics == {"mae": 2.0, "mse": 4.0, "rmse": 2.0}


def test_trainer_updates_spatiotemporal_gnn_end_to_end():
    torch.manual_seed(5)
    graph = build_grid_graph({"grid": torch.zeros((2, 2)).numpy()})
    sample = Data(
        x=torch.randn((4, 3, 2)),
        y=torch.tensor([[0.0], [0.0], [1.5], [0.0]]),
        target_mask=torch.tensor([[False], [False], [True], [False]]),
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        num_nodes=4,
    )
    loader = DataLoader([sample], batch_size=1, shuffle=False)
    model = SpatioTemporalGNN(
        input_channels=2,
        output_channels=1,
        edge_channels=4,
        temporal_hidden_channels=4,
        graph_hidden_channels=4,
        graph_layers=1,
        attention_heads=1,
        dropout=0.0,
    )
    initial_attention = model.graph_convolutions[0].att.detach().clone()
    trainer = GNNTrainer(
        model=model,
        learning_rate=0.02,
        optimization_loss="mse",
        patience=5,
        device="cpu",
    )

    result = trainer.fit(loader, loader, epochs=5)

    assert result.epochs_completed == 5
    assert torch.isfinite(torch.tensor(result.best_validation_loss))
    assert not torch.equal(model.graph_convolutions[0].att.detach(), initial_attention)


def test_trainer_propagates_epoch_to_training_dataset():
    class EpochDataset(Dataset):
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return _loader(1.0).dataset[0]

    dataset = EpochDataset()
    trainer = GNNTrainer(
        model=ScalarNodeModel(),
        learning_rate=0.1,
        optimization_loss="mse",
        patience=5,
        gradient_clip_norm=None,
        device="cpu",
    )
    trainer.fit(DataLoader(dataset, batch_size=1), _loader(1.0), epochs=3)
    assert dataset.epochs == [0, 1, 2]
