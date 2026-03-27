from __future__ import annotations

import numpy as np
import torch

from metraq_dip import experiments
from metraq_dip.trainer import tools as trainer_tools


def test_run_single_experiment_persists_minmax_map(monkeypatch, tmp_path):
    class DummyTrainer:
        def __init__(self, configuration: dict):
            self.pollutants = configuration["pollutants"]
            self.data = {"minmax_map": {7: (12.5, 3.5)}}
            self.dip_logger = {
                "train_data": torch.zeros((1, 1, 1, 1, 1)),
                "val_data": torch.zeros((1, 1, 1, 1, 1)),
                "test_data": torch.zeros((1, 1, 1, 1, 1)),
                "train_mask": torch.ones((1, 1, 1, 1, 1), dtype=torch.bool),
                "val_mask": torch.ones((1, 1, 1, 1, 1), dtype=torch.bool),
                "test_mask": torch.ones((1, 1, 1, 1, 1), dtype=torch.bool),
                "train_output": torch.zeros((1, 1, 1, 1, 1)),
                "train_loss": torch.zeros((1, 1, 1, 2)),
                "val_loss": torch.zeros((1, 1, 1, 2)),
                "test_loss": torch.zeros((1, 1, 1, 2)),
            }

        def __call__(self):
            return None

        def get_best_result(self):
            return (
                [{"loss": 1.0}, {"loss": 2.0}],
                np.zeros((1, 1), dtype=np.float32),
                np.array([[0]], dtype=np.int64),
            )

    captured: dict[str, object] = {}

    def fake_savez_compressed(file, **kwargs):
        captured["file"] = file
        captured["kwargs"] = kwargs

    monkeypatch.setattr(experiments, "DipTrainer", DummyTrainer)
    monkeypatch.setattr(
        experiments,
        "get_interpolation_loss",
        lambda *args, **kwargs: [{"loss": 0.1}, {"loss": 0.2}, {"loss": 0.3}, {"loss": 0.4}],
    )
    monkeypatch.setattr(experiments.np, "savez_compressed", fake_savez_compressed)

    experiments._run_single_experiment(
        config_base={"pollutants": [7]},
        experiment_output_folder=str(tmp_path),
        test_sensor_group=[10],
        sensor_group_key="group-a",
        time_window_iso="2024-01-01T00:00:00",
        disable_nested_tqdm=True,
    )

    assert captured["kwargs"]["minmax_map"] == {7: (12.5, 3.5)}


def test_load_experiment_data_restores_minmax_map(tmp_path):
    experiment_file = tmp_path / "exp_test.npz"
    np.savez(
        experiment_file,
        train_data=np.array([[[[[2.0]]]]], dtype=np.float32),
        test_data=np.array([[[[[1.0]]]]], dtype=np.float32),
        train_mask=np.ones((1, 1, 1, 1, 1), dtype=bool),
        val_mask=np.zeros((1, 1, 1, 1, 1), dtype=bool),
        test_mask=np.ones((1, 1, 1, 1, 1), dtype=bool),
        train_output=np.zeros((1, 1), dtype=np.float32),
        val_min_idx=np.array([[0]], dtype=np.int64),
        train_k_output=np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
        train_k_loss=np.zeros((1, 1, 1, 2), dtype=np.float32),
        val_k_loss=np.zeros((1, 1, 1, 2), dtype=np.float32),
        minmax_map={7: (12.5, 3.5)},
    )

    experiment_data = trainer_tools.load_experiment_data(experiment_file)

    assert experiment_data["minmax_map"] == {7: (12.5, 3.5)}
