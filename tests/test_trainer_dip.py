from __future__ import annotations

import torch

from metraq_dip.trainer import trainer_dip


def test_get_model_output_uses_all_available_steps_when_k_best_matches_epoch_count():
    k_output = torch.tensor([[[[[1.0]], [[3.0]], [[5.0]]]]])
    k_val_mask = torch.ones_like(k_output)
    k_val_data = torch.zeros_like(k_output)

    model_output, min_idx = trainer_dip.get_model_output(
        k_output=k_output,
        k_val_mask=k_val_mask,
        k_val_data=k_val_data,
        k_best_n=3,
    )

    assert model_output.shape == (1, 1)
    assert model_output.item() == 3.0
    assert sorted(min_idx[0].tolist()) == [0, 1, 2]


def test_get_best_result_uses_k_best_n_from_configuration(monkeypatch):
    trainer = trainer_dip.DipTrainer(
        configuration={
            "aq_dataset": "metraq",
            "aq_backend": "files",
            "date": "2024-01-01T00:00:00",
            "hours": 1,
            "ensemble_size": 1,
            "pollutants": [7],
            "k_best_n": 3,
        },
    )

    trainer.dip_logger = {
        "train_output": torch.zeros((1, 1, 1, 1, 1)),
        "val_mask": torch.ones((1, 1, 1, 1, 1)),
        "test_mask": torch.ones((1, 1, 1, 1, 1)),
        "val_data": torch.zeros((1, 1, 1, 1, 1)),
        "test_data": torch.ones((1, 1, 1, 1, 1)),
    }

    captured: dict[str, int] = {}

    def fake_get_model_output(*, k_output, k_val_mask, k_val_data, k_best_n):
        captured["k_best_n"] = k_best_n
        return torch.zeros((1, 1)), torch.tensor([[0]])

    monkeypatch.setattr(trainer_dip, "get_model_output", fake_get_model_output)
    trainer.get_best_result()

    assert captured["k_best_n"] == 3
