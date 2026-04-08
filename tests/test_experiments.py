from __future__ import annotations

import numpy as np

from metraq_dip import experiments


def test_run_single_experiment_persists_normalization_stats(monkeypatch, tmp_path):
    static_data = {
        "pollutants": [7],
        "test_data": np.array([[[[1.25]]]], dtype=np.float32),
        "test_mask": np.array([[[[True]]]], dtype=bool),
    }

    class DummyEnsembleOptimizer:
        def __init__(self, *, configuration, static_data, disable_tqdm=False, **kwargs):
            self._surface = np.array([[[12.5]]], dtype=np.float32)
            self._artifacts = {
                "surface": np.array([[[12.5]]], dtype=np.float32),
                "surface_reducer": "mean",
                "normalization_stats": {7: (10.0, 2.0)},
                "member_artifacts": [
                    {
                        "train_data": np.array([[[[1.0]]]], dtype=np.float32),
                        "val_data": np.array([[[[2.0]]]], dtype=np.float32),
                        "train_mask": np.array([[[[True]]]], dtype=bool),
                        "val_mask": np.array([[[[True]]]], dtype=bool),
                        "output_history": np.array([[[[1.25]]]], dtype=np.float32),
                        "train_l1_history": np.array([0.1], dtype=np.float32),
                        "train_mse_history": np.array([0.01], dtype=np.float32),
                        "val_l1_history": np.array([0.2], dtype=np.float32),
                        "val_mse_history": np.array([0.02], dtype=np.float32),
                        "selected_epoch_indices": np.array([0], dtype=np.int64),
                        "surface_model_space": np.array([[[1.25]]], dtype=np.float32),
                    }
                ],
            }

        def optimize(self) -> np.ndarray:
            return np.array(self._surface, copy=True)

        def get_artifacts(self) -> dict[str, object]:
            return dict(self._artifacts)

    captured: dict[str, object] = {}

    def fake_savez_compressed(file, **kwargs):
        captured["file"] = file
        captured["kwargs"] = kwargs

    monkeypatch.setattr(experiments, "collect_data", lambda **kwargs: static_data)
    monkeypatch.setattr(experiments, "DipEnsembleOptimizer", DummyEnsembleOptimizer)
    def fake_get_interpolation_loss(x_data, x_mask, y_data, y_mask, pollutants):
        assert isinstance(x_data, np.ndarray)
        assert isinstance(x_mask, np.ndarray)
        assert isinstance(y_data, np.ndarray)
        assert isinstance(y_mask, np.ndarray)
        return [{"loss": 0.1}, {"loss": 0.2}, {"loss": 0.3}, {"loss": 0.4}]

    monkeypatch.setattr(experiments, "get_interpolation_loss", fake_get_interpolation_loss)
    monkeypatch.setattr(experiments.np, "savez_compressed", fake_savez_compressed)

    row_result = experiments._run_single_experiment(
        config_base={
            "pollutants": [7],
            "hours": 1,
            "normalize": True,
            "validation_sensors": 1,
        },
        experiment_output_folder=str(tmp_path),
        test_sensor_group=[10],
        sensor_group_key="group-a",
        time_window_iso="2024-01-01T00:00:00",
        disable_nested_tqdm=True,
    )

    assert abs(row_result["DIP_L1Loss"]) < 1e-5
    assert abs(row_result["DIP_MSELoss"]) < 1e-5
    assert captured["kwargs"]["normalization_stats"] == {7: (10.0, 2.0)}
    np.testing.assert_allclose(captured["kwargs"]["train_output"], np.array([[1.25]], dtype=np.float32))
    np.testing.assert_allclose(captured["kwargs"]["train_output_real"], np.array([[12.5]], dtype=np.float32))
