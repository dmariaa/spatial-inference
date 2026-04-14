from __future__ import annotations

import numpy as np
import pandas as pd

from metraq_dip import experiments


def test_run_single_experiment_persists_normalization_stats(monkeypatch, tmp_path):
    static_data = {
        "pollutants": [7],
        "test_data": np.full((1, 24, 1, 1), 1.25, dtype=np.float32),
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
                        "train_data": np.full((1, 24, 1, 1), 1.0, dtype=np.float32),
                        "val_data": np.full((1, 24, 1, 1), 2.0, dtype=np.float32),
                        "train_mask": np.ones((1, 24, 1, 1), dtype=bool),
                        "val_mask": np.ones((1, 24, 1, 1), dtype=bool),
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
    monkeypatch.setattr(experiments, "get_aq_backend_for_config", lambda config: object())
    def fake_get_interpolation_loss(x_data, x_mask, y_data, y_mask, pollutants):
        assert isinstance(x_data, np.ndarray)
        assert isinstance(x_mask, np.ndarray)
        assert isinstance(y_data, np.ndarray)
        assert isinstance(y_mask, np.ndarray)
        assert x_data.shape == (1, 1, 1, 1)
        assert x_mask.shape == (1, 1, 1, 1)
        return [{"loss": 0.1}, {"loss": 0.2}, {"loss": 0.3}, {"loss": 0.4}]

    monkeypatch.setattr(experiments, "get_interpolation_loss", fake_get_interpolation_loss)
    monkeypatch.setattr(experiments.np, "savez_compressed", fake_savez_compressed)

    row_result = experiments._run_single_experiment(
        config_base={
            "aq_dataset": "metraq",
            "aq_backend": "files",
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


def test_run_experiments_writes_failure_log(monkeypatch, tmp_path, capsys):
    time_window = pd.Timestamp("2024-01-01T00:00:00")
    df = pd.DataFrame(
        {
            "time_window": [time_window],
            "sensor_group": pd.Series(["group-a"], dtype="string"),
            "processed": [False],
            "DIP_L1Loss": [0.0],
            "DIP_MSELoss": [0.0],
            "KRG_L1Loss": [0.0],
            "KRG_MSELoss": [0.0],
            "IDW_L1Loss": [0.0],
            "IDW_MSELoss": [0.0],
        }
    )

    monkeypatch.setattr(
        experiments,
        "_ensure_base_files",
        lambda config_file: ({}, str(tmp_path), np.array([[10]]), np.array([time_window]), df),
    )
    monkeypatch.setattr(experiments, "sensor_group_hash", lambda group: "group-a")

    def fake_run_single_experiment(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(experiments, "_run_single_experiment", fake_run_single_experiment)

    config_file = tmp_path / "config.yaml"
    config_file.write_text("session: test\n", encoding="utf-8")

    experiments.run_experiments(config_file=config_file, max_workers=1)

    failure_log = tmp_path / "failures.log"
    assert failure_log.exists()
    failure_text = failure_log.read_text(encoding="utf-8")
    assert "[1] group-a @ 2024-01-01T00:00:00" in failure_text
    assert "RuntimeError: boom" in failure_text

    captured = capsys.readouterr()
    assert "Full tracebacks written to" in captured.out
    assert "FAILED group-a" not in captured.out


def test_ensure_base_files_uses_backend_from_config(monkeypatch, tmp_path):
    class DummyTimeWindowsConfig:
        def model_dump(self) -> dict[str, object]:
            return {"year": 2024, "windows_per_month": 1, "start_hours": [8], "weekend_fraction": 0.4}

    class DummySessionConfig:
        aq_dataset = "metraq"
        aq_backend = "db"
        pollutants = [7]
        spread_test_groups = type("Spread", (), {"n_groups": 1, "group_size": 4, "max_uses_per_sensor": 2})()
        random_time_windows = DummyTimeWindowsConfig()
        all_time_windows = None

        def model_dump(self, *, exclude=None):
            return {
                "aq_dataset": "metraq",
                "aq_backend": "db",
                "pollutants": [7],
                "hours": 24,
                "epochs": 2,
                "ensemble_size": 1,
                "lr": 0.01,
                "normalize": False,
                "add_meteo": False,
                "add_time_channels": False,
                "add_coordinates": False,
                "add_distance_to_sensors": False,
                "add_traffic_data": False,
                "model": {
                    "base_channels": 16,
                    "levels": 3,
                    "preserve_time": False,
                    "learned_upsampling": False,
                    "skip_connections": True,
                },
            }

    captured_backend: dict[str, str | None] = {}
    monkeypatch.setattr(experiments, "load_session_config", lambda path: DummySessionConfig())
    monkeypatch.setattr(
        experiments,
        "get_random_time_windows",
        lambda **kwargs: [pd.Timestamp("2024-01-01T08:00:00")],
    )
    monkeypatch.setattr(
        experiments,
        "get_aq_backend_for_config",
        lambda config: {"dataset": config.aq_dataset, "backend": config.aq_backend},
    )

    def fake_get_spread_test_groups(**kwargs):
        captured_backend["dataset"] = kwargs["aq_backend"]["dataset"]
        captured_backend["backend"] = kwargs["aq_backend"]["backend"]
        return np.array([[10, 20, 30, 40]], dtype=np.int32), {}

    monkeypatch.setattr(experiments, "get_spread_test_groups", fake_get_spread_test_groups)

    config_file = tmp_path / "config.yaml"
    config_file.write_text("session: test\n", encoding="utf-8")

    config_base, _, test_sensors, time_windows, _ = experiments._ensure_base_files(config_file=config_file)

    assert captured_backend == {"dataset": "metraq", "backend": "db"}
    assert config_base["aq_dataset"] == "metraq"
    assert config_base["aq_backend"] == "db"
    np.testing.assert_array_equal(test_sensors, np.array([[10, 20, 30, 40]], dtype=np.int32))
    assert list(pd.to_datetime(time_windows)) == [pd.Timestamp("2024-01-01T08:00:00")]
