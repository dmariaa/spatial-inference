import importlib.util
from pathlib import Path
from click.testing import CliRunner


def test_cli_has_no_implicit_test_evaluation(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/train/train_convex_kernel.py"
    spec = importlib.util.spec_from_file_location("kernel_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner = CliRunner()
    result = runner.invoke(module.main, ["--help"])
    assert result.exit_code == 0
    assert "--test-sensor-ids" in result.output
    assert "--mode" in result.output
    result = runner.invoke(module.main, ["--output", str(tmp_path),
                                        "--test-sensor-ids", "1"])
    assert result.exit_code != 0
    assert "output already exists" in result.output


def test_synthetic_cache_training_without_test(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from metraq_gnn.data.cache import AQSensorCache
    from metraq_gnn.data.graph import build_grid_graph
    import json
    path = Path(__file__).resolve().parents[1] / "scripts/train/train_convex_kernel.py"
    spec = importlib.util.spec_from_file_location("kernel_training_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    times = pd.date_range("2015-01-01", periods=25, freq="h").append(
        pd.date_range("2023-01-01", periods=25, freq="h")).as_unit("ns")
    values = np.broadcast_to(np.arange(7, dtype=np.float32)[None, :, None] + 10,
                             (50, 7, 1)).copy()
    # Reserved TEST data are never needed, even to construct training samples.
    values[:, 6, :] = np.nan
    cache = AQSensorCache(
        path=tmp_path, metadata={"grid_shape": [1, 7], "cell_size_m": 1000},
        timestamps_ns=times.asi8, sensor_ids=np.arange(7),
        sensor_node_indices=np.arange(7), magnitude_ids=np.array([8]),
        values=values, availability=np.isfinite(values),
        graph=build_grid_graph({"grid": np.zeros((1, 7))}))
    monkeypatch.setattr(module, "load_aq_sensor_cache", lambda _: cache)
    loggers = []
    class FakeLogger:
        url = "https://example.invalid/local-test"
        def __init__(self, **kwargs):
            self.epochs = []
            self.finished = None
            loggers.append(self)
        def log_epoch(self, metrics, *, step):
            self.epochs.append(step)
        def log_summary(self, summary):
            pass
        def log_checkpoint(self, path):
            assert Path(path).is_file()
        def finish(self, *, exit_code):
            self.finished = exit_code
    monkeypatch.setattr(module, "WandbLogger", FakeLogger)
    for mode in ("fixed", "power", "neural"):
        output = tmp_path / mode
        result = CliRunner().invoke(module.main, [
            "--cache", str(tmp_path), "--output", str(output),
            "--mode", mode, "--test-sensor-ids", "6", "--validation-nodes", "2",
            "--epochs", "2", "--batch-size", "2", "--seed", "7",
            "--wandb", "--wandb-mode", "disabled"])
        assert result.exit_code == 0, result.output + repr(result.exception)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["test_evaluated"] is False
        assert summary["wandb_url"] == FakeLogger.url
        assert summary["training_seconds"] > 0
        assert loggers[-1].finished == 0
        assert len(loggers[-1].epochs) == (0 if mode == "fixed" else 2)
        assert np.isfinite(summary["power"])
        if mode != "fixed":
            assert (output / "best-model.pt").is_file()
            assert summary["epochs_completed"] == 2
