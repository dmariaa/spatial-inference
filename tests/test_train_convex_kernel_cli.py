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
        assert (output / "best-model.pt").is_file()
        if mode != "fixed":
            assert summary["epochs_completed"] == 2
        else:
            assert summary["epochs_completed"] == 0
            assert summary["best_epoch"] == -1
            assert summary["best_validation_mae_physical"] > 0


def test_train_2019_uses_only_recent_training_values_with_reference_partition(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pandas as pd
    from metraq_gnn.data.cache import AQSensorCache
    from metraq_gnn.data.graph import build_grid_graph
    from metraq_gnn.data.splits import split_sensor_nodes, compute_training_normalization

    path = Path(__file__).resolve().parents[1] / "scripts/train/train_convex_kernel.py"
    spec = importlib.util.spec_from_file_location("kernel_period_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    times = pd.date_range("2015-01-01", periods=25, freq="h").append(
        pd.date_range("2019-01-01", periods=25, freq="h")).append(
        pd.date_range("2023-01-01", periods=25, freq="h")).as_unit("ns")
    values = np.broadcast_to((10 + np.arange(5, dtype=np.float32))[None, :, None],(75,5,1)).copy()
    values[:25] += 10000
    values[50:] += 1000
    values[:,4] = np.nan
    cache = AQSensorCache(path=tmp_path, metadata=dict(grid_shape=[1,5],cell_size_m=1000),
        timestamps_ns=times.asi8,sensor_ids=np.arange(5),sensor_node_indices=np.arange(5),
        magnitude_ids=np.array([8]),values=values,availability=np.isfinite(values),
        graph=build_grid_graph({"grid":np.zeros((1,5))}))
    train, val, test = split_sensor_nodes(cache,validation_nodes=1,test_nodes=1,seed=42,test_sensor_ids=[4])
    old_mean, old_std = compute_training_normalization(
        cache,train_node_mask=train,start="2015-01-01",end="2022-12-31 23:00")
    reference = dict(seed=42,validation_nodes=1,test_nodes=1,train_nodes=3,test_sensor_ids=[4],
                     train_start="2015-01-01",normalization_mean=old_mean.tolist(),normalization_std=old_std.tolist())
    reference_path=tmp_path/"reference.json"
    reference_path.write_text(json.dumps(reference))
    monkeypatch.setattr(module,"load_aq_sensor_cache",lambda _:cache)
    output=tmp_path/"recent"
    result=CliRunner().invoke(module.main,[
        "--cache",str(tmp_path),"--output",str(output),"--mode","power",
        "--partition-summary",str(reference_path),"--train-start","2019-01-01",
        "--epochs","2","--batch-size","2","--seed","312537310"])
    assert result.exit_code==0,result.output+repr(result.exception)
    config=json.loads((output/"config.json").read_text())
    columns=train.reshape(-1)[cache.sensor_node_indices]
    recent=values[25:50,columns]
    np.testing.assert_allclose(config["normalization_mean"],[recent.mean()],rtol=1e-6)
    np.testing.assert_allclose(config["normalization_std"],[recent.std()],rtol=1e-6)
    assert config["partition_reference_normalization_checked"] is False
    assert config["train_period"]=="2019-2022"
    assert config["train_windows"]==2 and config["validation_windows"]==2
    assert config["train_start"]=="2019-01-01 00:00:00"
    assert config["normalization_period"]=="2019-01-01 00:00:00 to 2022-12-31 23:00:00"
    with np.load(output/"partition.npz") as saved:
        for key,expected in (("train_mask",train),("validation_mask",val),("test_mask",test)):
            np.testing.assert_array_equal(saved[key],expected)


def test_training_dates_cannot_include_validation_or_partial_hours(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/train/train_convex_kernel.py"
    spec = importlib.util.spec_from_file_location("kernel_invalid_dates_cli",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for extra in (["--train-end","2023-01-01"],["--train-start","2022-01-01","--train-end","2019-01-01"],
                  ["--train-start","2019-01-01 00:30"]):
        result=CliRunner().invoke(module.main,["--cache",str(tmp_path),"--output",str(tmp_path/"invalid")]+extra)
        assert result.exit_code!=0 and "TRAIN dates" in result.output
    assert not (tmp_path/"invalid").exists()
