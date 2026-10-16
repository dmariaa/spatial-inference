import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest
from click.testing import CliRunner

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "evaluate_cached_model", Path(__file__).resolve().parents[1] / "scripts/evaluation/evaluate_cached_model.py")
export_predictions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_predictions)
from metraq_gnn.data.cache import AQSensorCache
from metraq_gnn.data.graph import build_grid_graph
from metraq_gnn.model.convex_kernel import ConvexGeometricKernel


@pytest.mark.parametrize("mode", ["fixed", "neural", "local", "sensor-to-grid"])
@pytest.mark.parametrize("selected", [False, True])
def test_frozen_test_checkpoint_is_label_independent(tmp_path, monkeypatch, mode, selected):
    times = pd.date_range("2024-01-01", periods=25, freq="h").as_unit("ns")
    values = np.broadcast_to(np.array([10., 20., 30.])[None, :, None], (25, 3, 1)).copy()
    aq = AQSensorCache(
        path=tmp_path, metadata={"grid_shape": [1, 3], "cell_size_m": 1000},
        timestamps_ns=times.asi8, sensor_ids=np.array([10, 20, 30]),
        sensor_node_indices=np.arange(3), magnitude_ids=np.array([8]),
        values=values, availability=np.ones_like(values, dtype=bool),
        graph=build_grid_graph({"grid": np.zeros((1, 3))}))
    monkeypatch.setattr(export_predictions, "load_aq_sensor_cache", lambda _: aq)
    run = tmp_path / "run"
    run.mkdir()
    config = dict(train_nodes=[0], validation_nodes=[1], test_sensor_ids=[30],
                  normalization_mean=[0.], normalization_std=[1.], hours=24, seed=1,
                  mode=mode, cell_size_m=1000)
    is_kernel = mode in {"fixed", "neural"}
    if not is_kernel:
        config.pop("mode")
        config.update(architecture=mode, local_refinement_layers=1, time_features=False)
    (run / "config.json").write_text(json.dumps(config))
    best_epoch = -1 if mode == "fixed" else 4
    (run / "summary.json").write_text(json.dumps({"best_epoch": best_epoch}))

    np.savez(run / "partition.npz", train_mask=np.array([[1, 0, 0]], bool),
             validation_mask=np.array([[0, 1, 0]], bool), test_mask=np.array([[0, 0, 1]], bool),
             sensor_ids=aq.sensor_ids, sensor_node_indices=aq.sensor_node_indices)
    masks = (np.array([[1, 0, 0]], bool), np.array([[0, 1, 0]], bool), np.array([[0, 0, 1]], bool))
    kind = "convex-kernel" if is_kernel else "gnn"
    dataset = export_predictions.prepare_dataset(kind, aq, config, masks, None)
    model = export_predictions.build_model(kind, aq, config, masks, dataset[0])
    torch.save({"epoch": best_epoch, "model_state_dict": model.state_dict()}, run / "best-model.pt")
    extra = []
    if selected:
        index = tmp_path / "index.npz"
        np.savez(index, time_windows=np.array([times[-1].to_datetime64()]))
        extra = ["--test-windows-index", str(index)]
    digest = hashlib.sha256((run / "best-model.pt").read_bytes()).hexdigest()
    outputs = []
    for change in (0., 1000.):
        values[:, 2, :] = 30. + change
        output = tmp_path / (f"test-{int(change)}.csv" if selected else f"test-{int(change)}")
        result = CliRunner().invoke(export_predictions.main, [
            "--run-dir", str(run), "--cache", str(tmp_path), "--device", "cpu", "--output", str(output)] + extra)
        assert result.exit_code == 0, result.output + repr(result.exception)
        summary_path = output.with_suffix(".summary.json") if selected else output / "summary.json"
        summary = json.loads(summary_path.read_text())
        assert summary["test_windows"] == (1 if selected else 2)
        assert summary["test_target_count"] == (1 if selected else 2)
        assert summary["checkpoint_epoch"] == best_epoch
        assert summary["checkpoint_sha256"] == digest
        outputs.append(pd.read_csv(output if selected else output / "predictions.csv"))
        sensor_path = output.with_suffix(".by-sensor.csv") if selected else output / "by-sensor.csv"
        assert sensor_path.exists()
    np.testing.assert_array_equal(outputs[0]["prediction"], outputs[1]["prediction"])
    assert (outputs[1]["target"] - outputs[0]["target"] == 1000).all()
    assert hashlib.sha256((run / "best-model.pt").read_bytes()).hexdigest() == digest


def test_selected_windows_reject_missing_and_duplicate_timestamps(tmp_path):
    dataset = type("Windows", (), {"window_end_times": pd.date_range("2024-01-01", periods=2, freq="h")})()
    index = tmp_path / "index.npz"
    for times, message in [(["2025-01-01"], "unavailable"),
                           (["2024-01-01", "2024-01-01"], "duplicate")]:
        np.savez(index, time_windows=np.array(times))
        with pytest.raises(ValueError, match=message):
            export_predictions.selected_test_windows(dataset, index_path=index)


def test_reject_checkpoint_that_is_not_validation_selected(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"best_epoch": 5}))
    torch.save({"epoch": 4, "model_state_dict": {}}, tmp_path / "best-model.pt")
    result = CliRunner().invoke(export_predictions.main, [
        "--run-dir", str(tmp_path), "--cache", str(tmp_path)])
    assert result.exit_code != 0
    assert "not the validation-selected epoch" in result.output


def test_saved_partition_rejects_overlapping_test_context(tmp_path):
    aq = type("Cache", (), {"grid_shape": (1, 3), "sensor_ids": np.array([10, 20, 30]),
                            "sensor_node_indices": np.arange(3)})()
    np.savez(tmp_path / "partition.npz", sensor_ids=aq.sensor_ids,
             sensor_node_indices=aq.sensor_node_indices,
             train_mask=np.array([[1, 0, 1]], bool),
             validation_mask=np.array([[0, 1, 0]], bool), test_mask=np.array([[0, 0, 1]], bool))
    with pytest.raises(Exception, match="partitions overlap"):
        export_predictions.load_partition(tmp_path, aq, {})
