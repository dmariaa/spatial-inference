import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest
from click.testing import CliRunner

from metraq_gnn import kernel_evaluation
from metraq_gnn.data.cache import AQSensorCache
from metraq_gnn.data.graph import build_grid_graph
from metraq_gnn.model.convex_kernel import ConvexGeometricKernel


@pytest.mark.parametrize("mode", ["fixed", "neural"])
def test_frozen_test_checkpoint_is_label_independent(tmp_path, monkeypatch, mode):
    times = pd.date_range("2024-01-01", periods=25, freq="h").as_unit("ns")
    values = np.broadcast_to(np.array([10., 20., 30.])[None, :, None], (25, 3, 1)).copy()
    aq = AQSensorCache(
        path=tmp_path, metadata={"grid_shape": [1, 3], "cell_size_m": 1000},
        timestamps_ns=times.asi8, sensor_ids=np.array([10, 20, 30]),
        sensor_node_indices=np.arange(3), magnitude_ids=np.array([8]),
        values=values, availability=np.ones_like(values, dtype=bool),
        graph=build_grid_graph({"grid": np.zeros((1, 3))}))
    monkeypatch.setattr(kernel_evaluation, "load_aq_sensor_cache", lambda _: aq)
    run = tmp_path / "run"
    run.mkdir()
    config = dict(train_nodes=[0], validation_nodes=[1], test_sensor_ids=[30],
                  normalization_mean=[0.], normalization_std=[1.], hours=24, seed=1,
                  mode=mode, cell_size_m=1000)
    (run / "config.json").write_text(json.dumps(config))
    best_epoch = -1 if mode == "fixed" else 4
    (run / "summary.json").write_text(json.dumps({"best_epoch": best_epoch}))
    torch.save({"epoch": best_epoch, "model_state_dict": ConvexGeometricKernel(mode=mode).state_dict()},
               run / "best-model.pt")
    np.savez(run / "partition.npz", train_mask=np.array([[1, 0, 0]], bool),
             validation_mask=np.array([[0, 1, 0]], bool), test_mask=np.array([[0, 0, 1]], bool),
             sensor_ids=aq.sensor_ids, sensor_node_indices=aq.sensor_node_indices)
    digest = hashlib.sha256((run / "best-model.pt").read_bytes()).hexdigest()
    outputs = []
    for change in (0., 1000.):
        values[:, 2, :] = 30. + change
        output = tmp_path / f"test-{int(change)}"
        result = CliRunner().invoke(kernel_evaluation.main, [
            "--run-dir", str(run), "--cache", str(tmp_path), "--output", str(output)])
        assert result.exit_code == 0, result.output + repr(result.exception)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["test_windows"] == 2
        assert summary["test_target_count"] == 2
        assert summary["checkpoint_epoch"] == best_epoch
        assert summary["checkpoint_sha256"] == digest
        outputs.append(pd.read_csv(output / "predictions.csv"))
    np.testing.assert_array_equal(outputs[0]["prediction"], outputs[1]["prediction"])
    assert (outputs[1]["target"] - outputs[0]["target"] == 1000).all()
    assert hashlib.sha256((run / "best-model.pt").read_bytes()).hexdigest() == digest
