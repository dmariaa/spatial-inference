import pandas as pd
import pytest
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "kernel_comparison", Path(__file__).resolve().parents[1] / "scripts/evaluation/kernel_comparison.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
align_predictions = _module.align_predictions


def test_alignment_rejects_different_targets_or_keys():
    frame = pd.DataFrame(dict(window_end=["2024-01-01"], sensor_id=[1],
                              magnitude_id=[8], target=[10.], prediction=[9.]))
    assert len(align_predictions(frame, frame.copy())) == 1
    other = frame.copy()
    other["target"] = 11.
    with pytest.raises(ValueError, match="targets differ"):
        align_predictions(frame, other)
    other = frame.copy()
    other["sensor_id"] = 2
    with pytest.raises(ValueError, match="keys differ"):
        align_predictions(frame, other)
    with pytest.raises(ValueError, match="duplicate"):
        align_predictions(frame, pd.concat([frame,frame]))


def test_optional_wind_variant_uses_same_targets_and_keeps_baselines_isotropic(tmp_path):
    import hashlib
    import json
    import numpy as np
    from click.testing import CliRunner

    common = dict(hours=24, train_period="2015-2022", validation_period="2023",
                  seed=42, cell_size_m=1000, train_nodes=[0], validation_nodes=[1],
                  test_sensor_ids=[30], normalization_mean=[0.], normalization_std=[1.])
    arguments = []
    for mode, error in (("fixed", 4.), ("power", 3.), ("neural", 3.5), ("power-wind", 2.)):
        run, evaluation = tmp_path / mode, tmp_path / (mode + "-test")
        run.mkdir()
        evaluation.mkdir()
        is_wind = mode == "power-wind"
        config = dict(common, mode="power" if is_wind else mode)
        if is_wind:
            config.update(wind_mode="directional", wind_cache_fingerprint="wind-identity")
        (run / "config.json").write_text(json.dumps(config))
        training = dict(power=2., wind_strength=.5 if is_wind else 0., best_epoch=0,
                        epochs_completed=1, best_validation_mae=3., best_validation_mae_physical=3.)
        (run / "summary.json").write_text(json.dumps(training))
        (run / "best-model.pt").write_bytes(b"frozen-checkpoint")
        np.savez(run / "partition.npz", train_mask=[1, 0, 0],
                 validation_mask=[0, 1, 0], test_mask=[0, 0, 1])
        predictions = pd.DataFrame(dict(window_end=["2024-01-01"], sensor_id=[30],
                                       magnitude_id=[8], target=[10.], prediction=[10. + error]))
        predictions.to_csv(evaluation / "predictions.csv", index=False)
        pd.DataFrame(dict(sensor_id=[30], mae=[error])).to_csv(evaluation / "by-sensor.csv", index=False)
        summary = dict(test_mae_physical=error, test_rmse_physical=error, test_wape=error / 10.,
                       checkpoint_epoch=0, checkpoint_sha256=hashlib.sha256(b"frozen-checkpoint").hexdigest())
        if is_wind:
            summary["wind_cache_fingerprint"] = "wind-identity"
        (evaluation / "summary.json").write_text(json.dumps(summary))
        option = "wind" if is_wind else mode
        arguments += [f"--{option}-run", str(run), f"--{option}-test", str(evaluation)]
    output = tmp_path / "comparison"
    result = CliRunner().invoke(_module.main, arguments + ["--output", str(output)])
    assert result.exit_code == 0, result.output + repr(result.exception)
    frame = pd.read_csv(output / "comparison.csv")
    assert frame["mode"].tolist() == ["fixed", "power", "neural", "power-wind"]
    assert frame.loc[3, "wind_strength"] == .5
    assert frame.loc[3, "mae_change_vs_fixed_percent"] == -50.
    config_path = tmp_path / "neural/config.json"
    config = json.loads(config_path.read_text())
    config["wind_mode"] = "directional"
    config_path.write_text(json.dumps(config))
    result = CliRunner().invoke(_module.main, arguments + ["--output", str(tmp_path / "rejected")])
    assert result.exit_code != 0
    assert "unexpected model mode or wind mode" in str(result.exception)
