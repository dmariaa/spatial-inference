from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from metraq_dip.utils import fix_experiment_norm_stats as fixer


def _write_experiment_file(path: Path, *, minmax_map=None) -> None:
    payload = {
        "train_data": np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
        "test_data": np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
        "train_mask": np.ones((1, 1, 1, 1, 1), dtype=bool),
        "val_mask": np.zeros((1, 1, 1, 1, 1), dtype=bool),
        "test_mask": np.ones((1, 1, 1, 1, 1), dtype=bool),
        "train_output": np.zeros((1, 1), dtype=np.float32),
        "val_min_idx": np.array([[0]], dtype=np.int64),
        "train_k_output": np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
        "train_k_loss": np.zeros((1, 1, 1, 2), dtype=np.float32),
        "val_k_loss": np.zeros((1, 1, 1, 2), dtype=np.float32),
    }
    if minmax_map is not None:
        payload["minmax_map"] = minmax_map
    np.savez_compressed(path, **payload)


def test_repair_experiment_folder_backfills_missing_minmax_map(monkeypatch, tmp_path):
    session_folder = tmp_path
    np.savez(
        session_folder / "data.npz",
        test_sensors=np.array([[10, 20, 30, 40]], dtype=np.int32),
        time_windows=np.array(["2024-01-01T00:00:00", "2024-01-02T00:00:00"], dtype="datetime64[s]"),
    )

    missing_file = session_folder / "exp_10-20-30-40_20240101T000000.npz"
    existing_file = session_folder / "exp_10-20-30-40_20240102T000000.npz"
    _write_experiment_file(missing_file)
    _write_experiment_file(existing_file, minmax_map={7: (1.0, 2.0)})

    monkeypatch.setattr(
        fixer,
        "load_session_config",
        lambda path: SimpleNamespace(normalize=True, pollutants=[7], hours=24),
    )

    calls: list[tuple[list[int], int, tuple[int, ...], str]] = []

    def fake_recover_experiment_minmax_map(*, pollutants, hours, time_window, test_sensors):
        calls.append((pollutants, hours, test_sensors, time_window.isoformat()))
        return {7: (12.5, 3.5)}

    monkeypatch.setattr(fixer, "recover_experiment_minmax_map", fake_recover_experiment_minmax_map)

    summary = fixer.repair_experiment_folder(session_folder)

    assert summary.updated == 1
    assert summary.skipped_existing == 1
    assert summary.skipped_not_normalized == 0
    assert summary.failed == []
    assert calls == [([7], 24, (10, 20, 30, 40), "2024-01-01T00:00:00")]

    with np.load(missing_file, allow_pickle=True) as repaired:
        assert repaired["minmax_map"].item() == {7: (12.5, 3.5)}

    with np.load(existing_file, allow_pickle=True) as untouched:
        assert untouched["minmax_map"].item() == {7: (1.0, 2.0)}


def test_repair_experiment_folder_skips_unnormalized_sessions(monkeypatch, tmp_path):
    session_folder = tmp_path
    np.savez(
        session_folder / "data.npz",
        test_sensors=np.array([[10, 20, 30, 40]], dtype=np.int32),
        time_windows=np.array(["2024-01-01T00:00:00"], dtype="datetime64[s]"),
    )
    experiment_file = session_folder / "exp_10-20-30-40_20240101T000000.npz"
    _write_experiment_file(experiment_file)

    monkeypatch.setattr(
        fixer,
        "load_session_config",
        lambda path: SimpleNamespace(normalize=False, pollutants=[7], hours=24),
    )

    summary = fixer.repair_experiment_folder(session_folder)

    assert summary.updated == 0
    assert summary.skipped_existing == 0
    assert summary.skipped_not_normalized == 1
    assert summary.failed == []

    with np.load(experiment_file, allow_pickle=True) as archive:
        assert "minmax_map" not in archive.files
