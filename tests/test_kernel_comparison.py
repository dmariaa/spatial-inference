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
