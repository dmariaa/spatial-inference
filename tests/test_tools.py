from __future__ import annotations

import numpy as np

from metraq_dip.tools import tools


def test_calculate_interpolations_uses_mask_to_preserve_observed_zero_values():
    captured: dict[str, np.ndarray] = {}

    class DummyInterpolator:
        def __init__(self, x, y, z):
            captured["x"] = np.asarray(x, dtype=np.float32)
            captured["y"] = np.asarray(y, dtype=np.float32)
            captured["z"] = np.asarray(z, dtype=np.float32)

        def __call__(self, x, y, mode: str = "points"):
            assert mode == "points"
            return np.full(np.asarray(x).shape, 5.0, dtype=np.float32)

    x_data = np.array([[[[0.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    x_mask = np.array([[[[True, False], [False, True]]]], dtype=bool)

    interpolated = tools.calculate_interpolations(x_data, x_mask, DummyInterpolator)

    np.testing.assert_allclose(captured["x"], np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(captured["y"], np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(captured["z"], np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(
        interpolated,
        np.array([[[[0.0, 5.0], [5.0, 1.0]]]], dtype=np.float32),
    )
