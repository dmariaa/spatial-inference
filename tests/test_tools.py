from __future__ import annotations

import numpy as np

from metraq_dip.tools import tools


def test_calculate_interpolations_uses_mask_to_preserve_observed_zero_values():
    captured: dict[str, np.ndarray] = {}

    class DummyInterpolator:
        def __init__(self, data, mask):
            captured["data"] = np.asarray(data, dtype=np.float32)
            captured["mask"] = np.asarray(mask, dtype=bool)

        def __call__(self, x, y, mode: str = "grid"):
            assert mode == "grid"
            return np.full((len(y), len(x)), 5.0, dtype=np.float32)

    x_data = np.array([[[[0.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    x_mask = np.array([[[[True, False], [False, True]]]], dtype=bool)

    interpolated = tools.calculate_interpolations(x_data, x_mask, DummyInterpolator)

    np.testing.assert_allclose(captured["data"], x_data[0])
    np.testing.assert_array_equal(captured["mask"], x_mask[0])
    np.testing.assert_allclose(
        interpolated,
        np.array([[[[0.0, 5.0], [5.0, 1.0]]]], dtype=np.float32),
    )
