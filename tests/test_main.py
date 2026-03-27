from __future__ import annotations

import numpy as np

from metraq_dip import main


def test_denormalize_masked_restores_only_observed_entries():
    data = np.array([[[0.0, 1.0], [2.0, 0.0]]], dtype=np.float32)
    mask = np.array([[[False, True], [True, False]]], dtype=bool)

    restored = main._denormalize_masked(data, mask, mean=10.0, std=2.0)

    expected = np.array([[[0.0, 12.000001], [14.000002, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(restored, expected)


def test_denormalize_output_restores_full_array():
    data = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32)

    restored = main._denormalize_output(data, mean=10.0, std=2.0)

    expected = np.array([[10.0, 12.000001], [14.000002, 7.999999]], dtype=np.float32)
    np.testing.assert_allclose(restored, expected)
