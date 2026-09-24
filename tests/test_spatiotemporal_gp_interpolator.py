import numpy as np
import pytest

from metraq_dip.tools.interpolator import (
    IdwInterpolator,
    Interpolator,
    KrigingInterpolator,
    SpatioTemporalGPInterpolator,
)


def test_gp_uses_time_and_preserves_point_and_grid_shapes():
    data = np.array([[[1.0, 1.0]], [[9.0, 9.0]]])
    mask = np.ones_like(data, dtype=bool)
    gp = SpatioTemporalGPInterpolator(
        data,
        mask,
        noise_fraction=1e-6,
    )

    assert isinstance(gp, Interpolator)
    at_end = gp([0.5], [0], mode="points")
    assert at_end.shape == (1,)
    assert at_end[0] > 8

    grid = gp([0, 0.5, 1], [0, 1])
    assert grid.shape == (2, 3)
    assert np.isfinite(grid).all()


def test_gp_rejects_invalid_observations_and_prediction_times():
    with pytest.raises(ValueError, match="same shape"):
        SpatioTemporalGPInterpolator(np.ones((1, 1, 1)), np.ones((1, 1, 2), dtype=bool))

    with pytest.raises(ValueError, match="finite observation"):
        SpatioTemporalGPInterpolator(np.full((1, 1, 1), np.nan), np.ones((1, 1, 1), dtype=bool))


@pytest.mark.parametrize("interpolator_class", [IdwInterpolator, KrigingInterpolator])
def test_spatial_interpolators_ignore_history_before_target(interpolator_class):
    target = np.array([[1.0, 0.0, 3.0], [0.0, 0.0, 0.0], [5.0, 0.0, 7.0]])
    target_mask = target != 0
    mask = np.stack([np.ones_like(target_mask), target_mask])
    first_window = np.stack([np.ones_like(target), target])
    second_window = np.stack([np.full_like(target, 1000.0), target])

    first = interpolator_class(first_window, mask)([1.0], [1.0], mode="points")
    second = interpolator_class(second_window, mask)([1.0], [1.0], mode="points")

    np.testing.assert_allclose(first, second, rtol=0, atol=1e-10)
