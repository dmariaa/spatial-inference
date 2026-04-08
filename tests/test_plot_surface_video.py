from __future__ import annotations

import numpy as np

from metraq_dip.utils import plot_surface_video


def test_denormalize_masked_restores_only_observed_cells():
    data = np.array([[[0.0, 1.0], [2.0, 0.0]]], dtype=np.float32)
    mask = np.array([[[False, True], [True, False]]], dtype=bool)

    restored = plot_surface_video._denormalize_masked(data, mask, mean=10.0, std=2.0)

    expected = np.array([[[0.0, 12.000001], [14.000002, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(restored, expected)


def test_denormalize_full_restores_entire_array():
    data = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32)

    restored = plot_surface_video._denormalize_full(data, mean=10.0, std=2.0)

    expected = np.array([[10.0, 12.000001], [14.000002, 7.999999]], dtype=np.float32)
    np.testing.assert_allclose(restored, expected)


def test_denormalize_losses_rescales_l1_and_mse():
    losses = np.array([[[1.0, 4.0], [2.0, 9.0]]], dtype=np.float32)

    restored = plot_surface_video._denormalize_losses(losses, std=2.0)

    expected = np.array([[[2.000001, 16.000015], [4.000002, 36.000034]]], dtype=np.float32)
    np.testing.assert_allclose(restored, expected)


def test_get_interpolator_resolves_supported_methods():
    assert plot_surface_video._get_interpolator("kriging") is plot_surface_video.KrigingInterpolator
    assert plot_surface_video._get_interpolator("idw") is plot_surface_video.IdwInterpolator


def test_build_baseline_plot_data_uses_interpolated_surface_and_computes_losses(monkeypatch):
    surface = np.array([[[[1.0, 2.0], [4.0, 5.0]]]], dtype=np.float32)
    observed_slices: list[np.ndarray] = []

    def fake_calculate_interpolations(x_data, x_mask, interpolator):
        assert interpolator is plot_surface_video.IdwInterpolator
        observed_slices.append(x_data.copy())
        return surface

    monkeypatch.setattr(plot_surface_video, "calculate_interpolations", fake_calculate_interpolations)

    train_data = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    val_data = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    test_data = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    train_mask = np.zeros((1, 1, 2, 2, 2), dtype=bool)
    val_mask = np.zeros((1, 1, 2, 2, 2), dtype=bool)
    test_mask = np.zeros((1, 1, 2, 2, 2), dtype=bool)

    train_data[0, 0, 0, 0, 0] = 99.0
    train_mask[0, 0, 0, 0, 0] = True
    train_data[0, 0, 1, 0, 0] = 1.0
    train_mask[0, 0, 1, 0, 0] = True
    val_data[0, 0, 1, 0, 1] = 2.0
    val_mask[0, 0, 1, 0, 1] = True
    test_data[0, 0, 1, 1, 0] = 3.0
    test_mask[0, 0, 1, 1, 0] = True

    plot_data = plot_surface_video._build_baseline_plot_data(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        method="idw",
    )

    assert len(observed_slices) == 1
    np.testing.assert_allclose(
        observed_slices[0],
        np.array([[[[1.0, 2.0], [0.0, 0.0]]]], dtype=np.float32),
    )
    np.testing.assert_allclose(plot_data["y"], surface[0, 0:1])
    np.testing.assert_allclose(plot_data["train_loss"], np.array([[0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(plot_data["val_loss"], np.array([[0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(plot_data["test_loss"], np.array([[1.0, 1.0]], dtype=np.float32))


def test_build_dip_plot_data_uses_saved_output_surface():
    train_output = np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32)
    train_data = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    val_data = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    test_data = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    train_mask = np.zeros((1, 1, 2, 2, 2), dtype=bool)
    val_mask = np.zeros((1, 1, 2, 2, 2), dtype=bool)
    test_mask = np.zeros((1, 1, 2, 2, 2), dtype=bool)

    train_data[0, 0, 1, 0, 0] = 1.0
    train_mask[0, 0, 1, 0, 0] = True
    val_data[0, 0, 1, 0, 1] = 2.0
    val_mask[0, 0, 1, 0, 1] = True
    test_data[0, 0, 1, 1, 0] = 3.0
    test_mask[0, 0, 1, 1, 0] = True

    plot_data = plot_surface_video._build_dip_plot_data(
        train_output=train_output,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    np.testing.assert_allclose(plot_data["y"], train_output[None, ...])
    np.testing.assert_allclose(plot_data["train_loss"], np.array([[0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(plot_data["val_loss"], np.array([[0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(plot_data["test_loss"], np.array([[1.0, 1.0]], dtype=np.float32))
