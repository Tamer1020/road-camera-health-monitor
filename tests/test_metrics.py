"""Metric tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from road_health.metrics import (
    MetricExtractor,
    displacement,
    exposure_stats,
    frame_mad,
    laplacian_variance,
    noise_sigma,
    prepare_gray,
    reblur_ratio,
    tile_stats,
)


def test_blur_reduces_both_sharpness_metrics(gray_frame):
    blurred = cv2.GaussianBlur(gray_frame, (0, 0), 3.0)
    assert laplacian_variance(blurred) < 0.25 * laplacian_variance(gray_frame)
    assert reblur_ratio(blurred) < 0.5 * reblur_ratio(gray_frame)


def _line_scene(n_lines: int) -> np.ndarray:
    img = np.full((200, 320), 120, dtype=np.uint8)
    for i in range(n_lines):
        x = int(8 + i * (304 / max(1, n_lines)))
        cv2.line(img, (x, 10), (x, 190), (230,), 1)
        cv2.circle(img, (x, 100), 3, (30,), -1)
    return img


def test_reblur_ratio_is_more_scene_invariant_than_laplacian():
    sparse, dense = _line_scene(4), _line_scene(60)
    lap_spread = max(laplacian_variance(sparse), laplacian_variance(dense)) / min(
        laplacian_variance(sparse), laplacian_variance(dense)
    )
    rb_spread = max(reblur_ratio(sparse), reblur_ratio(dense)) / min(
        reblur_ratio(sparse), reblur_ratio(dense)
    )
    assert lap_spread > 5.0
    assert rb_spread < 2.0
    assert rb_spread < lap_spread / 4.0


def test_reblur_ratio_still_separates_focus_within_each_scene():
    for n in (4, 60):
        sharp = _line_scene(n)
        soft = cv2.GaussianBlur(sharp, (0, 0), 3.0)
        assert reblur_ratio(soft) < 0.1 * reblur_ratio(sharp)


def test_reblur_ratio_is_degenerate_on_a_featureless_frame():
    assert reblur_ratio(np.full((100, 100), 128, dtype=np.uint8)) == pytest.approx(0.0, abs=1e-6)


def test_exposure_stats_on_a_known_histogram():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:50] = 0
    img[50:] = 255
    s = exposure_stats(img)
    assert s["mean_luma"] == pytest.approx(127.5, abs=0.5)
    assert s["clip_low_frac"] == pytest.approx(0.5, abs=0.01)
    assert s["clip_high_frac"] == pytest.approx(0.5, abs=0.01)
    assert s["entropy"] == pytest.approx(1.0, abs=0.01)


def test_uniform_image_has_zero_entropy_and_span():
    s = exposure_stats(np.full((60, 60), 77, dtype=np.uint8))
    assert s["entropy"] == pytest.approx(0.0, abs=1e-6)
    assert s["contrast_span"] == pytest.approx(0.0, abs=1.0)
    assert s["std_luma"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("sigma", [1.0, 3.0, 8.0])
def test_noise_sigma_recovers_injected_gaussian_noise(sigma):
    rng = np.random.default_rng(int(sigma * 10))
    img = np.clip(128 + rng.normal(0, sigma, (300, 400)), 0, 255).astype(np.uint8)
    assert noise_sigma(img) == pytest.approx(sigma, rel=0.15)


def test_noise_sigma_of_a_noiseless_image_is_tiny(gray_frame):
    smooth = cv2.GaussianBlur(gray_frame, (0, 0), 2.0)
    assert noise_sigma(smooth) < 1.0


def test_frame_mad_is_zero_for_identical_frames(gray_frame):
    assert frame_mad(gray_frame, gray_frame.copy()) == 0.0


def test_frame_mad_first_frame_sentinel(gray_frame):
    assert frame_mad(gray_frame, None) == -1.0


def test_frame_mad_matches_expected_value_for_noise():
    rng = np.random.default_rng(3)
    s = 6.0
    a = np.clip(128 + rng.normal(0, s, (400, 400)), 0, 255).astype(np.uint8)
    b = np.clip(128 + rng.normal(0, s, (400, 400)), 0, 255).astype(np.uint8)
    assert frame_mad(a, b) == pytest.approx(1.128 * s, rel=0.08)


def test_tile_stats_match_a_naive_loop(gray_frame):
    rows, cols = 4, 5
    std, dens = tile_stats(gray_frame, rows, cols)
    edges = cv2.Canny(gray_frame, 50, 150) > 0
    h, w = gray_frame.shape
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    for r in range(rows):
        for c in range(cols):
            tile = gray_frame[ys[r] : ys[r + 1], xs[c] : xs[c + 1]]
            assert std[r, c] == pytest.approx(float(tile.std()), abs=1e-3)
            e = edges[ys[r] : ys[r + 1], xs[c] : xs[c + 1]]
            assert dens[r, c] == pytest.approx(float(e.mean()), abs=1e-6)


def test_dead_tiles_flag_only_the_occluded_region(gray_frame):
    ex = MetricExtractor(tile_rows=4, tile_cols=4)
    std, dens = tile_stats(gray_frame, 4, 4)
    ex.set_reference(None, std, dens)
    ex.base_global_std = float(gray_frame.std())

    covered = gray_frame.copy()
    covered[:, : gray_frame.shape[1] // 2] = 40
    cstd, cdens = tile_stats(covered, 4, 4)
    dead = ex.dead_tiles(cstd, cdens, float(covered.std()))
    assert dead[:, :2].all()
    assert not dead[:, 2:].any()


def test_global_darkening_does_not_flag_tiles(gray_frame):
    ex = MetricExtractor(tile_rows=4, tile_cols=4)
    std, dens = tile_stats(gray_frame, 4, 4)
    ex.set_reference(None, std, dens)
    ex.base_global_std = float(gray_frame.std())

    dark = (gray_frame.astype(np.float32) * 0.25).astype(np.uint8)
    dstd, ddens = tile_stats(dark, 4, 4)
    dead = ex.dead_tiles(dstd, ddens, float(dark.std()))
    assert dead.mean() < 0.35


def test_displacement_recovers_a_known_shift(gray_frame):
    ex = MetricExtractor()
    ref = ex.reference_edges(gray_frame)
    shifted = np.roll(gray_frame, 20, axis=1)
    shift_px, frac, corr = displacement(shifted, ref)
    diag = float(np.hypot(*gray_frame.shape[:2]))
    assert shift_px == pytest.approx(20.0, rel=0.25)
    assert frac == pytest.approx(20.0 / diag, rel=0.25)
    assert corr > 0.0


def test_displacement_of_identical_frame_is_zero(gray_frame):
    ex = MetricExtractor()
    ref = ex.reference_edges(gray_frame)
    shift_px, frac, corr = displacement(gray_frame, ref)
    assert shift_px < 1.0
    assert corr > 0.95


def test_displacement_without_reference_is_neutral(gray_frame):
    assert displacement(gray_frame, None) == (0.0, 0.0, 1.0)


def test_prepare_gray_fixes_the_analysis_scale():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    g = prepare_gray(frame, max_side=640)
    assert max(g.shape) == 640
    assert g.ndim == 2


def test_prepare_gray_applies_roi():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:10, :] = 255
    full = prepare_gray(frame, max_side=640)
    cropped = prepare_gray(frame, max_side=640, roi=(0.0, 0.2, 1.0, 0.8))
    assert cropped.shape[0] == 80
    assert full.mean() > cropped.mean()


def test_prepare_gray_rejects_empty_roi():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        prepare_gray(frame, roi=(0.999999, 0.999999, 0.0000001, 0.0000001))
