"""Temporal persistence and the threshold logic of individual rules."""

from __future__ import annotations

import numpy as np
import pytest

from road_health.baseline import Baseline, Stat, robust_stat
from road_health.config import default_config
from road_health.health_checks import CheckResult, HealthChecker
from road_health.metrics import FrameMetrics
from road_health.temporal import Debouncer, EventTracker, frame_status


def test_debouncer_requires_the_full_hold_down():
    d = Debouncer(enter_frames=5, exit_frames=5)
    for _ in range(4):
        assert d.update(True) is False
    assert d.update(True) is True


def test_single_bad_frame_never_raises():
    d = Debouncer(enter_frames=10, exit_frames=10)
    for i in range(50):
        assert d.update(i % 7 == 0) is False


def test_debouncer_clears_after_exit_window():
    d = Debouncer(enter_frames=4, exit_frames=8)
    for _ in range(4):
        d.update(True)
    assert d.active
    for _ in range(7):
        d.update(False)
    assert d.active
    d.update(False)
    assert not d.active


def test_leaky_bucket_tolerates_intermittent_good_frames():
    d = Debouncer(enter_frames=10, exit_frames=10)
    raised = False
    for i in range(40):
        raised |= d.update(i % 4 != 0)
    assert raised


def test_sparse_noise_does_not_latch():
    d = Debouncer(enter_frames=10, exit_frames=10)
    raised = False
    for i in range(200):
        raised |= d.update(i % 5 == 0)
    assert not raised


def test_asymmetric_enter_and_exit_are_independent():
    fast_raise = Debouncer(enter_frames=2, exit_frames=50)
    fast_raise.update(True)
    assert fast_raise.update(True)
    for _ in range(40):
        fast_raise.update(False)
    assert fast_raise.active


def _res(name, triggered, value=1.0, thr=0.5):
    return CheckResult(name, triggered, value, thr, "test")


def test_event_tracker_records_one_event_with_the_true_onset():
    conds = {"blur": "DEGRADED"}
    t = EventTracker(conds, {"blur": 5}, {"blur": 5})
    for i in range(3):
        t.update(i, i / 25, {"blur": _res("blur", False)})
    for i in range(3, 20):
        t.update(i, i / 25, {"blur": _res("blur", True)})
    assert len(t.events) == 1
    ev = t.events[0]
    assert ev.start_frame == 3
    assert ev.detection_latency_frames == 4
    assert ev.severity == "DEGRADED"


def test_event_closes_and_records_duration():
    conds = {"frozen": "UNUSABLE"}
    t = EventTracker(conds, {"frozen": 3}, {"frozen": 3})
    for i in range(10):
        t.update(i, i / 25, {"frozen": _res("frozen", True)})
    for i in range(10, 30):
        t.update(i, i / 25, {"frozen": _res("frozen", False)})
    assert len(t.events) == 1
    ev = t.events[0]
    assert ev.ongoing is False
    assert ev.end_frame is not None and ev.end_frame > ev.start_frame
    assert ev.duration_s() > 0


def test_event_still_open_at_end_of_stream_is_marked_ongoing():
    conds = {"blocked": "UNUSABLE"}
    t = EventTracker(conds, {"blocked": 2}, {"blocked": 2})
    for i in range(10):
        t.update(i, i / 25, {"blocked": _res("blocked", True)})
    t.finalize(9, 9 / 25)
    assert t.events[0].ongoing is True
    assert t.events[0].end_frame == 9


def test_event_records_co_active_conditions():
    conds = {"blocked": "UNUSABLE", "dark": "DEGRADED"}
    t = EventTracker(conds, {"blocked": 2, "dark": 2}, {"blocked": 2, "dark": 2})
    for i in range(10):
        t.update(
            i, i / 25,
            {"blocked": _res("blocked", True), "dark": _res("dark", True)},
        )
    ev = [e for e in t.events if e.condition == "blocked"][0]
    assert "dark" in ev.co_active


def test_event_peak_ignores_healthy_frames_inside_the_event():
    conds = {"blocked": "UNUSABLE"}
    t = EventTracker(conds, {"blocked": 3}, {"blocked": 20})
    for i in range(6):
        t.update(i, i / 25, {"blocked": _res("blocked", True, value=0.40, thr=0.35)})
    t.update(6, 6 / 25, {"blocked": _res("blocked", False, value=0.0, thr=0.35)})
    for i in range(7, 12):
        t.update(i, i / 25, {"blocked": _res("blocked", True, value=0.55, thr=0.35)})
    ev = t.events[0]
    assert ev.peak_value == pytest.approx(0.55)


def test_frame_status_takes_the_worst_active_severity():
    sev = {"blur": "DEGRADED", "blocked": "UNUSABLE"}
    assert frame_status({"blur": False, "blocked": False}, sev) == "OK"
    assert frame_status({"blur": True, "blocked": False}, sev) == "DEGRADED"
    assert frame_status({"blur": True, "blocked": True}, sev) == "UNUSABLE"


def test_robust_stat_ignores_outliers():
    v = np.concatenate([np.full(95, 10.0), np.full(5, 1000.0)])
    s = robust_stat(v)
    assert s.median == pytest.approx(10.0)
    assert s.mad < 1.0


def test_baseline_roundtrip(tmp_path, gray_frame):
    b = Baseline(
        created_utc="now", source="test", n_frames=10, max_side=640,
        roi=None, tile_rows=2, tile_cols=2, analysis_size=[160, 256],
        stats={"mean_luma": Stat(100.0, 5.0, 90.0, 110.0, 10)},
        tile_std=np.ones((2, 2), dtype=np.float32),
        tile_edge=np.full((2, 2), 0.02, dtype=np.float32),
        reference_gray=gray_frame,
    )
    p = b.save(tmp_path / "baseline.json")
    loaded = Baseline.load(p)
    assert loaded.stats["mean_luma"].median == pytest.approx(100.0)
    assert loaded.tile_std.shape == (2, 2)
    assert loaded.reference_gray is not None
    assert loaded.reference_gray.shape == gray_frame.shape


def test_baseline_reports_incompatibility():
    b = Baseline(max_side=640, roi=None, tile_rows=6, tile_cols=8)
    assert b.check_compatible(640, None, 6, 8) == []
    problems = b.check_compatible(320, [0.0, 0.1, 1.0, 0.9], 4, 4)
    assert len(problems) == 3


def _metrics(**kw) -> FrameMetrics:
    base = dict(
        frame_index=0, timestamp_s=0.0, mean_luma=110.0, std_luma=40.0,
        p05=40.0, p95=190.0, contrast_span=150.0, clip_low_frac=0.0,
        clip_high_frac=0.0, entropy=7.0, lap_var=5000.0, reblur_ratio=40.0,
        edge_density=0.025, dead_tile_frac=0.0, dead_tile_mask=(0,) * 48,
        noise_sigma=1.5, frame_mad=2.0, shift_px=0.0, shift_frac=0.0,
        edge_corr=0.95,
    )
    base.update(kw)
    return FrameMetrics(**base)


def _baseline() -> Baseline:
    return Baseline(
        max_side=640, tile_rows=6, tile_cols=8,
        reference_gray=np.zeros((360, 640), dtype=np.uint8),
        stats={
            "mean_luma": Stat(110.0, 4.0, 100.0, 120.0, 100),
            "std_luma": Stat(40.0, 2.0, 36.0, 44.0, 100),
            "contrast_span": Stat(150.0, 6.0, 140.0, 160.0, 100),
            "entropy": Stat(7.0, 0.1, 6.8, 7.2, 100),
            "lap_var": Stat(5000.0, 400.0, 4200.0, 5800.0, 100),
            "reblur_ratio": Stat(40.0, 3.0, 35.0, 45.0, 100),
            "edge_density": Stat(0.025, 0.002, 0.021, 0.029, 100),
            "noise_sigma": Stat(1.5, 0.1, 1.3, 1.7, 100),
            "frame_mad": Stat(2.0, 0.4, 1.4, 2.8, 100),
        },
    )


def _checker() -> HealthChecker:
    return HealthChecker(default_config(), _baseline(), fps=25.0)


def test_healthy_frame_triggers_nothing():
    res = _checker()(_metrics())
    assert not any(r.triggered for r in res.values()), {
        k: r.detail for k, r in res.items() if r.triggered
    }


def test_blur_fires_below_the_relative_threshold():
    c = _checker()
    assert not c(_metrics(reblur_ratio=20.0))["blur"].triggered
    assert c(_metrics(reblur_ratio=10.0))["blur"].triggered


def test_blur_is_unavailable_on_a_railed_frame():
    r = _checker()(_metrics(mean_luma=8.0, reblur_ratio=2.0, clip_low_frac=0.6))["blur"]
    assert r.available is False and r.triggered is False


def test_dark_requires_information_loss_not_just_low_light():
    c = _checker()
    night = c(_metrics(mean_luma=45.0, contrast_span=140.0))["dark"]
    assert not night.triggered
    assert c(_metrics(mean_luma=45.0, clip_low_frac=0.5))["dark"].triggered
    assert c(_metrics(mean_luma=30.0, contrast_span=40.0))["dark"].triggered


def test_bright_fires_on_clipping():
    assert _checker()(_metrics(clip_high_frac=0.4))["bright"].triggered


def test_low_contrast_ignores_exposure_faults():
    c = _checker()
    assert c(_metrics(contrast_span=50.0))["low_contrast"].triggered
    assert not c(_metrics(contrast_span=50.0, mean_luma=230.0))["low_contrast"].triggered
    assert not c(_metrics(contrast_span=50.0, clip_low_frac=0.4))["low_contrast"].triggered


def test_blocked_uniform_field_path():
    r = _checker()(_metrics(std_luma=1.0, dead_tile_frac=0.0))["blocked"]
    assert r.triggered and "uniform" in r.detail


def test_blocked_is_unavailable_when_exposure_is_extreme():
    r = _checker()(_metrics(mean_luma=8.0, dead_tile_frac=0.9))["blocked"]
    assert r.available is False and r.triggered is False


def test_blur_is_suppressed_while_blocked():
    res = _checker()(_metrics(std_luma=1.0, reblur_ratio=1.0))
    assert res["blocked"].triggered
    assert res["blur"].triggered is False


def test_frozen_uses_the_noise_floor_not_a_fixed_number():
    c = _checker()
    assert not c(_metrics(noise_sigma=4.0, frame_mad=4.5))["frozen"].triggered
    assert c(_metrics(noise_sigma=4.0, frame_mad=0.01))["frozen"].triggered
    assert not c(_metrics(noise_sigma=0.4, frame_mad=0.45))["frozen"].triggered


def test_frozen_unavailable_without_a_usable_noise_floor():
    r = _checker()(_metrics(noise_sigma=0.05, frame_mad=0.0))["frozen"]
    assert r.available is False and r.triggered is False


def test_frozen_unavailable_on_the_first_frame():
    r = _checker()(_metrics(frame_mad=-1.0))["frozen"]
    assert r.available is False


def test_moved_fires_on_sustained_displacement():
    c = _checker()
    assert not c(_metrics(shift_frac=0.01))["moved"].triggered
    assert c(_metrics(shift_frac=0.10))["moved"].triggered


def test_moved_unavailable_when_framing_cannot_be_verified():
    c = _checker()
    r = c(_metrics(shift_frac=0.5, edge_density=0.0, reblur_ratio=2.0))["moved"]
    assert r.available is False and r.triggered is False


def test_moved_unavailable_without_a_reference_view():
    cfg = default_config()
    c = HealthChecker(cfg, None, fps=25.0)
    r = c(_metrics(shift_frac=0.9))["moved"]
    assert r.available is False and r.triggered is False


def test_disabled_checks_are_not_evaluated():
    cfg = default_config()
    cfg.checks["shake"].enabled = False
    res = HealthChecker(cfg, _baseline(), fps=25.0)(_metrics())
    assert "shake" not in res
