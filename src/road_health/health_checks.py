"""Per-frame health rules.

Each rule answers one question about one frame.  No rule looks at time - that
is ``temporal.py``.  Splitting them means a rule can be tightened without
touching the alarm logic and vice versa.

Design notes that apply to every rule
-------------------------------------
* Where a baseline exists, the rule is expressed as a **ratio to that camera's
  own median**, not as an absolute number.
* Where no baseline exists, the rule falls back to an absolute threshold and
  the run metadata records that it did.  Fallback mode is explicitly less
  trustworthy and the README says so.
* "Low light" is not a fault.  A correctly exposed night scene has a low mean
  luma and is perfectly usable.  What matters is whether *information was
  lost* - the histogram crushed against an end of the range, or the usable
  dynamic range collapsed.  Every exposure rule therefore requires both a
  level condition and an information-loss condition.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

from road_health.baseline import Baseline
from road_health.config import Config
from road_health.metrics import FrameMetrics


@dataclass
class CheckResult:
    name: str
    triggered: bool
    value: float
    threshold: float
    detail: str
    available: bool = True
    suppressed_by: Optional[str] = None


CONDITIONS = ("blur", "dark", "bright", "low_contrast", "blocked", "frozen", "moved", "shake")


class HealthChecker:
    """Evaluates all enabled rules for one frame."""

    def __init__(self, config: Config, baseline: Optional[Baseline], fps: float):
        self.cfg = config
        self.base = baseline
        self.fps = fps
        window = int(
            round(
                config.checks["shake"].params.get("window_seconds", 1.0)
                * fps
                / max(1, config.analysis.stride)
            )
        ) if "shake" in config.checks else 0
        self._shift_hist: Deque[float] = deque(maxlen=max(3, window))
        self.has_reference = baseline is not None and baseline.reference_gray is not None

    def _exposure_sane(
        self, m: FrameMetrics, lo: float, hi: float
    ) -> Optional[str]:
        st = self._stat("mean_luma")
        if st is None or st.median <= 0:
            return None
        ratio = m.mean_luma / st.median
        if lo <= ratio <= hi:
            return None
        return f"exposure x{ratio:.2f} of baseline, outside [{lo:g}, {hi:g}]"

    def _framing_reliable(self, m: FrameMetrics) -> Optional[str]:
        ed = self._stat("edge_density")
        mn = self._stat("mean_luma")
        sp = self._stat("contrast_span")
        if ed is not None and ed.median > 0 and m.edge_density < 0.5 * ed.median:
            return f"edge content collapsed ({m.edge_density:.4f} vs {ed.median:.4f})"
        if mn is not None and mn.median > 0:
            ratio = m.mean_luma / mn.median
            if not (0.55 <= ratio <= 1.60):
                return f"exposure far from baseline (x{ratio:.2f})"
        if sp is not None and sp.median > 0 and m.contrast_span < 0.5 * sp.median:
            return f"contrast collapsed ({m.contrast_span:.0f} vs {sp.median:.0f})"
        return None

    def _stat(self, name: str):
        if self.base is None:
            return None
        return self.base.stats.get(name)

    def _blur(self, m: FrameMetrics) -> CheckResult:
        p = self.cfg.checks["blur"].params
        reason = self._exposure_sane(m, p["exposure_lo"], p["exposure_hi"])
        if reason:
            return CheckResult(
                "blur", False, m.reblur_ratio, 0.0,
                f"sharpness not measurable: {reason}", available=False,
            )
        st = self._stat("reblur_ratio")
        if st is not None and st.median > 0:
            thr = p["rel_min"] * st.median
            mode = f"rel<{p['rel_min']:.2f} of baseline {st.median:.2f}"
        else:
            thr = p["abs_min"]
            mode = f"absolute fallback <{thr:.2f}"
        return CheckResult(
            "blur", m.reblur_ratio < thr, m.reblur_ratio, thr,
            f"reblur_ratio={m.reblur_ratio:.2f} thr={thr:.2f} ({mode})",
        )

    def _dark(self, m: FrameMetrics) -> CheckResult:
        p = self.cfg.checks["dark"].params
        st = self._stat("mean_luma")
        st_span = self._stat("contrast_span")
        crushed = m.clip_low_frac >= p["clip_low_frac"]
        if st is not None and st.median > 0 and st_span is not None:
            low_level = m.mean_luma < p["rel_mean_luma"] * st.median
            lost_range = m.contrast_span < 0.55 * st_span.median
            thr = p["rel_mean_luma"] * st.median
            mode = "baseline-relative"
        else:
            low_level = m.mean_luma < p["abs_mean_luma"]
            lost_range = m.contrast_span < 45.0
            thr = p["abs_mean_luma"]
            mode = "absolute fallback"
        trig = crushed or (low_level and lost_range)
        return CheckResult(
            "dark", trig, m.mean_luma, thr,
            f"mean={m.mean_luma:.1f} thr={thr:.1f} span={m.contrast_span:.0f} "
            f"clip_low={m.clip_low_frac:.3f} ({mode})",
        )

    def _bright(self, m: FrameMetrics) -> CheckResult:
        p = self.cfg.checks["bright"].params
        st = self._stat("mean_luma")
        blown = m.clip_high_frac >= p["clip_high_frac"]
        if st is not None and st.median > 0:
            high_level = m.mean_luma > max(p["abs_mean_luma"], 1.7 * st.median)
        else:
            high_level = m.mean_luma > p["abs_mean_luma"]
        lost_range = m.contrast_span < 60.0
        trig = blown or (high_level and lost_range)
        return CheckResult(
            "bright", trig, m.clip_high_frac, p["clip_high_frac"],
            f"clip_high={m.clip_high_frac:.3f} mean={m.mean_luma:.1f} "
            f"span={m.contrast_span:.0f}",
        )

    def _low_contrast(self, m: FrameMetrics) -> CheckResult:
        p = self.cfg.checks["low_contrast"].params
        st = self._stat("contrast_span")
        if st is not None and st.median > 0:
            thr = p["rel_span"] * st.median
            mode = "baseline-relative"
        else:
            thr = p["abs_span"]
            mode = "absolute fallback"

        st_mean = self._stat("mean_luma")
        if st_mean is not None and st_mean.median > 0:
            ratio = m.mean_luma / st_mean.median
            normal_exposure = p["exposure_lo"] <= ratio <= p["exposure_hi"]
        else:
            normal_exposure = 45.0 <= m.mean_luma <= 200.0
        railed = (
            m.clip_low_frac >= p["max_clip_frac"] or m.clip_high_frac >= p["max_clip_frac"]
        )
        trig = (m.contrast_span < thr) and normal_exposure and not railed
        return CheckResult(
            "low_contrast", trig, m.contrast_span, thr,
            f"span={m.contrast_span:.0f} thr={thr:.0f} mean={m.mean_luma:.0f} ({mode})",
        )

    def _blocked(self, m: FrameMetrics) -> CheckResult:
        p = self.cfg.checks["blocked"].params
        thr = p["dead_tile_frac"]
        uniform = m.std_luma < p["uniform_std_max"]
        if uniform:
            return CheckResult(
                "blocked", True, 1.0, thr,
                f"uniform field: std_luma={m.std_luma:.2f} < {p['uniform_std_max']:g}",
            )
        reason = self._exposure_sane(m, p["exposure_lo"], p["exposure_hi"])
        if reason:
            return CheckResult(
                "blocked", False, m.dead_tile_frac, thr,
                f"occlusion not separable from an exposure fault: {reason}",
                available=False,
            )
        return CheckResult(
            "blocked", m.dead_tile_frac >= thr, m.dead_tile_frac, thr,
            f"dead_tiles={m.dead_tile_frac:.2f} thr={thr:.2f} std_luma={m.std_luma:.1f}",
        )

    def _frozen(self, m: FrameMetrics) -> CheckResult:
        p = self.cfg.checks["frozen"].params
        sigma = max(0.0, m.noise_sigma)
        expected_mad = 1.128 * sigma
        thr = max(p["abs_mad"], p["noise_ratio"] * expected_mad)
        if m.frame_mad < 0:
            return CheckResult("frozen", False, 0.0, thr, "no predecessor frame", available=False)
        if sigma < p["min_noise_sigma"]:
            return CheckResult(
                "frozen", False, m.frame_mad, thr,
                f"noise floor too low to decide (sigma={sigma:.3f} < "
                f"{p['min_noise_sigma']:.2f})",
                available=False,
            )
        return CheckResult(
            "frozen", m.frame_mad < thr, m.frame_mad, thr,
            f"mad={m.frame_mad:.3f} thr={thr:.3f} "
            f"(noise_sigma={sigma:.2f}, expected_mad={expected_mad:.2f})",
        )

    def _moved(self, m: FrameMetrics) -> CheckResult:
        p = self.cfg.checks["moved"].params
        if not self.has_reference:
            return CheckResult(
                "moved", False, 0.0, p["shift_frac"],
                "no reference view (baseline not loaded)", available=False,
            )
        reason = self._framing_reliable(m)
        if reason:
            return CheckResult(
                "moved", False, m.shift_frac, p["shift_frac"],
                f"cannot verify framing: {reason}", available=False,
            )
        trig = (m.shift_frac > p["shift_frac"]) or (m.edge_corr < p["edge_corr_min"])
        return CheckResult(
            "moved", trig, m.shift_frac, p["shift_frac"],
            f"shift={m.shift_px:.1f}px ({m.shift_frac*100:.1f}% diag) "
            f"edge_corr={m.edge_corr:.2f}",
        )

    def _shake(self, m: FrameMetrics) -> CheckResult:
        p = self.cfg.checks["shake"].params
        if not self.has_reference:
            return CheckResult(
                "shake", False, 0.0, p["jitter_frac"],
                "no reference view (baseline not loaded)", available=False,
            )
        reason = self._framing_reliable(m)
        if reason:
            self._shift_hist.clear()
            return CheckResult(
                "shake", False, 0.0, p["jitter_frac"],
                f"cannot verify framing: {reason}", available=False,
            )
        self._shift_hist.append(m.shift_frac)
        if len(self._shift_hist) < self._shift_hist.maxlen:
            return CheckResult(
                "shake", False, 0.0, p["jitter_frac"], "filling jitter window",
                available=False,
            )
        vals = list(self._shift_hist)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        jitter = var ** 0.5
        return CheckResult(
            "shake", jitter > p["jitter_frac"], jitter, p["jitter_frac"],
            f"jitter_std={jitter*100:.2f}% diag",
        )

    _RULES = {
        "blur": _blur,
        "dark": _dark,
        "bright": _bright,
        "low_contrast": _low_contrast,
        "blocked": _blocked,
        "frozen": _frozen,
        "moved": _moved,
        "shake": _shake,
    }

    def __call__(self, m: FrameMetrics) -> Dict[str, CheckResult]:
        results: Dict[str, CheckResult] = {}
        for name, rule in self._RULES.items():
            cfg = self.cfg.checks.get(name)
            if cfg is None or not cfg.enabled:
                continue
            results[name] = rule(self, m)

        if results.get("blocked") and results["blocked"].triggered and "blur" in results:
            if results["blur"].triggered:
                results["blur"].triggered = False
                results["blur"].suppressed_by = "blocked"
        return results
