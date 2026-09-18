"""Per-camera baseline ("what does healthy look like on *this* camera").

This module is the reason the project is not a pile of magic numbers.

Almost every image-quality metric worth computing is scene dependent.  A
Laplacian variance of 90 is excellent for a camera pointed at an empty desert
highway and terrible for one pointed at a tree line.  An absolute threshold
therefore cannot be both sensitive and specific across a fleet.

The standard industrial answer is to characterise each camera during
commissioning and then detect *deviation from that camera's own normal*.  That
is what a ``Baseline`` is: robust location and scale estimates for each metric,
plus a median background image of the commissioned field of view.

Robust statistics (median / MAD) are used rather than mean / std because the
calibration clip will contain vehicles, shadows and the occasional truck
parked in front of the lens.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from road_health.errors import BaselineError
from road_health.metrics import FrameMetrics, MetricExtractor, tile_stats

log = logging.getLogger(__name__)

BASELINE_VERSION = 1
BASELINE_METRICS = (
    "mean_luma",
    "std_luma",
    "contrast_span",
    "entropy",
    "lap_var",
    "reblur_ratio",
    "edge_density",
    "noise_sigma",
    "frame_mad",
)
_MAD_TO_SIGMA = 1.4826


@dataclass
class Stat:
    median: float
    mad: float
    p05: float
    p95: float
    n: int

    def z(self, value: float) -> float:
        return (value - self.median) / (self.mad + 1e-6)

    def rel(self, value: float) -> float:
        return value / (abs(self.median) + 1e-9)

    def to_dict(self) -> Dict[str, float]:
        return {
            "median": self.median,
            "mad": self.mad,
            "p05": self.p05,
            "p95": self.p95,
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Stat":
        return cls(
            median=float(d["median"]),
            mad=float(d["mad"]),
            p05=float(d["p05"]),
            p95=float(d["p95"]),
            n=int(d["n"]),
        )


def robust_stat(values: np.ndarray) -> Stat:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Stat(0.0, 0.0, 0.0, 0.0, 0)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)) * _MAD_TO_SIGMA)
    return Stat(
        median=med,
        mad=mad,
        p05=float(np.percentile(v, 5)),
        p95=float(np.percentile(v, 95)),
        n=int(v.size),
    )


@dataclass
class Baseline:
    """Characterisation of one camera in its healthy state."""

    version: int = BASELINE_VERSION
    created_utc: str = ""
    source: str = ""
    n_frames: int = 0
    max_side: int = 640
    roi: Optional[List[float]] = None
    tile_rows: int = 6
    tile_cols: int = 8
    analysis_size: Optional[List[int]] = None

    stats: Dict[str, Stat] = field(default_factory=dict)
    tile_std: Optional[np.ndarray] = None
    tile_edge: Optional[np.ndarray] = None
    reference_gray: Optional[np.ndarray] = None

    @classmethod
    def from_samples(
        cls,
        metrics: List[FrameMetrics],
        gray_frames: List[np.ndarray],
        *,
        source: str,
        max_side: int,
        roi: Optional[List[float]],
        tile_rows: int,
        tile_cols: int,
    ) -> "Baseline":
        if not metrics or not gray_frames:
            raise BaselineError("cannot build a baseline from zero frames")

        stats: Dict[str, Stat] = {}
        for name in BASELINE_METRICS:
            vals = np.array([getattr(m, name) for m in metrics], dtype=np.float64)
            if name == "frame_mad":
                vals = vals[vals >= 0]
                if vals.size == 0:
                    vals = np.array([0.0])
            stats[name] = robust_stat(vals)

        stack = np.stack(gray_frames, axis=0)
        reference = np.median(stack, axis=0).astype(np.uint8)
        t_std, t_edge = tile_stats(reference, tile_rows, tile_cols)

        std_stack, edge_stack = [], []
        for g in gray_frames:
            s, e = tile_stats(g, tile_rows, tile_cols)
            std_stack.append(s)
            edge_stack.append(e)
        t_std = np.median(np.stack(std_stack, axis=0), axis=0)
        t_edge = np.median(np.stack(edge_stack, axis=0), axis=0)

        return cls(
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=source,
            n_frames=len(metrics),
            max_side=max_side,
            roi=list(roi) if roi else None,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            analysis_size=[int(reference.shape[0]), int(reference.shape[1])],
            stats=stats,
            tile_std=t_std,
            tile_edge=t_edge,
            reference_gray=reference,
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ref_name = p.stem + "_reference.png"
        payload = {
            "version": self.version,
            "created_utc": self.created_utc,
            "source": self.source,
            "n_frames": self.n_frames,
            "max_side": self.max_side,
            "roi": self.roi,
            "tile_rows": self.tile_rows,
            "tile_cols": self.tile_cols,
            "analysis_size": self.analysis_size,
            "stats": {k: v.to_dict() for k, v in self.stats.items()},
            "tile_std": self.tile_std.tolist() if self.tile_std is not None else None,
            "tile_edge": self.tile_edge.tolist() if self.tile_edge is not None else None,
            "reference_image": ref_name,
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self.reference_gray is not None:
            cv2.imwrite(str(p.parent / ref_name), self.reference_gray)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        p = Path(path)
        if not p.is_file():
            raise BaselineError(f"baseline file not found: {p}")
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BaselineError(f"baseline file is not valid JSON: {exc}") from exc

        if int(payload.get("version", -1)) != BASELINE_VERSION:
            raise BaselineError(
                f"baseline version {payload.get('version')} is not supported by this "
                f"build (expected {BASELINE_VERSION}); re-run calibrate"
            )

        ref = None
        ref_name = payload.get("reference_image")
        if ref_name:
            ref_path = p.parent / ref_name
            if ref_path.is_file():
                ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
            else:
                log.warning("baseline reference image missing: %s", ref_path)

        return cls(
            version=int(payload["version"]),
            created_utc=payload.get("created_utc", ""),
            source=payload.get("source", ""),
            n_frames=int(payload.get("n_frames", 0)),
            max_side=int(payload.get("max_side", 640)),
            roi=payload.get("roi"),
            tile_rows=int(payload.get("tile_rows", 6)),
            tile_cols=int(payload.get("tile_cols", 8)),
            analysis_size=payload.get("analysis_size"),
            stats={k: Stat.from_dict(v) for k, v in payload.get("stats", {}).items()},
            tile_std=np.array(payload["tile_std"], dtype=np.float32)
            if payload.get("tile_std") is not None
            else None,
            tile_edge=np.array(payload["tile_edge"], dtype=np.float32)
            if payload.get("tile_edge") is not None
            else None,
            reference_gray=ref,
        )

    def check_compatible(self, max_side: int, roi, tile_rows: int, tile_cols: int) -> List[str]:
        problems: List[str] = []
        if self.max_side != max_side:
            problems.append(
                f"baseline was computed at max_side={self.max_side}, "
                f"run uses {max_side}"
            )
        base_roi = tuple(self.roi) if self.roi else None
        run_roi = tuple(roi) if roi else None
        if base_roi != run_roi:
            problems.append(f"baseline ROI {base_roi} != run ROI {run_roi}")
        if (self.tile_rows, self.tile_cols) != (tile_rows, tile_cols):
            problems.append(
                f"baseline tile grid {self.tile_rows}x{self.tile_cols} != "
                f"run grid {tile_rows}x{tile_cols}"
            )
        return problems

    def attach_to(self, extractor: MetricExtractor) -> None:
        ref_edges = (
            extractor.reference_edges(self.reference_gray)
            if self.reference_gray is not None
            else None
        )
        extractor.set_reference(ref_edges, self.tile_std, self.tile_edge)
        st = self.stats.get("std_luma")
        extractor.base_global_std = st.median if st is not None else None

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {k: v.to_dict() for k, v in self.stats.items()}
