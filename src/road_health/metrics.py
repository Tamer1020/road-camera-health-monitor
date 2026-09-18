"""Per-frame image-quality metrics.

Each metric is documented with: what it measures, why it was chosen over the
obvious alternative, and how it fails.  Nothing here makes a decision; that is
``health_checks.py``.  Keeping measurement and decision separate is what makes
threshold calibration possible at all.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_EPS = 1e-9


@dataclass
class FrameMetrics:
    frame_index: int
    timestamp_s: float
    mean_luma: float
    std_luma: float
    p05: float
    p95: float
    contrast_span: float
    clip_low_frac: float
    clip_high_frac: float
    entropy: float
    lap_var: float
    reblur_ratio: float
    edge_density: float
    dead_tile_frac: float
    dead_tile_mask: Tuple[int, ...]
    noise_sigma: float
    frame_mad: float
    shift_px: float
    shift_frac: float
    edge_corr: float

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row.pop("dead_tile_mask")
        return row

    @classmethod
    def csv_columns(cls) -> list[str]:
        return [f.name for f in fields(cls) if f.name != "dead_tile_mask"]


def prepare_gray(
    frame_bgr: np.ndarray,
    max_side: int = 640,
    roi: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    img = frame_bgr
    if roi is not None:
        h, w = img.shape[:2]
        x = int(round(roi[0] * w))
        y = int(round(roi[1] * h))
        rw = max(1, int(round(roi[2] * w)))
        rh = max(1, int(round(roi[3] * h)))
        img = img[y : min(h, y + rh), x : min(w, x + rw)]
        if img.size == 0:
            raise ValueError("ROI selects an empty region")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        gray = cv2.resize(
            gray,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return gray


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def reblur_ratio(gray: np.ndarray, sigma: float = 1.5) -> float:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return float(laplacian_variance(gray) / (laplacian_variance(blurred) + _EPS))


def edge_density(gray: np.ndarray, low: int = 50, high: int = 150) -> float:
    edges = cv2.Canny(gray, low, high)
    return float(np.count_nonzero(edges)) / float(edges.size)


def exposure_stats(gray: np.ndarray) -> Dict[str, float]:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(gray.size)
    p = hist / (total + _EPS)
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())

    cdf = np.cumsum(p)
    p05 = float(np.searchsorted(cdf, 0.05))
    p95 = float(np.searchsorted(cdf, 0.95))

    return {
        "mean_luma": float(gray.mean()),
        "std_luma": float(gray.std()),
        "p05": p05,
        "p95": p95,
        "contrast_span": float(p95 - p05),
        "clip_low_frac": float(hist[:3].sum() / total),
        "clip_high_frac": float(hist[253:].sum() / total),
        "entropy": entropy,
    }


def noise_sigma(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    conv = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, _NOISE_KERNEL)
    conv = conv[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(conv).mean() / 6.0)


def _block_sums(integral: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return (
        integral[np.ix_(ys[1:], xs[1:])]
        - integral[np.ix_(ys[:-1], xs[1:])]
        - integral[np.ix_(ys[1:], xs[:-1])]
        + integral[np.ix_(ys[:-1], xs[:-1])]
    )


def tile_stats(
    gray: np.ndarray, rows: int, cols: int, edges: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape[:2]
    if edges is None:
        edges = cv2.Canny(gray, 50, 150)
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    counts = np.maximum(1, np.outer(np.diff(ys), np.diff(xs))).astype(np.float64)

    s_i, s_i2 = cv2.integral2(gray)
    mean = _block_sums(s_i, ys, xs) / counts
    mean_sq = _block_sums(s_i2, ys, xs) / counts
    std = np.sqrt(np.maximum(0.0, mean_sq - mean * mean)).astype(np.float32)

    e_i = cv2.integral((edges > 0).astype(np.uint8))
    dens = (_block_sums(e_i, ys, xs) / counts).astype(np.float32)
    return std, dens


def frame_mad(gray: np.ndarray, prev_gray: Optional[np.ndarray]) -> float:
    if prev_gray is None or prev_gray.shape != gray.shape:
        return -1.0
    return float(cv2.absdiff(gray, prev_gray).mean())


REG_MAX_SIDE = 320


def _edge_magnitude(gray: np.ndarray, max_side: int = REG_MAX_SIDE) -> np.ndarray:
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        gray = cv2.resize(
            gray, (max(8, int(w * scale)), max(8, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def displacement(
    gray: np.ndarray,
    ref_edges: Optional[np.ndarray],
    hann: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    if ref_edges is None:
        return 0.0, 0.0, 1.0
    cur = _edge_magnitude(gray)
    if cur.shape != ref_edges.shape:
        return 0.0, 0.0, 1.0

    cur64 = cur.astype(np.float64)
    ref64 = ref_edges.astype(np.float64)
    if hann is None:
        hann = cv2.createHanningWindow((cur.shape[1], cur.shape[0]), cv2.CV_64F)
    (dx, dy), _response = cv2.phaseCorrelate(ref64, cur64, hann)

    a = cur64 - cur64.mean()
    b = ref64 - ref64.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum())) + _EPS
    corr = float((a * b).sum() / denom)

    shift_small = float(np.hypot(dx, dy))
    diag_small = float(np.hypot(*cur.shape[:2]))
    frac = shift_small / (diag_small + _EPS)
    shift_px = frac * float(np.hypot(*gray.shape[:2]))
    return shift_px, frac, corr


class MetricExtractor:
    def __init__(
        self,
        tile_rows: int = 6,
        tile_cols: int = 8,
        tile_abs_std: float = 4.0,
        tile_rel_std: float = 0.35,
        tile_rel_edge: float = 0.30,
        gain_floor: float = 0.25,
    ):
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.tile_abs_std = tile_abs_std
        self.tile_rel_std = tile_rel_std
        self.tile_rel_edge = tile_rel_edge
        self.gain_floor = gain_floor
        self.base_global_std: Optional[float] = None
        self._prev: Optional[np.ndarray] = None
        self._ref_edges: Optional[np.ndarray] = None
        self._hann: Optional[np.ndarray] = None
        self._ref_tile_std: Optional[np.ndarray] = None
        self._ref_tile_edge: Optional[np.ndarray] = None

    def set_reference(
        self,
        ref_edges: Optional[np.ndarray],
        ref_tile_std: Optional[np.ndarray] = None,
        ref_tile_edge: Optional[np.ndarray] = None,
    ) -> None:
        self._ref_edges = ref_edges
        self._ref_tile_std = ref_tile_std
        self._ref_tile_edge = ref_tile_edge
        if ref_edges is not None:
            self._hann = cv2.createHanningWindow(
                (ref_edges.shape[1], ref_edges.shape[0]), cv2.CV_64F
            )

    def reference_edges(self, gray: np.ndarray) -> np.ndarray:
        return _edge_magnitude(gray)

    def dead_tiles(
        self, std: np.ndarray, dens: np.ndarray, global_std: float
    ) -> np.ndarray:
        if self._ref_tile_std is not None and self._ref_tile_edge is not None:
            gain = 1.0
            if self.base_global_std and self.base_global_std > 0:
                gain = float(
                    np.clip(global_std / self.base_global_std, self.gain_floor, 3.0)
                )
            lost_std = std < self.tile_rel_std * self._ref_tile_std * gain
            lost_edge = dens < np.maximum(
                self.tile_rel_edge * self._ref_tile_edge * gain, 1e-4
            )
            return (lost_std & lost_edge).astype(np.uint8)
        return ((std < self.tile_abs_std) & (dens < 0.005)).astype(np.uint8)

    def __call__(self, gray: np.ndarray, frame_index: int, timestamp_s: float) -> FrameMetrics:
        ex = exposure_stats(gray)
        edges = cv2.Canny(gray, 50, 150)
        std_tiles, dens_tiles = tile_stats(gray, self.tile_rows, self.tile_cols, edges)
        dead = self.dead_tiles(std_tiles, dens_tiles, ex["std_luma"])
        shift_px, shift_frac, corr = displacement(gray, self._ref_edges, self._hann)

        m = FrameMetrics(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            mean_luma=ex["mean_luma"],
            std_luma=ex["std_luma"],
            p05=ex["p05"],
            p95=ex["p95"],
            contrast_span=ex["contrast_span"],
            clip_low_frac=ex["clip_low_frac"],
            clip_high_frac=ex["clip_high_frac"],
            entropy=ex["entropy"],
            lap_var=laplacian_variance(gray),
            reblur_ratio=reblur_ratio(gray),
            edge_density=float(np.count_nonzero(edges)) / float(edges.size),
            dead_tile_frac=float(dead.mean()),
            dead_tile_mask=tuple(int(v) for v in dead.ravel()),
            noise_sigma=noise_sigma(gray),
            frame_mad=frame_mad(gray, self._prev),
            shift_px=shift_px,
            shift_frac=shift_frac,
            edge_corr=corr,
        )
        self._prev = gray.copy()
        return m
