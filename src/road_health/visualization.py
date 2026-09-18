"""Annotation overlay for the output video.

The overlay exists to make a run reviewable by a human in seconds: what the
tool thought, when, and on the strength of which number.  It is deliberately
plain - an operator screenshot has to survive being pasted into a maintenance
ticket.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from road_health.health_checks import CheckResult

_STATUS_COLOR = {
    "OK": (80, 175, 80),
    "DEGRADED": (40, 180, 230),
    "UNUSABLE": (60, 60, 220),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, _FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, _FONT, scale, color, thick, cv2.LINE_AA)


def draw_tile_overlay(
    frame: np.ndarray,
    dead_mask: Sequence[int],
    rows: int,
    cols: int,
    roi_px: Optional[tuple] = None,
) -> None:
    """Shade the tiles whose texture collapsed, so the operator can see *where*."""
    if not dead_mask or not any(dead_mask):
        return
    h, w = frame.shape[:2]
    x0, y0, rw, rh = roi_px if roi_px else (0, 0, w, h)
    ys = np.linspace(y0, y0 + rh, rows + 1).astype(int)
    xs = np.linspace(x0, x0 + rw, cols + 1).astype(int)
    overlay = frame.copy()
    for r in range(rows):
        for c in range(cols):
            if not dead_mask[r * cols + c]:
                continue
            cv2.rectangle(
                overlay, (xs[c], ys[r]), (xs[c + 1] - 1, ys[r + 1] - 1), (60, 60, 220), -1
            )
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, dst=frame)


def annotate(
    frame: np.ndarray,
    *,
    status: str,
    frame_index: int,
    timestamp_s: float,
    active: Dict[str, bool],
    results: Dict[str, CheckResult],
    dead_mask: Sequence[int] = (),
    tile_rows: int = 6,
    tile_cols: int = 8,
    roi_px: Optional[tuple] = None,
    camera_id: str = "",
) -> np.ndarray:
    """Return an annotated copy of ``frame``."""
    out = frame.copy()
    h, w = out.shape[:2]
    colour = _STATUS_COLOR.get(status, (200, 200, 200))

    if active.get("blocked"):
        draw_tile_overlay(out, dead_mask, tile_rows, tile_cols, roi_px)

    if roi_px is not None:
        x, y, rw, rh = roi_px
        cv2.rectangle(out, (x, y), (x + rw - 1, y + rh - 1), (200, 200, 200), 1)

    bar_h = max(28, int(h * 0.075))
    cv2.rectangle(out, (0, 0), (w, bar_h), (25, 25, 25), -1)
    cv2.rectangle(out, (0, 0), (w, bar_h), colour, 2)

    label = f"{status}"
    if camera_id:
        label = f"{camera_id}  |  {label}"
    _text(out, label, (10, int(bar_h * 0.68)), scale=0.62, color=colour, thick=2)
    _text(
        out,
        f"f={frame_index}  t={timestamp_s:6.2f}s",
        (w - 210, int(bar_h * 0.68)),
        scale=0.5,
        color=(220, 220, 220),
    )

    y = bar_h + 18
    for name, is_active in active.items():
        if not is_active:
            continue
        res = results.get(name)
        detail = res.detail if res else ""
        _text(out, f"[{name.upper()}] {detail}", (10, y), scale=0.44, color=colour)
        y += 18

    unavailable = [n for n, r in results.items() if not r.available]
    if unavailable:
        _text(
            out,
            "not evaluated: " + ", ".join(sorted(unavailable)),
            (10, h - 10),
            scale=0.4,
            color=(160, 160, 160),
        )
    return out
