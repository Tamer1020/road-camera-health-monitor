#!/usr/bin/env python3
"""Build an evaluation corpus with exact, frame-level ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

FAULT_START = 150
FAULT_END = 330
MOVED_START = 200


def _ramp(i: int, start: int, n: int = 20) -> float:
    return float(np.clip((i - start) / float(n), 0.0, 1.0))


def c_none(f, i, rng):
    return f


def c_blur(f, i, rng):
    if not (FAULT_START <= i <= FAULT_END):
        return f
    sigma = 0.4 + 4.2 * _ramp(i, FAULT_START, 25)
    return cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)


def c_dark(f, i, rng):
    if not (FAULT_START <= i <= FAULT_END):
        return f
    a = _ramp(i, FAULT_START, 25)
    gain = 1.0 - 0.80 * a
    return np.clip(f.astype(np.float32) * gain - 14.0 * a, 0, 255).astype(np.uint8)


def c_bright(f, i, rng):
    if not (FAULT_START <= i <= FAULT_END):
        return f
    a = _ramp(i, FAULT_START, 25)
    gain = 1.0 + 1.6 * a
    return np.clip(f.astype(np.float32) * gain + 55.0 * a, 0, 255).astype(np.uint8)


def c_low_contrast(f, i, rng):
    if not (FAULT_START <= i <= FAULT_END):
        return f
    a = _ramp(i, FAULT_START, 25)
    t = 1.0 - 0.72 * a
    airlight = 168.0
    return np.clip(f.astype(np.float32) * t + airlight * (1 - t), 0, 255).astype(np.uint8)


def c_blocked_partial(f, i, rng):
    if not (FAULT_START <= i <= FAULT_END):
        return f
    out = f.copy()
    h, w = out.shape[:2]
    poly = np.array(
        [[0, 0], [int(w * 0.46), 0], [int(w * 0.40), h], [0, h]], dtype=np.int32
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    mask = cv2.GaussianBlur(mask, (0, 0), 6)
    a = (_ramp(i, FAULT_START, 12) * (mask.astype(np.float32) / 255.0))[..., None]
    cover = np.full_like(out, 42, dtype=np.float32)
    cover += rng.normal(0, 1.2, cover.shape)
    return np.clip(out.astype(np.float32) * (1 - a) + cover * a, 0, 255).astype(np.uint8)


def c_blocked_full(f, i, rng):
    if not (FAULT_START <= i <= FAULT_END):
        return f
    a = _ramp(i, FAULT_START, 12)
    cover = np.full_like(f, 58, dtype=np.float32) + rng.normal(0, 1.4, f.shape)
    return np.clip(f.astype(np.float32) * (1 - a) + cover * a, 0, 255).astype(np.uint8)


class FrozenCorruption:
    def __init__(self):
        self.held: Optional[np.ndarray] = None

    def __call__(self, f, i, rng):
        if i == FAULT_START - 1:
            self.held = f.copy()
        if FAULT_START <= i <= FAULT_END and self.held is not None:
            return self.held.copy()
        return f


def c_blur_from_start(f, i, rng):
    return cv2.GaussianBlur(f, (0, 0), sigmaX=4.6, sigmaY=4.6)


def c_moved(f, i, rng):
    if i < MOVED_START:
        return f
    h, w = f.shape[:2]
    a = _ramp(i, MOVED_START, 8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 4.5 * a, 1.0)
    M[0, 2] += 74.0 * a
    M[1, 2] += -38.0 * a
    return cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def c_shake(f, i, rng):
    if not (FAULT_START <= i <= FAULT_END):
        return f
    a = _ramp(i, FAULT_START, 10)
    dx = int(round(a * (4.5 * np.sin(i * 1.9) + rng.normal(0, 1.6))))
    dy = int(round(a * (3.5 * np.sin(i * 2.7 + 1.1) + rng.normal(0, 1.2))))
    return np.roll(np.roll(f, dx, axis=1), dy, axis=0)


CORRUPTIONS: Dict[str, Dict] = {
    "normal": {"fn": c_none, "condition": None},
    "blur": {"fn": c_blur, "condition": "blur"},
    "dark": {"fn": c_dark, "condition": "dark"},
    "bright": {"fn": c_bright, "condition": "bright"},
    "low_contrast": {"fn": c_low_contrast, "condition": "low_contrast"},
    "blocked_partial": {"fn": c_blocked_partial, "condition": "blocked"},
    "blocked_full": {"fn": c_blocked_full, "condition": "blocked"},
    "frozen": {"fn": FrozenCorruption, "condition": "frozen"},
    "moved": {"fn": c_moved, "condition": "moved"},
    "blur_from_start": {"fn": c_blur_from_start, "condition": "blur"},
    "shake": {"fn": c_shake, "condition": "shake"},
}


def read_all(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def write_clip(frames: List[np.ndarray], path: Path, fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not wr.isOpened():
        path = path.with_suffix(".avi")
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        wr.write(f)
    wr.release()
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", default="sample_data/road_clean.mp4")
    ap.add_argument("--outdir", "-o", default="sample_data/eval")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    src = Path(args.source)
    frames = read_all(src)
    if len(frames) < FAULT_END + 20:
        raise SystemExit(
            f"source has {len(frames)} frames; need at least {FAULT_END + 20}"
        )
    print(f"source: {src} ({len(frames)} frames)")

    outdir = Path(args.outdir)
    labels: Dict[str, Dict] = {
        "source": str(src),
        "n_frames": len(frames),
        "fps": args.fps,
        "clips": {},
    }

    for name, spec in CORRUPTIONS.items():
        rng = np.random.default_rng(args.seed)
        fn: Callable = spec["fn"]() if isinstance(spec["fn"], type) else spec["fn"]
        out = [fn(f.copy(), i, rng) for i, f in enumerate(frames)]
        path = write_clip(out, outdir / f"{name}.mp4", args.fps)

        cond = spec["condition"]
        if cond is None:
            window = None
        elif name == "moved":
            window = [MOVED_START, len(frames) - 1]
        elif name == "blur_from_start":
            window = [0, len(frames) - 1]
        else:
            window = [FAULT_START, FAULT_END]

        labels["clips"][name] = {
            "path": str(path),
            "condition": cond,
            "fault_window": window,
            "n_fault_frames": 0 if window is None else window[1] - window[0] + 1,
        }
        print(f"  {name:16s} -> {path}  fault={window}")

    labels_path = outdir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"labels: {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
