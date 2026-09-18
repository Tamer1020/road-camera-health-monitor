#!/usr/bin/env python3
"""Generate a synthetic road-camera clip.

Why synthetic source video is in this repository at all
------------------------------------------------------
Public road/traffic footage is almost never redistributable, and committing it
to a public GitHub repo is a licensing problem, not a technical one. A
procedurally generated scene sidesteps that and buys two things the evaluation
actually needs: a known-healthy reference and a controllable sensor noise floor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 360
HORIZON = 126
ROAD_TOP_HALF = 26
ROAD_BOTTOM_HALF = 330
CENTER_X = 316


def _noise_texture(rng: np.random.Generator, shape, scale: int, amp: float) -> np.ndarray:
    small = rng.normal(0.0, 1.0, (max(2, shape[0] // scale), max(2, shape[1] // scale)))
    big = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return (big * amp).astype(np.float32)


def _road_half_width(y: np.ndarray | float):
    t = np.clip((np.asarray(y, dtype=np.float32) - HORIZON) / float(H - HORIZON), 0.0, 1.0)
    return ROAD_TOP_HALF + t * (ROAD_BOTTOM_HALF - ROAD_TOP_HALF)


def build_background(rng: np.random.Generator) -> np.ndarray:
    bg = np.zeros((H, W, 3), dtype=np.float32)
    for y in range(HORIZON):
        f = y / float(HORIZON)
        bg[y, :] = (188 - 28 * f, 176 - 22 * f, 160 - 14 * f)
    bg[:HORIZON] += _noise_texture(rng, (HORIZON, W), 40, 4.0)[..., None]

    ground = np.zeros((H - HORIZON, W, 3), dtype=np.float32)
    ground[:] = (78, 96, 74)
    ground += _noise_texture(rng, (H - HORIZON, W), 6, 9.0)[..., None]
    bg[HORIZON:] = ground

    asphalt = np.full((H, W), 96.0, dtype=np.float32)
    asphalt += _noise_texture(rng, (H, W), 4, 7.5)
    asphalt += _noise_texture(rng, (H, W), 18, 5.0)
    ys = np.arange(HORIZON, H)
    half = _road_half_width(ys)
    for i, y in enumerate(ys):
        x0 = int(CENTER_X - half[i])
        x1 = int(CENTER_X + half[i])
        x0c, x1c = max(0, x0), min(W, x1)
        if x1c <= x0c:
            continue
        shade = 0.82 + 0.18 * ((y - HORIZON) / (H - HORIZON))
        bg[y, x0c:x1c] = (asphalt[y, x0c:x1c] * shade)[:, None]

    for i, y in enumerate(ys):
        for sign in (-1, 1):
            x = int(CENTER_X + sign * (half[i] - 4))
            if 0 <= x < W:
                bg[y, max(0, x - 1) : min(W, x + 2)] = 205.0

    for i, y in enumerate(ys[::2]):
        hw = _road_half_width(y)
        x = int(CENTER_X + hw + 8)
        if 0 <= x < W - 2:
            bg[y - 1 : y + 1, x : min(W, x + 3)] = (150, 152, 155)

    for px, py_top in ((88, 150), (566, 142), (200, 168)):
        cv2.line(bg, (px, py_top), (px, H - 40), (70, 70, 72), 2)
        cv2.line(bg, (px, py_top), (px + 22, py_top - 6), (70, 70, 72), 2)
        cv2.circle(bg, (px + 24, py_top - 6), 4, (210, 208, 195), -1)

    cv2.rectangle(bg, (380, 140), (492, 182), (44, 82, 52), -1)
    cv2.rectangle(bg, (380, 140), (492, 182), (225, 225, 225), 2)
    cv2.putText(bg, "E 311", (390, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2)
    cv2.line(bg, (436, 182), (436, 250), (60, 60, 62), 3)

    return np.clip(bg, 0, 255)


class Traffic:
    def __init__(self, rng: np.random.Generator, n: int = 7):
        self.rng = rng
        self.v = []
        for _ in range(n):
            self.v.append(self._spawn(depth=float(rng.uniform(0.05, 1.0))))

    def _spawn(self, depth: float | None = None):
        rng = self.rng
        lane = float(rng.choice([-0.55, -0.2, 0.2, 0.55]))
        approaching = lane < 0
        return {
            "d": 1.0 if depth is None else depth,
            "lane": lane,
            "speed": float(rng.uniform(0.0035, 0.0075)) * (1 if approaching else -1),
            "color": tuple(float(c) for c in rng.integers(35, 215, 3)),
            "w": float(rng.uniform(1.5, 2.4)),
            "h": float(rng.uniform(1.2, 1.9)),
        }

    def step(self):
        for i, veh in enumerate(self.v):
            veh["d"] -= veh["speed"]
            if veh["d"] <= 0.02 or veh["d"] >= 1.02:
                self.v[i] = self._spawn(depth=1.0 if veh["speed"] > 0 else 0.03)

    def draw(self, img: np.ndarray):
        for veh in sorted(self.v, key=lambda v: -v["d"]):
            d = float(np.clip(veh["d"], 0.0, 1.0))
            y = HORIZON + (H - HORIZON) * (1.0 - d) ** 1.9
            if y <= HORIZON + 2 or y >= H + 30:
                continue
            hw = _road_half_width(y)
            scale = hw / ROAD_BOTTOM_HALF
            cx = CENTER_X + veh["lane"] * hw
            bw = max(2.0, veh["w"] * 38 * scale)
            bh = max(2.0, veh["h"] * 30 * scale)
            x0, y0 = int(cx - bw / 2), int(y - bh)
            x1, y1 = int(cx + bw / 2), int(y)
            cv2.ellipse(
                img, (int(cx), y1), (max(1, int(bw * 0.55)), max(1, int(bh * 0.12))),
                0, 0, 360, (46, 52, 46), -1,
            )
            cv2.rectangle(img, (x0, y0), (x1, y1), veh["color"], -1)
            if bh > 8:
                cv2.rectangle(
                    img, (x0 + 2, y0 + 2), (x1 - 2, y0 + max(3, int(bh * 0.42))),
                    (28, 34, 40), -1,
                )
            if bw > 10:
                cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), 1)


def draw_lane_dashes(img: np.ndarray, phase: float):
    for y in range(HORIZON + 3, H):
        hw = _road_half_width(y)
        scale = hw / ROAD_BOTTOM_HALF
        z = 1.0 / max(scale, 1e-3)
        if int((z * 0.9 + phase)) % 2 == 0:
            continue
        x = int(CENTER_X)
        t = max(1, int(round(2.6 * scale)))
        cv2.line(img, (x - t, y), (x + t, y), (210, 210, 205), 1)


def render(
    n_frames: int,
    fps: float,
    seed: int,
    noise_sigma: float,
    out_path: Path,
) -> Path:
    rng = np.random.default_rng(seed)
    bg = build_background(rng)
    traffic = Traffic(rng, n=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        out_path = out_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("no usable VideoWriter backend")

    phase = 0.0
    for i in range(n_frames):
        frame = bg.copy()
        draw_lane_dashes(frame, phase)
        traffic.draw(frame)
        traffic.step()
        phase += 0.42
        gain = 1.0 + 0.02 * np.sin(i / 47.0)
        frame *= gain
        frame += rng.normal(0.0, noise_sigma, frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))

    writer.release()
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic road-camera clip.")
    ap.add_argument("--output", "-o", default="sample_data/road_clean.mp4")
    ap.add_argument("--frames", "-n", type=int, default=400)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise-sigma", type=float, default=2.2)
    args = ap.parse_args()

    path = render(args.frames, args.fps, args.seed, args.noise_sigma, Path(args.output))
    print(f"wrote {path} ({args.frames} frames @ {args.fps} fps, {W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
