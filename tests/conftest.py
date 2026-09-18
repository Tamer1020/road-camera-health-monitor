"""Shared fixtures.

Every fixture builds its own video from scratch so the test suite has no
dependency on sample_data/ being present, and so CI is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

FPS = 25.0
SIZE = (256, 160)


def _textured_frame(rng: np.random.Generator, shift: int = 0) -> np.ndarray:
    base = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.float32)
    base[:] = 110.0
    small = rng.normal(0, 30, (SIZE[1] // 8, SIZE[0] // 8))
    base += cv2.resize(small, SIZE, interpolation=cv2.INTER_CUBIC)[..., None]
    for x in range(10, SIZE[0] - 10, 24):
        cv2.line(base, (x + shift, 10), (x + shift, SIZE[1] - 10), (210, 210, 210), 1)
    cv2.rectangle(base, (30 + shift, 40), (90 + shift, 100), (60, 160, 90), -1)
    cv2.circle(base, (180 + shift, 80), 22, (230, 90, 60), -1)
    return np.clip(base, 0, 255).astype(np.uint8)


def write_video(path: Path, frames) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    if not wr.isOpened():
        path = path.with_suffix(".avi")
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    for f in frames:
        wr.write(f)
    wr.release()
    return path


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def clean_video(tmp_path_factory) -> Path:
    r = np.random.default_rng(99)
    frames = []
    for i in range(120):
        f = _textured_frame(np.random.default_rng(99)).astype(np.float32)
        cv2.circle(f, (20 + (i * 2) % (SIZE[0] - 40), 130), 8, (40, 40, 200), -1)
        f += r.normal(0, 3.0, f.shape)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return write_video(tmp_path_factory.mktemp("vid") / "clean.mp4", frames)


@pytest.fixture(scope="session")
def frozen_video(tmp_path_factory) -> Path:
    r = np.random.default_rng(99)
    frames = []
    held = None
    for i in range(120):
        f = _textured_frame(np.random.default_rng(99)).astype(np.float32)
        cv2.circle(f, (20 + (i * 2) % (SIZE[0] - 40), 130), 8, (40, 40, 200), -1)
        f = np.clip(f + r.normal(0, 3.0, f.shape), 0, 255).astype(np.uint8)
        if i == 59:
            held = f.copy()
        frames.append(held.copy() if (i >= 60 and held is not None) else f)
    return write_video(tmp_path_factory.mktemp("vid") / "frozen.mp4", frames)


@pytest.fixture(scope="session")
def gray_frame(rng) -> np.ndarray:
    return cv2.cvtColor(_textured_frame(np.random.default_rng(5)), cv2.COLOR_BGR2GRAY)
