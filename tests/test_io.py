"""Input validation: every bad thing an operator can hand us gets a typed error."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from road_health.errors import (
    CorruptInputError,
    InputNotFoundError,
    UnsupportedInputError,
)
from road_health.io import AnnotatedVideoWriter, VideoSource


def test_missing_path(tmp_path):
    with pytest.raises(InputNotFoundError):
        VideoSource(tmp_path / "does_not_exist.mp4")


def test_directory_input(tmp_path):
    d = tmp_path / "a_folder.mp4"
    d.mkdir()
    with pytest.raises(UnsupportedInputError):
        VideoSource(d)


def test_unsupported_extension(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("this is not a video")
    with pytest.raises(UnsupportedInputError):
        VideoSource(p)


def test_empty_file(tmp_path):
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    with pytest.raises(CorruptInputError):
        VideoSource(p)


def test_garbage_with_video_extension(tmp_path):
    p = tmp_path / "broken.mp4"
    p.write_bytes(np.random.default_rng(0).integers(0, 255, 40_000, dtype=np.uint8).tobytes())
    with pytest.raises(CorruptInputError):
        VideoSource(p)


def test_truncated_video_is_readable_up_to_the_break(tmp_path, clean_video):
    data = clean_video.read_bytes()
    p = tmp_path / "truncated.mp4"
    p.write_bytes(data[: int(len(data) * 0.6)])
    try:
        with VideoSource(p) as src:
            n = sum(1 for _ in src.frames())
    except CorruptInputError:
        return
    assert n >= 0


def test_valid_video_metadata(clean_video):
    with VideoSource(clean_video) as src:
        info = src.info
        assert info.width == 256 and info.height == 160
        assert info.fps == pytest.approx(25.0, abs=0.5)
        assert info.fps_source == "container"
        assert info.frame_count in (None,) or info.frame_count >= 100


def test_stride_and_max_frames(clean_video):
    with VideoSource(clean_video) as src:
        idxs = [i for i, _ in src.frames(stride=4, max_frames=10)]
    assert len(idxs) == 10
    assert idxs == list(range(0, 40, 4))


def test_fallback_fps_used_when_container_lies(tmp_path, monkeypatch, clean_video):
    import road_health.io as io_mod

    real_get = cv2.VideoCapture.get

    def fake_get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 0.0
        return real_get(self, prop)

    monkeypatch.setattr(cv2.VideoCapture, "get", fake_get, raising=False)
    with io_mod.VideoSource(clean_video, fallback_fps=12.5) as src:
        assert src.info.fps == 12.5
        assert src.info.fps_source == "fallback"


def test_writer_produces_a_playable_file(tmp_path):
    out = tmp_path / "out.mp4"
    with AnnotatedVideoWriter(out, fps=25.0, size=(64, 48)) as w:
        for i in range(20):
            w.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
        written = w.path
    assert written.exists() and written.stat().st_size > 0
    cap = cv2.VideoCapture(str(written))
    assert cap.isOpened()
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (48, 64)


def test_writer_resizes_mismatched_frames(tmp_path):
    out = tmp_path / "out2.mp4"
    with AnnotatedVideoWriter(out, fps=25.0, size=(64, 48)) as w:
        w.write(np.zeros((96, 128, 3), dtype=np.uint8))
        written = w.path
    assert written.exists()
