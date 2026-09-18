"""Input validation, frame decoding and output video writing.

Everything that can go wrong with an operator-supplied path is turned into a
typed error here, so the pipeline itself never has to guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from road_health.errors import (
    CorruptInputError,
    InputNotFoundError,
    UnsupportedInputError,
)

log = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".mpg", ".mpeg", ".webm", ".ts"}

FPS_SANE_RANGE = (0.5, 240.0)


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    fps_source: str
    frame_count: Optional[int]
    fourcc: str
    duration_s: Optional[float]


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return ""
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")


class VideoSource:
    """A validated, iterable video source."""

    def __init__(self, path: str | Path, fallback_fps: float = 25.0):
        self.path = Path(path)
        self._validate_path()
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise CorruptInputError(
                f"OpenCV could not open '{self.path}'. The file exists but is not a "
                f"decodable video (truncated, unsupported codec, or not a video at all)."
            )
        ok, first = self._cap.read()
        if not ok or first is None:
            self._cap.release()
            raise CorruptInputError(
                f"'{self.path}' opened but the first frame could not be decoded."
            )
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        h, w = first.shape[:2]
        raw_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if FPS_SANE_RANGE[0] <= raw_fps <= FPS_SANE_RANGE[1]:
            fps, fps_source = raw_fps, "container"
        else:
            log.warning(
                "container reported fps=%.3f which is outside the plausible range; "
                "falling back to %.2f fps", raw_fps, fallback_fps,
            )
            fps, fps_source = float(fallback_fps), "fallback"

        raw_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = raw_count if raw_count > 0 else None

        self.info = VideoInfo(
            path=str(self.path),
            width=w,
            height=h,
            fps=fps,
            fps_source=fps_source,
            frame_count=frame_count,
            fourcc=_fourcc_to_str(self._cap.get(cv2.CAP_PROP_FOURCC)),
            duration_s=(frame_count / fps) if frame_count else None,
        )

    def _validate_path(self) -> None:
        if not self.path.exists():
            raise InputNotFoundError(f"input path does not exist: {self.path}")
        if self.path.is_dir():
            raise UnsupportedInputError(
                f"input is a directory, expected a single video file: {self.path}"
            )
        if not self.path.is_file():
            raise UnsupportedInputError(f"input is not a regular file: {self.path}")
        if self.path.stat().st_size == 0:
            raise CorruptInputError(f"input file is empty (0 bytes): {self.path}")
        if self.path.suffix.lower() not in VIDEO_SUFFIXES:
            raise UnsupportedInputError(
                f"unsupported extension '{self.path.suffix}'. Supported: "
                + ", ".join(sorted(VIDEO_SUFFIXES))
            )

    def frames(self, stride: int = 1, max_frames: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(source_frame_index, bgr_frame)`` honouring ``stride``.

        Decode errors mid-stream end the iteration rather than raising: a
        truncated recording is a normal thing to receive from a real NVR, and
        the run should still produce a report for the part that decoded.
        """
        idx = -1
        emitted = 0
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            idx += 1
            if idx % stride:
                continue
            yield idx, frame
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                break

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class AnnotatedVideoWriter:
    """VideoWriter with a codec fallback.

    ``mp4v`` is not guaranteed to be present in every OpenCV build.  Rather
    than silently producing a 0-byte file (the OpenCV default behaviour), we
    verify the writer opened and fall back to MJPG/AVI.
    """

    def __init__(self, path: str | Path, fps: float, size: Tuple[int, int], codec: str = "mp4v"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.codec = codec
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*codec), fps, size
        )
        if not self._writer.isOpened():
            fallback = self.path.with_suffix(".avi")
            log.warning(
                "codec '%s' unavailable for %s; falling back to MJPG/%s",
                codec, self.path.name, fallback.name,
            )
            self._writer = cv2.VideoWriter(
                str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), fps, size
            )
            if not self._writer.isOpened():
                raise CorruptInputError(
                    "could not open any VideoWriter (tried mp4v and MJPG); "
                    "annotated video cannot be written"
                )
            self.path = fallback
            self.codec = "MJPG"

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
