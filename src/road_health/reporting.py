"""Run artefacts: CSV, events, metadata, log.

The guiding rule for every artefact: an engineer who was not present for the
run must be able to reconstruct what happened and why, without re-running it.
That means recording the *inputs to the decision* (the config actually used,
the baseline actually loaded, the fallback paths actually taken) and not only
the decision.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np

from road_health import __version__
from road_health.metrics import FrameMetrics
from road_health.temporal import Event


def setup_logging(log_path: Optional[Path], verbose: bool = False) -> logging.Logger:
    root = logging.getLogger("road_health")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(stream)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)
    return root


class MetricsCsvWriter:
    """Streaming CSV writer - never holds the whole run in memory."""

    def __init__(self, path: Path, extra_columns: Iterable[str] = ()):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", newline="", encoding="utf-8")
        self._columns = FrameMetrics.csv_columns() + list(extra_columns)
        self._writer = csv.DictWriter(self._fh, fieldnames=self._columns)
        self._writer.writeheader()
        self.n_rows = 0

    def write(self, metrics: FrameMetrics, extra: Dict[str, Any]) -> None:
        row = metrics.to_row()
        row.update(extra)
        self._writer.writerow({k: row.get(k, "") for k in self._columns})
        self.n_rows += 1

    def close(self) -> None:
        self._fh.close()


@dataclass
class Benchmark:
    """End-to-end and per-stage timing.

    The distinction that matters: ``analysis_fps`` is the rate at which the
    health logic can consume frames; ``pipeline_fps`` includes decoding and
    writing the annotated video.  Quoting the first as if it were the second
    is the standard way benchmark numbers become fiction.
    """

    n_frames_read: int = 0
    n_frames_analysed: int = 0
    stage_seconds: Dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0
    input_fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        analysed = max(1, self.n_frames_analysed)
        analysis_time = (
            self.stage_seconds.get("preprocess", 0.0)
            + self.stage_seconds.get("metrics", 0.0)
            + self.stage_seconds.get("checks", 0.0)
            + self.stage_seconds.get("temporal", 0.0)
        )
        source_fps = self.n_frames_read / max(1e-9, self.wall_seconds)
        return {
            "frames_read": self.n_frames_read,
            "frames_analysed": self.n_frames_analysed,
            "wall_seconds": round(self.wall_seconds, 3),
            "input_fps": round(self.input_fps, 3),
            "pipeline_fps_end_to_end": round(source_fps, 2),
            "analysed_fps_end_to_end": round(
                self.n_frames_analysed / max(1e-9, self.wall_seconds), 2
            ),
            "analysis_fps_excluding_io": round(analysed / max(1e-9, analysis_time), 2),
            "realtime_factor": round(source_fps / max(1e-9, self.input_fps), 2),
            "ms_per_analysed_frame": {
                k: round(1000.0 * v / analysed, 3) for k, v in sorted(self.stage_seconds.items())
            },
            "stage_seconds": {k: round(v, 3) for k, v in sorted(self.stage_seconds.items())},
        }


def write_events(path: Path, events: List[Event], summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "road-health/events/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary,
        "events": [e.to_dict() for e in events],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("schema", "road-health/run-metadata/1")
    payload["tool"] = {
        "name": "road-health",
        "version": __version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
