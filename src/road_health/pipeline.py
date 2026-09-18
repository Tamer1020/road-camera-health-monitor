"""Pipeline orchestration.

    video -> validate -> decode -> preprocess -> metrics -> rules
          -> temporal persistence -> status -> annotate -> reports

Two entry points: ``calibrate`` (characterise a healthy camera) and
``inspect_video`` (monitor a clip against that characterisation).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from road_health.baseline import Baseline
from road_health.config import Config
from road_health.errors import BaselineError, CorruptInputError
from road_health.health_checks import HealthChecker
from road_health.io import AnnotatedVideoWriter, VideoSource
from road_health.metrics import FrameMetrics, MetricExtractor, prepare_gray
from road_health.reporting import (
    Benchmark,
    MetricsCsvWriter,
    write_events,
    write_metadata,
)
from road_health.temporal import EventTracker, frame_status
from road_health.visualization import annotate

log = logging.getLogger("road_health.pipeline")


class Stopwatch:
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self._t0 = 0.0
        self._name = ""

    def start(self, name: str) -> "Stopwatch":
        self._name = name
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> None:
        self.totals[self._name] = self.totals.get(self._name, 0.0) + (
            time.perf_counter() - self._t0
        )

    def __enter__(self) -> "Stopwatch":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _roi_tuple(cfg: Config) -> Optional[Tuple[float, float, float, float]]:
    if cfg.analysis.roi is None:
        return None
    return tuple(cfg.analysis.roi)


def _roi_pixels(cfg: Config, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    roi = _roi_tuple(cfg)
    if roi is None:
        return None
    x, y, w, h = roi
    return (
        int(round(x * width)),
        int(round(y * height)),
        max(1, int(round(w * width))),
        max(1, int(round(h * height))),
    )


def _make_extractor(cfg: Config) -> MetricExtractor:
    bp = cfg.checks["blocked"].params if "blocked" in cfg.checks else {}
    return MetricExtractor(
        tile_rows=cfg.analysis.tile_rows,
        tile_cols=cfg.analysis.tile_cols,
        tile_abs_std=float(bp.get("tile_abs_std", 4.0)),
        tile_rel_std=float(bp.get("tile_rel_std", 0.35)),
        tile_rel_edge=float(bp.get("tile_rel_edge", 0.30)),
        gain_floor=float(bp.get("gain_floor", 0.25)),
    )


def calibrate(
    input_path: str | Path,
    config: Config,
    output_path: str | Path,
    max_samples: int = 200,
    max_seconds: Optional[float] = None,
) -> Baseline:
    with VideoSource(input_path, fallback_fps=config.analysis.fallback_fps) as src:
        info = src.info
        total = info.frame_count
        if max_seconds is not None and total:
            total = min(total, int(max_seconds * info.fps))
        stride = max(1, (total // max_samples)) if total else config.analysis.stride
        log.info(
            "calibrating from %s (%dx%d @ %.2f fps, %s frames) sampling every %d frames",
            info.path, info.width, info.height, info.fps,
            info.frame_count if info.frame_count else "unknown", stride,
        )

        extractor = _make_extractor(config)
        grays: List[np.ndarray] = []
        metrics: List[FrameMetrics] = []
        roi = _roi_tuple(config)
        for idx, frame in src.frames(stride=stride, max_frames=max_samples):
            gray = prepare_gray(frame, config.analysis.max_side, roi)
            grays.append(gray)
            metrics.append(extractor(gray, idx, idx / info.fps))

    if len(metrics) < 5:
        raise BaselineError(
            f"only {len(metrics)} frames could be sampled; a baseline needs at least 5. "
            "Use a longer calibration clip."
        )

    base = Baseline.from_samples(
        metrics,
        grays,
        source=str(input_path),
        max_side=config.analysis.max_side,
        roi=config.analysis.roi,
        tile_rows=config.analysis.tile_rows,
        tile_cols=config.analysis.tile_cols,
    )
    saved = base.save(output_path)
    log.info("baseline written to %s (%d sampled frames)", saved, base.n_frames)
    return base


def _auto_baseline(
    input_path: str | Path, config: Config, fps: float
) -> Optional[Baseline]:
    n = max(
        config.baseline.min_warmup_frames,
        int(config.baseline.warmup_seconds * fps),
    )
    with VideoSource(input_path, fallback_fps=config.analysis.fallback_fps) as src:
        extractor = _make_extractor(config)
        roi = _roi_tuple(config)
        grays, metrics = [], []
        for idx, frame in src.frames(stride=1, max_frames=n):
            gray = prepare_gray(frame, config.analysis.max_side, roi)
            grays.append(gray)
            metrics.append(extractor(gray, idx, idx / fps))
    if len(metrics) < 5:
        log.warning("clip too short for auto-baseline (%d frames); continuing without", len(metrics))
        return None
    return Baseline.from_samples(
        metrics, grays,
        source=f"auto:{input_path}",
        max_side=config.analysis.max_side,
        roi=config.analysis.roi,
        tile_rows=config.analysis.tile_rows,
        tile_cols=config.analysis.tile_cols,
    )


@dataclass
class RunResult:
    output_dir: Path
    status_counts: Dict[str, int]
    condition_frames: Dict[str, int]
    events: List[Dict[str, Any]]
    benchmark: Dict[str, Any]
    baseline_source: str
    metadata_path: Path


def inspect_video(
    input_path: str | Path,
    config: Config,
    output_dir: str | Path,
    camera_id: str = "",
    write_video: Optional[bool] = None,
) -> RunResult:
    t_wall = time.perf_counter()
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sw = Stopwatch()

    with VideoSource(input_path, fallback_fps=config.analysis.fallback_fps) as src:
        info = src.info
        log.info(
            "input: %s  %dx%d  %.3f fps (%s)  frames=%s  fourcc=%s",
            info.path, info.width, info.height, info.fps, info.fps_source,
            info.frame_count if info.frame_count is not None else "unknown",
            info.fourcc or "?",
        )

        baseline: Optional[Baseline] = None
        baseline_source = "none"
        baseline_warnings: List[str] = []
        if config.baseline.mode == "file":
            baseline = Baseline.load(config.baseline.path)
            baseline_source = f"file:{config.baseline.path}"
            baseline_warnings = baseline.check_compatible(
                config.analysis.max_side,
                config.analysis.roi,
                config.analysis.tile_rows,
                config.analysis.tile_cols,
            )
            for w in baseline_warnings:
                log.warning("baseline incompatibility: %s", w)
        elif config.baseline.mode == "auto":
            baseline = _auto_baseline(input_path, config, info.fps)
            if baseline is not None:
                baseline_source = (
                    f"auto:first {baseline.n_frames} frames "
                    f"(~{baseline.n_frames / info.fps:.1f}s)"
                )
                baseline_warnings.append(
                    "auto-baseline assumes the start of the clip is healthy"
                )
        log.info("baseline: %s", baseline_source)

        extractor = _make_extractor(config)
        if baseline is not None:
            baseline.attach_to(extractor)

        checker = HealthChecker(config, baseline, info.fps)
        conditions = {
            n: c.severity for n, c in config.checks.items() if c.enabled
        }
        tracker = EventTracker(
            conditions,
            {n: config.enter_frames(n, info.fps) for n in conditions},
            {n: config.exit_frames(n, info.fps) for n in conditions},
        )

        do_video = config.output.write_video if write_video is None else write_video
        writer = None
        if do_video:
            writer = AnnotatedVideoWriter(
                outdir / "annotated_video.mp4",
                fps=info.fps / max(1, config.analysis.stride),
                size=(info.width, info.height),
                codec=config.output.video_codec,
            )

        csv_writer = None
        if config.output.csv:
            extra_cols = ["status"] + [f"flag_{n}" for n in conditions] + [
                f"active_{n}" for n in conditions
            ]
            csv_writer = MetricsCsvWriter(outdir / "frame_metrics.csv", extra_cols)

        roi = _roi_tuple(config)
        roi_px = _roi_pixels(config, info.width, info.height)
        status_counts: Counter = Counter()
        condition_frames: Counter = Counter()
        n_read = 0
        last_index, last_time = 0, 0.0

        decode_t0 = time.perf_counter()
        for idx, frame in src.frames(stride=config.analysis.stride):
            sw.totals["decode"] = sw.totals.get("decode", 0.0) + (
                time.perf_counter() - decode_t0
            )
            n_read += 1
            ts = idx / info.fps

            with sw.start("preprocess"):
                gray = prepare_gray(frame, config.analysis.max_side, roi)
            with sw.start("metrics"):
                m = extractor(gray, idx, ts)
            with sw.start("checks"):
                results = checker(m)
            with sw.start("temporal"):
                active = tracker.update(idx, ts, results)
                status = frame_status(active, conditions)

            status_counts[status] += 1
            for name, is_active in active.items():
                if is_active:
                    condition_frames[name] += 1

            if csv_writer is not None:
                extra: Dict[str, Any] = {"status": status}
                for name in conditions:
                    r = results.get(name)
                    extra[f"flag_{name}"] = int(bool(r and r.triggered))
                    extra[f"active_{name}"] = int(bool(active.get(name)))
                csv_writer.write(m, extra)

            if writer is not None and (n_read - 1) % max(1, config.output.annotate_stride) == 0:
                with sw.start("annotate"):
                    vis = annotate(
                        frame,
                        status=status,
                        frame_index=idx,
                        timestamp_s=ts,
                        active=active,
                        results=results,
                        dead_mask=m.dead_tile_mask,
                        tile_rows=config.analysis.tile_rows,
                        tile_cols=config.analysis.tile_cols,
                        roi_px=roi_px,
                        camera_id=camera_id,
                    )
                with sw.start("write_video"):
                    writer.write(vis)

            last_index, last_time = idx, ts
            decode_t0 = time.perf_counter()

        tracker.finalize(last_index, last_time)
        if writer is not None:
            writer.close()
        if csv_writer is not None:
            csv_writer.close()

    if n_read == 0:
        raise CorruptInputError(f"no frames could be decoded from {input_path}")

    bench = Benchmark(
        n_frames_read=last_index + 1,
        n_frames_analysed=n_read,
        stage_seconds=sw.totals,
        wall_seconds=time.perf_counter() - t_wall,
        input_fps=info.fps,
    )

    total = max(1, n_read)
    summary = {
        "frames_analysed": n_read,
        "status_frames": dict(status_counts),
        "status_fraction": {k: round(v / total, 4) for k, v in status_counts.items()},
        "usable_fraction": round(
            (status_counts.get("OK", 0) + status_counts.get("DEGRADED", 0)) / total, 4
        ),
        "condition_active_frames": dict(condition_frames),
        "n_events": len(tracker.events),
    }

    if config.output.events:
        write_events(outdir / "events.json", tracker.events, summary)

    meta = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "camera_id": camera_id,
        "input": {
            "path": info.path,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "fps_source": info.fps_source,
            "frame_count": info.frame_count,
            "fourcc": info.fourcc,
            "duration_s": info.duration_s,
        },
        "baseline": {
            "source": baseline_source,
            "warnings": baseline_warnings,
            "stats": baseline.summary() if baseline else None,
            "n_frames": baseline.n_frames if baseline else 0,
        },
        "config": config.to_dict(),
        "effective_thresholds": {
            n: {
                "severity": conditions[n],
                "enter_frames": config.enter_frames(n, info.fps),
                "exit_frames": config.exit_frames(n, info.fps),
            }
            for n in conditions
        },
        "timebase": "frame_index / container_fps (CAP_PROP_POS_MSEC is not trusted)",
        "summary": summary,
        "benchmark": bench.to_dict(),
    }
    if config.output.metadata:
        write_metadata(outdir / "run_metadata.json", meta)

    log.info(
        "done: %d frames, %d events, %.1f fps end-to-end (%.2fx realtime)",
        n_read, len(tracker.events),
        bench.to_dict()["pipeline_fps_end_to_end"],
        bench.to_dict()["realtime_factor"],
    )

    return RunResult(
        output_dir=outdir,
        status_counts=dict(status_counts),
        condition_frames=dict(condition_frames),
        events=[e.to_dict() for e in tracker.events],
        benchmark=bench.to_dict(),
        baseline_source=baseline_source,
        metadata_path=outdir / "run_metadata.json",
    )
