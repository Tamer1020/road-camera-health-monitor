#!/usr/bin/env python3
"""End-to-end throughput benchmark.

Measures the whole pipeline, not one OpenCV call.  Three configurations are
reported because they answer three different questions:

  full            what an operator actually gets, annotated video included
  analysis-only   how fast the health logic can consume frames (--no-video)
  stride 3        the cheapest throughput lever, and what it costs in latency
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from road_health.config import load_config
from road_health.pipeline import inspect_video
from road_health.reporting import setup_logging


def run(name: str, video: str, cfg, outdir: Path, write_video: bool):
    res = inspect_video(video, cfg, outdir / name, write_video=write_video)
    b = res.benchmark
    return {
        "configuration": name,
        "frames": b["frames_analysed"],
        "wall_s": b["wall_seconds"],
        "end_to_end_fps": b["pipeline_fps_end_to_end"],
        "analysis_fps": b["analysis_fps_excluding_io"],
        "realtime_factor": b["realtime_factor"],
        "ms_per_frame": b["ms_per_analysed_frame"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="sample_data/eval/normal.mp4")
    ap.add_argument("--baseline", "-b", default="sample_data/eval/baseline.json")
    ap.add_argument("--config", "-c", default="configs/default.yaml")
    ap.add_argument("--outdir", "-o", default="outputs/benchmark")
    args = ap.parse_args()

    setup_logging(None, False)
    import logging

    logging.getLogger("road_health").setLevel(logging.WARNING)
    outdir = Path(args.outdir)

    def cfg(**over):
        c = load_config(args.config)
        c.baseline.mode = "file"
        c.baseline.path = args.baseline
        for k, v in over.items():
            setattr(c.analysis, k, v)
        return c

    rows = [
        run("full_with_annotated_video", args.input, cfg(), outdir, True),
        run("analysis_only", args.input, cfg(), outdir, False),
        run("analysis_only_stride3", args.input, cfg(stride=3), outdir, False),
        run("analysis_only_maxside320", args.input, cfg(max_side=320), outdir, False),
    ]

    cap = cv2.VideoCapture(args.input)
    src = {
        "path": args.input,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
    }
    cap.release()

    payload = {
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "cpu_count_visible": __import__("os").cpu_count(),
            "note": "single-threaded pipeline; OpenCV may use internal threading",
        },
        "source": src,
        "results": rows,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "benchmark.json").write_text(json.dumps(payload, indent=2))

    print(f"source: {src['width']}x{src['height']} @ {src['fps']} fps\n")
    hdr = f"{'configuration':30s} {'fps e2e':>9s} {'analysis fps':>13s} {'x realtime':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['configuration']:30s} {r['end_to_end_fps']:9.1f} "
            f"{r['analysis_fps']:13.1f} {r['realtime_factor']:11.2f}"
        )
    print("\nper-stage cost, full pipeline (ms per analysed frame):")
    for k, v in sorted(rows[0]["ms_per_frame"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:7.3f}")
    print(f"\nwrote {outdir / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
