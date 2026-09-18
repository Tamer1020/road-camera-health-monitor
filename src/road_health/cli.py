"""Command-line interface.

Exit codes are meaningful so this can sit in a cron job or a pipeline stage:

    0  run completed, camera healthy for the whole clip
    1  unexpected internal error
    2  input not found
    3  unsupported input
    4  corrupt / undecodable input
    5  bad config
    6  bad baseline
    10 run completed, DEGRADED conditions were raised
    11 run completed, UNUSABLE conditions were raised
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from road_health import __version__
from road_health.config import load_config
from road_health.errors import RoadHealthError
from road_health.pipeline import calibrate, inspect_video
from road_health.reporting import setup_logging

EXIT_DEGRADED = 10
EXIT_UNUSABLE = 11


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="road-health",
        description="Camera health monitoring for road / ITS video feeds (classical CV).",
    )
    p.add_argument("--version", action="version", version=f"road-health {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    insp = sub.add_parser(
        "inspect-video",
        help="analyse a video and produce health diagnostics",
    )
    insp.add_argument("--input", "-i", required=True, help="path to the video file")
    insp.add_argument("--output", "-o", required=True, help="output run directory")
    insp.add_argument("--config", "-c", default=None, help="YAML config (optional)")
    insp.add_argument(
        "--baseline", "-b", default=None,
        help="baseline.json from `calibrate`; overrides baseline.mode to 'file'",
    )
    insp.add_argument(
        "--no-baseline", action="store_true",
        help="disable baselines entirely (absolute thresholds only)",
    )
    insp.add_argument("--camera-id", default="", help="label written into reports and overlay")
    insp.add_argument("--stride", type=int, default=None, help="override analysis.stride")
    insp.add_argument(
        "--no-video", action="store_true",
        help="skip the annotated video (use this when benchmarking analysis throughput)",
    )
    insp.add_argument("--verbose", "-v", action="store_true")
    insp.add_argument(
        "--no-status-exit", action="store_true",
        help="always exit 0 on a completed run",
    )

    cal = sub.add_parser(
        "calibrate",
        help="characterise a healthy camera and write a baseline",
    )
    cal.add_argument("--input", "-i", required=True, help="clip of the camera in a healthy state")
    cal.add_argument("--output", "-o", required=True, help="path to write baseline.json")
    cal.add_argument("--config", "-c", default=None)
    cal.add_argument("--max-samples", type=int, default=200)
    cal.add_argument("--max-seconds", type=float, default=None)
    cal.add_argument("--verbose", "-v", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config)

        if args.command == "calibrate":
            setup_logging(None, args.verbose)
            base = calibrate(
                args.input, cfg, args.output,
                max_samples=args.max_samples, max_seconds=args.max_seconds,
            )
            print(json.dumps({"baseline": str(args.output), "stats": base.summary()}, indent=2))
            return 0

        outdir = Path(args.output)
        outdir.mkdir(parents=True, exist_ok=True)
        setup_logging(outdir / "run.log", args.verbose)

        if args.no_baseline:
            cfg.baseline.mode = "none"
        elif args.baseline:
            cfg.baseline.mode = "file"
            cfg.baseline.path = args.baseline
        if args.stride is not None:
            cfg.analysis.stride = max(1, args.stride)

        result = inspect_video(
            args.input, cfg, outdir,
            camera_id=args.camera_id,
            write_video=(False if args.no_video else None),
        )

        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "baseline": result.baseline_source,
                    "status_frames": result.status_counts,
                    "conditions": result.condition_frames,
                    "events": len(result.events),
                    "fps_end_to_end": result.benchmark["pipeline_fps_end_to_end"],
                    "realtime_factor": result.benchmark["realtime_factor"],
                },
                indent=2,
            )
        )

        if args.no_status_exit:
            return 0
        if result.status_counts.get("UNUSABLE"):
            return EXIT_UNUSABLE
        if result.status_counts.get("DEGRADED"):
            return EXIT_DEGRADED
        return 0

    except RoadHealthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
