#!/usr/bin/env python3
"""Evaluate the monitor against the ground-truthed corpus."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_health.baseline import Baseline
from road_health.config import load_config
from road_health.pipeline import inspect_video
from road_health.reporting import setup_logging

CONDITIONS = ["blur", "dark", "bright", "low_contrast", "blocked", "frozen", "moved", "shake"]
RAMP_FRAMES = 25


def read_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cols: Dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]) if r[key] != "" else np.nan)
            except ValueError:
                vals.append(np.nan)
        cols[key] = np.array(vals)
    return cols


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def window_mask(n: int, window: Optional[List[int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    if window is not None:
        m[window[0] : window[1] + 1] = True
    return m


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = x[lo : i + 1].std()
    return out


def sweep_rules(base: Baseline, cols: Dict[str, np.ndarray], cfg) -> Dict[str, Any]:
    st = base.stats
    reliable = (
        (cols["edge_density"] >= 0.5 * st["edge_density"].median)
        & (cols["mean_luma"] >= 0.55 * st["mean_luma"].median)
        & (cols["mean_luma"] <= 1.60 * st["mean_luma"].median)
        & (cols["contrast_span"] >= 0.5 * st["contrast_span"].median)
    )
    jitter = _rolling_std(cols["shift_frac"], 25)

    def exposure_ok(lo, hi):
        r = cols["mean_luma"] / st["mean_luma"].median
        return (r >= lo) & (r <= hi)

    return {
        "blur": {
            "param": "checks.blur.params.rel_min",
            "values": [0.15, 0.25, 0.35, 0.50, 0.65, 0.80],
            "fn": lambda k: exposure_ok(
                cfg.checks["blur"].params["exposure_lo"],
                cfg.checks["blur"].params["exposure_hi"],
            )
            & (cols["reblur_ratio"] < k * st["reblur_ratio"].median),
        },
        "dark": {
            "param": "checks.dark.params.rel_mean_luma",
            "values": [0.25, 0.35, 0.45, 0.55, 0.70],
            "fn": lambda k: (cols["clip_low_frac"] >= cfg.checks["dark"].params["clip_low_frac"])
            | (
                (cols["mean_luma"] < k * st["mean_luma"].median)
                & (cols["contrast_span"] < 0.55 * st["contrast_span"].median)
            ),
        },
        "bright": {
            "param": "checks.bright.params.clip_high_frac",
            "values": [0.05, 0.10, 0.20, 0.35, 0.50],
            "fn": lambda k: (cols["clip_high_frac"] >= k)
            | (
                (cols["mean_luma"] > max(205.0, 1.7 * st["mean_luma"].median))
                & (cols["contrast_span"] < 60.0)
            ),
        },
        "low_contrast": {
            "param": "checks.low_contrast.params.rel_span",
            "values": [0.30, 0.40, 0.50, 0.65, 0.80],
            "fn": lambda k: (cols["contrast_span"] < k * st["contrast_span"].median)
            & (cols["mean_luma"] >= 0.55 * st["mean_luma"].median)
            & (cols["mean_luma"] <= 1.60 * st["mean_luma"].median)
            & (cols["clip_low_frac"] < 0.05)
            & (cols["clip_high_frac"] < 0.05),
        },
        "blocked": {
            "param": "checks.blocked.params.dead_tile_frac",
            "values": [0.20, 0.30, 0.35, 0.45, 0.60, 0.80],
            "fn": lambda k: (cols["std_luma"] < cfg.checks["blocked"].params["uniform_std_max"])
            | (
                (cols["dead_tile_frac"] >= k)
                & exposure_ok(
                    cfg.checks["blocked"].params["exposure_lo"],
                    cfg.checks["blocked"].params["exposure_hi"],
                )
            ),
        },
        "frozen": {
            "param": "checks.frozen.params.noise_ratio",
            "values": [0.10, 0.20, 0.35, 0.50, 0.70],
            "fn": lambda k: (cols["frame_mad"] >= 0)
            & (cols["noise_sigma"] >= cfg.checks["frozen"].params["min_noise_sigma"])
            & (
                cols["frame_mad"]
                < np.maximum(
                    cfg.checks["frozen"].params["abs_mad"], k * 1.128 * cols["noise_sigma"]
                )
            ),
        },
        "moved": {
            "param": "checks.moved.params.shift_frac",
            "values": [0.01, 0.02, 0.04, 0.07, 0.12],
            "fn": lambda k: reliable
            & (
                (cols["shift_frac"] > k)
                | (cols["edge_corr"] < cfg.checks["moved"].params["edge_corr_min"])
            ),
        },
        "shake": {
            "param": "checks.shake.params.jitter_frac",
            "values": [0.0004, 0.0008, 0.0012, 0.0020, 0.0040],
            "fn": lambda k: reliable & (jitter > k),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="sample_data/eval/labels.json")
    ap.add_argument("--baseline", default="sample_data/eval/baseline.json")
    ap.add_argument("--outdir", "-o", default="outputs/evaluation")
    ap.add_argument("--config", "-c", default="configs/default.yaml")
    ap.add_argument("--baseline-mode", choices=["file", "auto", "none"], default="file")
    ap.add_argument("--video", action="store_true", help="also write annotated videos")
    args = ap.parse_args()

    setup_logging(None, verbose=False)
    import logging
    logging.getLogger("road_health").setLevel(logging.WARNING)

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    outdir = Path(args.outdir) / args.baseline_mode
    outdir.mkdir(parents=True, exist_ok=True)

    base = Baseline.load(args.baseline)

    per_clip: Dict[str, Any] = {}
    cross: Dict[str, Dict[str, float]] = {}
    throughput: List[Dict[str, float]] = []
    pooled_cols: Dict[str, List[np.ndarray]] = defaultdict(list)
    pooled_pos: Dict[str, List[np.ndarray]] = defaultdict(list)

    for clip_name, spec in labels["clips"].items():
        cfg = load_config(args.config)
        cfg.baseline.mode = args.baseline_mode
        if args.baseline_mode == "file":
            cfg.baseline.path = args.baseline
        run_dir = outdir / clip_name
        res = inspect_video(
            spec["path"], cfg, run_dir,
            camera_id=f"EVAL/{clip_name}",
            write_video=args.video,
        )
        throughput.append(
            {
                "clip": clip_name,
                "fps_end_to_end": res.benchmark["pipeline_fps_end_to_end"],
                "analysis_fps": res.benchmark["analysis_fps_excluding_io"],
                "realtime_factor": res.benchmark["realtime_factor"],
                "ms_per_frame": res.benchmark["ms_per_analysed_frame"],
            }
        )

        cols = read_csv(run_dir / "frame_metrics.csv")
        n = len(cols["frame_index"])
        gt = window_mask(n, spec["fault_window"])
        cond = spec["condition"]

        cross[clip_name] = {
            c: float(cols[f"active_{c}"][gt].mean()) if gt.any() and f"active_{c}" in cols else 0.0
            for c in CONDITIONS
            if f"active_{c}" in cols
        }

        for key, arr in cols.items():
            pooled_cols[key].append(arr)
        for c in CONDITIONS:
            pooled_pos[c].append(gt if cond == c else np.zeros(n, dtype=bool))

        entry: Dict[str, Any] = {
            "condition": cond,
            "fault_window": spec["fault_window"],
            "n_frames": n,
        }

        if cond is None:
            entry["false_alarm_frame_rate"] = {
                c: float(cols[f"active_{c}"].mean())
                for c in CONDITIONS
                if f"active_{c}" in cols
            }
            events = json.loads((run_dir / "events.json").read_text())["events"]
            entry["false_alarm_events"] = len(events)
            entry["events"] = [
                {k: e[k] for k in ("condition", "start_frame", "duration_s")} for e in events
            ]
            entry["status_fraction"] = {
                k: round(v / n, 4) for k, v in res.status_counts.items()
            }
        else:
            act = cols[f"active_{cond}"].astype(bool)
            start, end = spec["fault_window"]
            hold = cfg.enter_frames(cond, labels["fps"])
            clear = cfg.exit_frames(cond, labels["fps"])

            settle = min(start + RAMP_FRAMES + hold, max(start, end - 1))
            steady = np.zeros(n, dtype=bool)
            steady[settle : end + 1] = True

            tail = np.zeros(n, dtype=bool)
            tail[end + 1 : min(n, end + 1 + clear + RAMP_FRAMES)] = True
            true_neg_region = ~gt & ~tail

            tp = int((act & gt).sum())
            fp = int((act & ~gt).sum())
            fn = int((~act & gt).sum())
            p, r, f1 = prf(tp, fp, fn)
            idx = np.where(act & gt)[0]
            latency = int(idx[0] - start) if idx.size else None
            after = np.where(~act[end + 1 :])[0]
            recovery = int(after[0]) if after.size else None
            entry.update(
                {
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                    "recall_steady_state": round(
                        float(act[steady].mean()) if steady.any() else 0.0, 4
                    ),
                    "false_positive_frames_outside_fault_and_recovery": int(
                        (act & true_neg_region).sum()
                    ),
                    "recovery_latency_frames": recovery,
                    "clear_down_frames": clear,
                    "detect_latency_frames": latency,
                    "detect_latency_s": None if latency is None else round(latency / labels["fps"], 2),
                    "hold_down_frames": cfg.enter_frames(cond, labels["fps"]),
                    "events": len(
                        [
                            e
                            for e in json.loads((run_dir / "events.json").read_text())["events"]
                            if e["condition"] == cond
                        ]
                    ),
                    "co_active_in_window": {
                        c: round(v, 3) for c, v in cross[clip_name].items() if c != cond and v > 0.02
                    },
                }
            )
        per_clip[clip_name] = entry

    pooled = {k: np.concatenate(v) for k, v in pooled_cols.items()}
    pos = {c: np.concatenate(v) for c, v in pooled_pos.items()}
    cfg = load_config(args.config)
    rules = sweep_rules(base, pooled, cfg)
    sweep: Dict[str, Any] = {}
    for cond, spec in rules.items():
        rows = []
        for k in spec["values"]:
            flag = np.asarray(spec["fn"](k), dtype=bool)
            g = pos[cond]
            tp = int((flag & g).sum())
            fp = int((flag & ~g).sum())
            fn = int((~flag & g).sum())
            p, r, f1 = prf(tp, fp, fn)
            rows.append(
                {"threshold": k, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
            )
        sweep[cond] = {"param": spec["param"], "operating_point": cfg.checks[cond].params, "curve": rows}

    summary = {
        "baseline_mode": args.baseline_mode,
        "corpus": {
            "clips": len(labels["clips"]),
            "frames_per_clip": labels["n_frames"],
            "fps": labels["fps"],
            "source": labels["source"],
        },
        "per_clip": per_clip,
        "cross_condition_activity_in_fault_window": cross,
        "threshold_sweep_raw_flags": sweep,
        "throughput": throughput,
        "throughput_median_fps_end_to_end": round(
            statistics.median(t["fps_end_to_end"] for t in throughput), 2
        ),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(outdir / "report.md", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "threshold_sweep_raw_flags"}, indent=2)[:4000])
    print(f"\nwrote {outdir/'summary.json'} and {outdir/'report.md'}")
    return 0


def write_report(path: Path, s: Dict[str, Any]) -> None:
    L: List[str] = []
    L.append(f"# Evaluation report (baseline mode: `{s['baseline_mode']}`)\n")
    c = s["corpus"]
    L.append(
        f"Corpus: {c['clips']} clips x {c['frames_per_clip']} frames @ {c['fps']} fps, "
        f"generated from `{c['source']}`.\n"
    )

    L.append("\n## Detection performance (post-temporal state, frame level)\n")
    L.append(
        "| clip | condition | precision | recall (whole window) | recall (steady state) "
        "| F1 | detect latency | hold-down | FP frames | events |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for clip, e in s["per_clip"].items():
        if e["condition"] is None:
            continue
        lat = (
            f"{e['detect_latency_frames']} fr ({e['detect_latency_s']} s)"
            if e["detect_latency_frames"] is not None
            else "NOT DETECTED"
        )
        L.append(
            f"| {clip} | {e['condition']} | {e['precision']:.3f} | {e['recall']:.3f} | "
            f"{e['recall_steady_state']:.3f} | {e['f1']:.3f} | {lat} | "
            f"{e['hold_down_frames']} fr | "
            f"{e['false_positive_frames_outside_fault_and_recovery']} | {e['events']} |"
        )

    normal = s["per_clip"].get("normal")
    if normal:
        L.append("\n## False alarms on the clean clip\n")
        L.append(f"Events raised: **{normal['false_alarm_events']}**\n")
        L.append("| condition | fraction of frames active |")
        L.append("|---|---|")
        for k, v in normal["false_alarm_frame_rate"].items():
            L.append(f"| {k} | {v:.4f} |")
        L.append(f"\nStatus distribution: `{normal['status_fraction']}`\n")

    L.append("\n## Cross-condition activity inside each fault window\n")
    conds = list(next(iter(s["cross_condition_activity_in_fault_window"].values())).keys())
    L.append("| clip | " + " | ".join(conds) + " |")
    L.append("|---" * (len(conds) + 1) + "|")
    for clip, row in s["cross_condition_activity_in_fault_window"].items():
        L.append(f"| {clip} | " + " | ".join(f"{row.get(k, 0):.2f}" for k in conds) + " |")

    L.append("\n## Threshold sensitivity (raw per-frame flags, pooled over all clips)\n")
    for cond, sp in s["threshold_sweep_raw_flags"].items():
        L.append(f"\n**{cond}** - `{sp['param']}`\n")
        L.append("| threshold | precision | recall | F1 |")
        L.append("|---|---|---|---|")
        for row in sp["curve"]:
            L.append(
                f"| {row['threshold']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
            )

    L.append("\n## Throughput\n")
    L.append("| clip | end-to-end fps | analysis-only fps | x realtime |")
    L.append("|---|---|---|---|")
    for t in s["throughput"]:
        L.append(
            f"| {t['clip']} | {t['fps_end_to_end']:.1f} | {t['analysis_fps']:.1f} | "
            f"{t['realtime_factor']:.2f} |"
        )
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
