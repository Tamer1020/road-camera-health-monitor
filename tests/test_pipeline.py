"""End-to-end pipeline, artefact generation and CLI exit codes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from road_health.baseline import Baseline
from road_health.cli import main as cli_main
from road_health.config import default_config
from road_health.errors import CorruptInputError, InputNotFoundError
from road_health.pipeline import calibrate, inspect_video


def _cfg(write_video=False, mode="auto"):
    cfg = default_config()
    cfg.baseline.mode = mode
    cfg.baseline.warmup_seconds = 1.0
    cfg.baseline.min_warmup_frames = 20
    cfg.output.write_video = write_video
    return cfg


def test_inspect_produces_all_artefacts(clean_video, tmp_path):
    out = tmp_path / "run"
    res = inspect_video(clean_video, _cfg(write_video=True), out, camera_id="CAM-1")
    assert (out / "frame_metrics.csv").is_file()
    assert (out / "events.json").is_file()
    assert (out / "run_metadata.json").is_file()
    video = list(out.glob("annotated_video.*"))
    assert video and video[0].stat().st_size > 0
    rows = list(csv.DictReader((out / "frame_metrics.csv").open()))
    assert len(rows) == res.benchmark["frames_analysed"]
    assert {"status", "frame_index", "reblur_ratio", "flag_blur", "active_blur"} <= set(rows[0])
    assert all(r["status"] in ("OK", "DEGRADED", "UNUSABLE") for r in rows)


def test_metadata_is_complete_and_honest(clean_video, tmp_path):
    out = tmp_path / "run"
    inspect_video(clean_video, _cfg(), out)
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["tool"]["version"] and meta["tool"]["opencv"]
    assert meta["input"]["fps_source"] in ("container", "fallback")
    assert meta["baseline"]["source"].startswith("auto:")
    assert any("auto-baseline" in w for w in meta["baseline"]["warnings"])
    assert meta["config"]["checks"]["blur"]["params"]["rel_min"] > 0
    assert set(meta["benchmark"]) >= {
        "frames_read", "pipeline_fps_end_to_end", "analysis_fps_excluding_io",
        "realtime_factor", "ms_per_analysed_frame",
    }


def test_clean_clip_raises_no_events(clean_video, tmp_path):
    res = inspect_video(clean_video, _cfg(), tmp_path / "run")
    assert res.events == []
    assert res.status_counts.get("OK", 0) == res.benchmark["frames_analysed"]


def test_frozen_clip_is_detected_end_to_end(frozen_video, tmp_path):
    cfg = _cfg()
    cfg.checks["frozen"].enter_seconds = 0.4
    res = inspect_video(frozen_video, cfg, tmp_path / "run")
    frozen_events = [e for e in res.events if e["condition"] == "frozen"]
    assert frozen_events, f"no frozen event; events={res.events}"
    ev = frozen_events[0]
    assert 55 <= ev["start_frame"] <= 70
    assert ev["severity"] == "UNUSABLE"
    assert res.status_counts.get("UNUSABLE", 0) > 0


def test_events_json_schema(frozen_video, tmp_path):
    cfg = _cfg()
    cfg.checks["frozen"].enter_seconds = 0.4
    out = tmp_path / "run"
    inspect_video(frozen_video, cfg, out)
    payload = json.loads((out / "events.json").read_text())
    assert payload["schema"] == "road-health/events/1"
    assert {"frames_analysed", "status_frames", "usable_fraction"} <= set(payload["summary"])
    for ev in payload["events"]:
        assert {"condition", "severity", "start_frame", "start_time_s",
                "detection_latency_frames", "co_active"} <= set(ev)


def test_stride_reduces_analysed_frames(clean_video, tmp_path):
    cfg = _cfg()
    cfg.analysis.stride = 4
    res = inspect_video(clean_video, cfg, tmp_path / "run")
    assert 25 <= res.benchmark["frames_analysed"] <= 35


def test_throughput_is_reported_in_source_frames(clean_video, tmp_path):
    cfg = _cfg()
    cfg.analysis.stride = 4
    b = inspect_video(clean_video, cfg, tmp_path / "run").benchmark
    assert b["frames_read"] > b["frames_analysed"]
    assert b["frames_read"] >= 4 * b["frames_analysed"] - 4
    assert b["pipeline_fps_end_to_end"] > b["analysed_fps_end_to_end"]
    assert b["realtime_factor"] == pytest.approx(
        b["pipeline_fps_end_to_end"] / b["input_fps"], rel=0.02
    )


def test_no_baseline_mode_marks_framing_checks_unavailable(clean_video, tmp_path):
    cfg = _cfg(mode="none")
    out = tmp_path / "run"
    res = inspect_video(clean_video, cfg, out)
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["baseline"]["source"] == "none"
    rows = list(csv.DictReader((out / "frame_metrics.csv").open()))
    assert all(r["active_moved"] == "0" for r in rows)
    assert res.baseline_source == "none"


def test_calibrate_writes_a_usable_baseline(clean_video, tmp_path):
    cfg = _cfg()
    b = calibrate(clean_video, cfg, tmp_path / "baseline.json", max_samples=40)
    assert b.n_frames >= 5
    assert (tmp_path / "baseline.json").is_file()
    assert (tmp_path / "baseline_reference.png").is_file()
    loaded = Baseline.load(tmp_path / "baseline.json")
    assert loaded.reference_gray is not None
    assert loaded.check_compatible(
        cfg.analysis.max_side, cfg.analysis.roi,
        cfg.analysis.tile_rows, cfg.analysis.tile_cols,
    ) == []


def test_run_with_file_baseline(clean_video, tmp_path):
    cfg = _cfg()
    calibrate(clean_video, cfg, tmp_path / "baseline.json", max_samples=40)
    cfg.baseline.mode = "file"
    cfg.baseline.path = str(tmp_path / "baseline.json")
    out = tmp_path / "run"
    res = inspect_video(clean_video, cfg, out)
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["baseline"]["source"].startswith("file:")
    assert meta["baseline"]["warnings"] == []
    assert res.events == []


def test_incompatible_baseline_is_flagged_not_silently_used(clean_video, tmp_path):
    cfg = _cfg()
    calibrate(clean_video, cfg, tmp_path / "baseline.json", max_samples=40)
    cfg.baseline.mode = "file"
    cfg.baseline.path = str(tmp_path / "baseline.json")
    cfg.analysis.max_side = 320
    out = tmp_path / "run"
    inspect_video(clean_video, cfg, out)
    meta = json.loads((out / "run_metadata.json").read_text())
    assert any("max_side" in w for w in meta["baseline"]["warnings"])


def test_pipeline_propagates_typed_errors(tmp_path):
    with pytest.raises(InputNotFoundError):
        inspect_video(tmp_path / "nope.mp4", _cfg(), tmp_path / "run")


def test_pipeline_rejects_corrupt_input(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x01\x02" * 5000)
    with pytest.raises(CorruptInputError):
        inspect_video(bad, _cfg(), tmp_path / "run")


def test_cli_exit_code_zero_on_healthy_clip(clean_video, tmp_path, capsys):
    code = cli_main(
        ["inspect-video", "-i", str(clean_video), "-o", str(tmp_path / "run"), "--no-video"]
    )
    assert code == 0
    assert (tmp_path / "run" / "run.log").is_file()
    assert json.loads(capsys.readouterr().out)["events"] == 0


def test_cli_exit_code_11_on_unusable(frozen_video, tmp_path, capsys):
    cfg_path = tmp_path / "fast.yaml"
    cfg_path.write_text("checks:\n  frozen:\n    enter_seconds: 0.4\n")
    code = cli_main(
        ["inspect-video", "-i", str(frozen_video), "-o", str(tmp_path / "run"),
         "-c", str(cfg_path), "--no-video"]
    )
    capsys.readouterr()
    assert code == 11


def test_cli_no_status_exit_always_returns_zero(frozen_video, tmp_path, capsys):
    cfg_path = tmp_path / "fast.yaml"
    cfg_path.write_text("checks:\n  frozen:\n    enter_seconds: 0.4\n")
    code = cli_main(
        ["inspect-video", "-i", str(frozen_video), "-o", str(tmp_path / "run"),
         "-c", str(cfg_path), "--no-video", "--no-status-exit"]
    )
    capsys.readouterr()
    assert code == 0


@pytest.mark.parametrize(
    "args,expected",
    [
        (["inspect-video", "-i", "/no/such/file.mp4", "-o", "OUT"], 2),
        (["inspect-video", "-i", "NOTES", "-o", "OUT"], 3),
    ],
)
def test_cli_error_exit_codes(tmp_path, args, expected, capsys):
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    args = [a.replace("NOTES", str(notes)).replace("OUT", str(tmp_path / "run")) for a in args]
    code = cli_main(args)
    capsys.readouterr()
    assert code == expected


def test_cli_calibrate_roundtrip(clean_video, tmp_path, capsys):
    code = cli_main(
        ["calibrate", "-i", str(clean_video), "-o", str(tmp_path / "b.json"),
         "--max-samples", "40"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "reblur_ratio" in payload["stats"]
