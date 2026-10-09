#!/usr/bin/env python3
"""Render a 60 s captioned portfolio video from actual road-health outputs.

Run the capture commands in docs/DEMO.md first. Requires Pillow and FFmpeg in
addition to the project dependencies. No states are inferred or drawn over the
source annotation; the large side labels are read from frame_metrics.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, N = 1280, 720, 25, 1500
BG = (9, 17, 29)
PANEL = (17, 30, 46)
WHITE = (235, 243, 251)
MUTED = (159, 181, 202)
CYAN = (67, 213, 216)
LINE = (40, 62, 81)
COLORS = {"OK": (97, 215, 160), "DEGRADED": (250, 196, 94), "UNUSABLE": (255, 118, 118)}
SHOTS = [
    (0, 125, "normal", 0, "01 / HEALTHY INPUT", "Camera reliability for perception pipelines"),
    (125, 288, "blur", 125, "02 / BLUR", "A fault appears before the alarm"),
    (288, 475, "blocked_partial", 125, "03 / PARTIAL OBSTRUCTION", "Persistent texture loss becomes an event"),
    (475, 625, "frozen", 125, "04 / FROZEN FRAMES", "The scene stalls. Frame time keeps advancing."),
]
CAPTIONS = [
    (0, 5, "Camera-health monitoring with OpenCV and NumPy.\nSynthetic scene + injected faults."),
    (5, 11.52, "Blur develops at source t = 6.00 s.\nThe alarm waits for persistent evidence."),
    (11.52, 19, "Partial obstruction removes local texture.\nThe actual output highlights the affected tiles."),
    (19, 25, "Frozen frames: the image stops while timestamps advance.\nThe monitor raises an UNUSABLE event."),
    (25, 35, "Same startup blur, same source timestamps.\nA healthy reference detects it; auto-baselining can learn it as normal."),
    (35, 43, "Events and frame metrics are exported as JSON and CSV.\nThe values shown come from this demo run."),
    (43, 54, "Previously reported results, with their scope.\nSynthetic development corpus; no claim of real-world accuracy."),
    (54, 60, "Code, evaluation and reproduction instructions on GitHub.\nReal-camera and target Edge-device validation are future work."),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Clip:
    def __init__(self, folder: Path):
        self.folder = folder
        self.video = folder / "annotated_video.mp4"
        if not self.video.exists():
            self.video = folder / "annotated_video.avi"
        self.cap = cv2.VideoCapture(str(self.video))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open {self.video}")
        if abs(self.cap.get(cv2.CAP_PROP_FPS) - FPS) > 0.01:
            raise ValueError("This edit requires a 25 fps capture; regenerate the documented corpus.")
        if tuple(int(self.cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)) != (640, 360):
            raise ValueError("This edit requires the documented 640 x 360 corpus.")
        with (folder / "frame_metrics.csv").open(newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.events = json.loads((folder / "events.json").read_text())
        self.metadata = json.loads((folder / "run_metadata.json").read_text())
        if len(self.rows) != 400 or int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) != 400:
            raise ValueError(f"Expected 400 frames: {folder}")
        for i, row in enumerate(self.rows):
            if int(row["frame_index"]) != i or abs(float(row["timestamp_s"]) - i / FPS) > 1e-6:
                raise ValueError(f"Frame/CSV timebase mismatch: {folder}")
        self.next_index = 0

    def frame(self, index: int) -> Image.Image:
        if index != self.next_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise ValueError(f"Missing frame {index}: {self.video}")
        self.next_index = index + 1
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


class Renderer:
    def __init__(self, root: Path, font_dir: Path):
        self.clips = {name: Clip(root / "file" / name) for name in
                      ("normal", "blur", "blocked_partial", "frozen", "blur_from_start")}
        self.clips["auto"] = Clip(root / "auto" / "blur_from_start")
        self.font_dir = font_dir
        self.fonts = {}
        self.text_bounds = []
        self.event = next(e for e in self.clips["blocked_partial"].events["events"] if e["condition"] == "blocked")
        self.activation = next(i for i, row in enumerate(self.clips["blocked_partial"].rows) if row["active_blocked"] == "1")
        if self.clips["normal"].events["events"]:
            raise ValueError("The healthy shot has events; review the capture before publishing.")
        for clip, condition in (("blur", "blur"), ("blocked_partial", "blocked"), ("frozen", "frozen"), ("blur_from_start", "blur")):
            if not any(r[f"active_{condition}"] == "1" for r in self.clips[clip].rows):
                raise ValueError(f"Expected event missing: {clip}")
        if any(r["active_blur"] == "1" for r in self.clips["auto"].rows):
            raise ValueError("The startup comparison changed; revise the edit to match actual behavior.")

    def font(self, size=22, style="regular"):
        key = size, style
        if key not in self.fonts:
            name = {"regular": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf", "mono": "DejaVuSansMono.ttf"}[style]
            self.fonts[key] = ImageFont.truetype(str(self.font_dir / name), size)
        return self.fonts[key]

    def text(self, xy, content, size=22, color=WHITE, style="regular", max_width=None, spacing=8):
        font = self.font(size, style)
        if max_width is not None:
            lines = []
            for paragraph in content.split("\n"):
                line = ""
                for word in paragraph.split():
                    candidate = (line + " " + word).strip()
                    if line and self.d.textlength(candidate, font=font) > max_width:
                        lines.append(line)
                        line = word
                    else:
                        line = candidate
                lines.append(line)
            content = "\n".join(lines)
        box = self.d.multiline_textbbox(xy, content, font=font, spacing=spacing)
        if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
            raise ValueError(f"Text outside frame: {content!r} {box}")
        self.text_bounds.append(box)
        self.d.multiline_text(xy, content, font=font, fill=color, spacing=spacing)
        return box

    def base(self, index, eyebrow, title, footer="Actual pipeline output  |  1x playback"):
        self.im = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.im)
        self.text_bounds = []
        self.text((40, 23), "ROAD CAMERA HEALTH MONITOR", 16, CYAN, "bold")
        self.text((40, 58), title, 30, WHITE, "bold")
        self.text((40, 104), eyebrow, 17, MUTED)
        self.d.line((40, 648, 1240, 648), fill=LINE, width=1)
        self.text((40, 669), "SYNTHETIC SCENE + INJECTED FAULTS", 16, CYAN, "bold")
        self.text((676, 669), footer, 16, MUTED)
        self.d.rectangle((0, 713, W, 719), fill=LINE)
        self.d.rectangle((0, 713, round(W * (index + 1) / N), 719), fill=CYAN)

    def status(self, x, y, value, width=280):
        color = COLORS[value]
        self.d.rounded_rectangle((x, y, x + width, y + 54), radius=9, fill=tuple(int(c * .13) for c in color), outline=color, width=1)
        self.text((x + 17, y + 10), value, 27, color, "bold")

    def single(self, index, shot):
        start, end, name, source_start, eyebrow, title = shot
        k = source_start + index - start
        clip = self.clips[name]
        row = clip.rows[k]
        self.base(index, eyebrow, title)
        self.im.paste(clip.frame(k).resize((832, 468), Image.Resampling.LANCZOS), (40, 144))
        self.d.rounded_rectangle((896, 144, 1240, 612), radius=14, fill=PANEL, outline=LINE)
        self.text((920, 167), "STATE FROM FRAME CSV", 16, MUTED, "bold")
        self.status(920, 201, row["status"], 296)
        if name == "normal":
            self.text((920, 284), "Per-camera\ncalibration", 26, WHITE, "bold")
            self.text((920, 382), "Persistent fault events\nAnnotated video\nCSV + JSON reports", 21, MUTED, spacing=15)
            self.text((920, 535), "Python / OpenCV\nNumPy", 20, CYAN)
            self.text((40, 623), "A healthy reference makes later measurements relative to this camera.", 18, MUTED)
        else:
            condition = {"blur": "blur", "blocked_partial": "blocked", "frozen": "frozen"}[name]
            raw = row[f"flag_{condition}"] == "1"
            active = row[f"active_{condition}"] == "1"
            self.text((920, 285), condition.upper(), 22, WHITE, "bold")
            self.text((920, 331), "Raw rule flag", 19, MUTED)
            self.text((920, 361), str(raw).lower(), 27, CYAN if raw else WHITE, "mono")
            self.text((920, 414), "Persistent fault", 19, MUTED)
            self.text((920, 444), str(active).lower(), 27, COLORS[row["status"]], "mono")
            onset = "Fault begins at 6.00 s" if k < 150 else "Injected fault is present"
            self.text((920, 519), onset, 19, MUTED, max_width=288)
            self.text((920, 555), f"Source time {k/FPS:05.2f} s", 20, WHITE, "mono")
            note = {
                "blur": "A raw threshold crossing must persist before the alarm changes state.",
                "blocked_partial": "The original annotation shades tiles with collapsed texture after activation.",
                "frozen": "Watch the source timestamp advance while the image content is held.",
            }[name]
            self.text((40, 623), note, 18, MUTED)

    def comparison(self, index):
        k = index - 625
        self.base(index, "05 / CALIBRATION MATTERS", "Startup blur: the reference changes the decision")
        for x, name, title in [(40, "blur_from_start", "KNOWN-HEALTHY FILE BASELINE"), (656, "auto", "AUTO BASELINE")]:
            self.text((x, 158), title, 20, CYAN, "bold")
            self.im.paste(self.clips[name].frame(k).resize((584, 329), Image.Resampling.LANCZOS), (x, 199))
            self.status(x, 543, self.clips[name].rows[k]["status"], 268)
        self.text((40, 612), "Same source time in both panels; existing annotations are preserved.", 18, MUTED)
        self.text((943, 549), "First 4 s used to learn", 17, MUTED)
        self.text((943, 575), "baseline before inspection", 17, MUTED)

    def evidence(self, index):
        self.base(index, "06 / INSPECTABLE OUTPUTS", "Events and measurements remain reviewable", "Actual JSON + CSV excerpts")
        self.d.rounded_rectangle((40, 153, 624, 599), radius=12, fill=PANEL, outline=LINE)
        self.d.rounded_rectangle((648, 153, 1240, 599), radius=12, fill=PANEL, outline=LINE)
        self.text((64, 178), "events.json", 24, CYAN, "bold")
        self.text((64, 218), "Selected fields / partial-obstruction run", 17, MUTED)
        keys = ("condition", "severity", "start_time_s", "end_time_s", "detection_latency_frames", "ongoing")
        block = json.dumps({k: self.event[k] for k in keys}, indent=2)
        self.text((64, 265), block, 19, WHITE, "mono", spacing=13)
        self.text((672, 178), "frame_metrics.csv", 24, CYAN, "bold")
        self.text((672, 218), "Selected columns / around event activation", 17, MUTED)
        fields = ("frame_index", "timestamp_s", "status", "active_blocked")
        self.text((672, 274), ",".join(fields), 16, CYAN, "mono")
        for j, k in enumerate(range(self.activation - 2, self.activation + 3)):
            row = self.clips["blocked_partial"].rows[k]
            if k == self.activation:
                self.d.rectangle((666, 312 + j*43, 1222, 347 + j*43), fill=(43, 51, 56))
            self.text((672, 316 + j*43), ",".join(row[f] for f in fields), 19, WHITE, "mono")
        self.text((672, 550), "Values copied directly from the capture.", 18, MUTED)
        self.text((40, 618), "Event start marks raw evidence; the CSV shows when the persistent state activates.", 18, MUTED)

    def results(self, index):
        self.base(index, "07 / PREVIOUSLY REPORTED EVALUATION", "Published results, with the scope beside each number", "Source: docs/EVALUATION.md")
        cards = [
            (40, "FULL PIPELINE", "57.6 fps", "2.31x realtime", ["640x360 / 25 fps input", "Includes annotated-video output", "OpenCV 4.13 / reported container"]),
            (448, "FAULT DETECTION", "100%", "steady-state recall", ["8 fault types / healthy file baseline", "Synthetic development corpus", "After ramp + temporal hold-down"]),
            (856, "CLEAN CLIP", "0", "false-alarm events", ["One clean synthetic clip", "400 frames / 16 seconds", "Not a field false-alarm rate"]),
        ]
        for x, label, number, sub, lines in cards:
            self.d.rounded_rectangle((x, 160, x+384, 519), radius=12, fill=PANEL, outline=LINE)
            self.text((x+22, 185), label, 17, CYAN, "bold")
            self.text((x+22, 238), number, 48, WHITE, "bold")
            self.text((x+22, 311), sub, 23, WHITE)
            for j, line in enumerate(lines):
                self.text((x+22, 377+j*37), line, 17, MUTED)
        self.text((40, 552), "11 clips from one generated scene; thresholds were selected on this corpus.", 21, WHITE)
        self.text((40, 594), "These historical numbers are separate from the current demo capture, not a new benchmark.", 19, MUTED)

    def outro(self, index):
        self.base(index, "08 / CODE + EVALUATION + REPRODUCTION", "Explore the engineering behind the demo", "Future work is not a measured result")
        self.text((40, 188), "Camera reliability\nfor perception pipelines", 48, WHITE, "bold", spacing=15)
        self.text((43, 347), "Calibration  /  Temporal persistence  /  Structured outputs", 24, CYAN)
        self.d.rounded_rectangle((40, 427, 1240, 504), radius=12, fill=PANEL, outline=LINE)
        self.text((63, 450), "github.com/Tamer1020/road-camera-health-monitor", 30, WHITE, "mono")
        self.text((40, 542), "NEXT VALIDATION", 17, CYAN, "bold")
        self.text((40, 578), "Held-out real footage  /  Target Edge hardware  /  Live RTSP", 26, WHITE)

    def render(self, index):
        for shot in SHOTS:
            if shot[0] <= index < shot[1]:
                self.single(index, shot)
                return self.im
        if index < 875:
            self.comparison(index)
        elif index < 1075:
            self.evidence(index)
        elif index < 1350:
            self.results(index)
        else:
            self.outro(index)
        return self.im


def srt_time(t):
    ms = round(t * 1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", type=Path, default=Path("outputs/demo_capture"))
    ap.add_argument("--output", type=Path, default=Path("docs/demo/road-camera-health-demo.mp4"))
    ap.add_argument("--font-dir", type=Path, default=Path("/usr/share/fonts/truetype/dejavu"))
    ap.add_argument("--source-revision", required=True, help="Git commit used for the source capture")
    ap.add_argument("--preview-only", action="store_true")
    args = ap.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("FFmpeg is required on PATH")
    if not args.output.suffix.lower() == ".mp4":
        raise SystemExit("Use an .mp4 output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(args.captures, args.font_dir)
    preview_dir = Path("outputs/demo_render")
    preview_dir.mkdir(parents=True, exist_ok=True)
    for i in (60, 170, 230, 425, 560, 790, 965, 1175, 1400):
        renderer.render(i).save(preview_dir / f"frame-{i:04}.png")
    if args.preview_only:
        print(f"Previews: {preview_dir}")
        return
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
           "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(N):
            encoder.stdin.write(renderer.render(i).tobytes())
            if i % 250 == 0:
                print(f"Rendered {i}/{N} frames", flush=True)
    except Exception:
        encoder.stdin.close()
        encoder.wait()
        raise
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("FFmpeg encoding failed")
    subtitle = args.output.with_suffix(".srt")
    subtitle.write_text("\n\n".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{t}" for i, (a,b,t) in enumerate(CAPTIONS, 1)) + "\n")
    renderer.render(790).save(args.output.parent / "poster.jpg", quality=93)
    capture_info = {}
    for key, clip in renderer.clips.items():
        capture_info[key] = {
            "folder": str(clip.folder),
            "annotated_video_sha256": sha256(clip.video),
            "frame_metrics_sha256": sha256(clip.folder / "frame_metrics.csv"),
            "events_sha256": sha256(clip.folder / "events.json"),
            "run_started_utc": clip.metadata["run_started_utc"],
            "baseline_source": clip.metadata["baseline"]["source"],
            "tool": clip.metadata["tool"],
            "state_transitions": [{"frame_index": int(r["frame_index"]), "timestamp_s": float(r["timestamp_s"]), "status": r["status"]}
                                  for j, r in enumerate(clip.rows) if j == 0 or r["status"] != clip.rows[j-1]["status"]],
        }
    manifest = {
        "purpose": "Demonstration capture, not a new benchmark or real-world validation",
        "source_revision": args.source_revision,
        "config_path": "configs/default.yaml",
        "config_sha256": sha256(Path("configs/default.yaml")),
        "source_scene_seed": 7,
        "corruption_seed": 11,
        "source_dimensions": [640, 360],
        "source_fps": FPS,
        "frames_per_source_clip": 400,
        "output_dimensions": [W, H],
        "output_fps": FPS,
        "output_frames": N,
        "output_duration_s": N/FPS,
        "audio": "none; captions and on-screen text",
        "playback": "1x within every moving excerpt; explicit chapter cuts between cases",
        "auto_baseline": "Offline pre-pass learns from first 100 frames (4 s), then inspection starts at frame 0",
        "historical_result_source": "docs/EVALUATION.md at source_revision; not runtime measurements of this capture",
        "shots": [{"demo_start_frame": a, "demo_end_frame_exclusive": b, "clip": c, "source_start_frame": d, "source_end_frame_exclusive": d+b-a} for a,b,c,d,_,_ in SHOTS]
                 + [{"demo_start_frame": 625, "demo_end_frame_exclusive": 875, "clips": ["blur_from_start", "auto"], "source_start_frame": 0, "source_end_frame_exclusive": 250}],
        "captures": capture_info,
        "output_sha256": sha256(args.output),
    }
    (args.output.parent / "capture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    evidence = args.output.parent / "evidence"
    evidence.mkdir(exist_ok=True)
    shutil.copyfile(args.captures / "file/blocked_partial/events.json", evidence / "blocked-partial-events.json")
    fields = ("frame_index", "timestamp_s", "status", "active_blocked")
    with (evidence / "blocked-partial-transition.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in renderer.clips["blocked_partial"].rows[renderer.activation-2:renderer.activation+3]:
            writer.writerow({k: row[k] for k in fields})
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes), captions, poster, manifest and evidence")
    for clip in renderer.clips.values():
        clip.cap.release()


if __name__ == "__main__":
    main()
