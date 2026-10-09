# 60-second demo guide

[Project overview](../README.md) · [Evaluation](EVALUATION.md) · [Portfolio copy](PORTFOLIO.md)

This is a recording plan and a set of reproducible commands. A finished demo
video is not included in this documentation change.

## Goal

Show a working camera-health pipeline, the calibration decision behind it, and
the limits of the evidence in one minute. Record actual annotated outputs.
Keep **"Synthetic scene + injected faults"** visible during synthetic footage.

## Prepare the recordings

Complete the [README quick start](../README.md#quick-start) first. From the
repository root, generate annotated outputs for the existing evaluation cases:

```bash
python scripts/run_evaluation.py --baseline-mode file --video
python scripts/run_evaluation.py --baseline-mode auto --video
```

These use the existing corpus, baseline, and default configuration. Each run
writes its own report; no new published performance claim follows automatically
from recording it.

| Shot | Output to open |
|---|---|
| Healthy scene | `outputs/evaluation/file/normal/annotated_video.mp4` |
| Blur developing | `outputs/evaluation/file/blur/annotated_video.mp4` |
| Partial obstruction | `outputs/evaluation/file/blocked_partial/annotated_video.mp4` |
| Frozen frames | `outputs/evaluation/file/frozen/annotated_video.mp4` |
| Startup blur, commissioned baseline | `outputs/evaluation/file/blur_from_start/annotated_video.mp4` |
| Startup blur, auto baseline | `outputs/evaluation/auto/blur_from_start/annotated_video.mp4` |
| Event evidence | `outputs/evaluation/file/blocked_partial/events.json` |
| Metrics evidence | `outputs/evaluation/file/blocked_partial/frame_metrics.csv` |

Use the actual output filename if codec fallback produces another container.
Evaluation reports are under `outputs/evaluation/{file,auto}/`. Keep them beside
the recording and note the Git commit and configuration used.

## Storyboard

| Time | Visual | Message |
|---|---|---|
| 0–5 s | Healthy annotated clip with the project title | Camera reliability for perception pipelines; show the synthetic-source label immediately |
| 5–15 s | Actual blur and obstruction excerpts, showing the fault before the alarm | The monitor converts image-quality changes into persistent fault events |
| 15–23 s | Frozen-frame excerpt with visible frame/time overlay and event state | Temporal evidence distinguishes a stalled image from ordinary scene content under the detector's assumptions |
| 23–38 s | Startup-blur comparison, file and auto outputs side by side at the same source timestamps | A known-healthy baseline detects an existing fault; auto mode can learn it as normal |
| 38–45 s | Real generated event JSON and per-frame CSV, briefly highlighted | Outputs can be inspected by a human or consumed by another program |
| 45–54 s | Published-results card with scope printed beside each number | 57.6 source fps / 2.31× realtime, including video output; 100% steady-state recall on the controlled corpus; 0 events on one 16 s clean clip |
| 54–60 s | Repository URL and next-validation caption | Real-camera validation and target Edge-device measurements remain future work |

For the results card, include **"640×360 · 25 fps source · OpenCV 4.13 · reported
container benchmark"**. Label the detection results **"Synthetic development
corpus · file baseline"**. Keep the qualification text readable on a small screen.

## Narration

> I built a camera-health monitor for perception pipelines using OpenCV and NumPy.
> It checks recorded road video for image degradation and persistent faults,
> then exports annotated video and machine-readable reports.
> This demo uses a synthetic scene with injected faults.
> Each camera is compared with a known-healthy baseline, and temporal persistence
> delays alarms until evidence accumulates.
> Here, calibration catches blur present from the start; automatic baselining
> learns that fault as normal.
> The repository includes reproducible evaluation and a reported full-pipeline
> benchmark above the source frame rate.
> Real-camera validation, live RTSP, and target Edge-device benchmarking are next.

## Recording rules

- Show the actual `OK`, `DEGRADED`, `UNUSABLE`, and unavailable-check overlays.
  Do not redraw a state to make a detection appear earlier.
- Keep fault onset and the subsequent alarm visible. If an excerpt skips time,
  label the cut; if playback is accelerated, label its speed.
- Compare file and auto modes at matching source times. Auto mode has an initial
  calibration period; keep this visible instead of cropping it into an
  apparently immediate healthy decision.
- For a recovery shot, use only a recovery actually present in the generated
  clip. Some configured clear-down periods extend beyond its end.
- Treat the hero image separately: it contains a real frame with injected
  degradation for illustration. It is not the benchmark dataset.
- Do not imply that a detector, tracker, maintenance system, or RTSP service is
  integrated. The pipeline currently emits diagnostic outputs.
- Use either the existing published benchmark with its context or separately
  identified new measurements. Do not mix a new host's timings with the old table.

## Export and publication

Export a readable 16:9 MP4 with captions; keep the total near 60 seconds. A short
preview can show one real state transition from the generated output. Label it
as synthetic and preserve its playback timing.

After the final MP4 exists, add a working video link near the README hero and
replace the top "Demo guide" link with "Watch demo". Keep this guide linked in
the project guide. Use the same recording for GitHub and a LinkedIn post, with
the repository URL and synthetic-evaluation scope visible.
