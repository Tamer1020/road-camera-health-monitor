# Portfolio presentation

[Project overview](../README.md) · [Demo guide](DEMO.md)

Copy and setup guidance for the GitHub repository and a short portfolio entry.
This file does not change repository settings or indicate that a demo has been
recorded or published.

## Positioning

**Display title**

Road Camera Health Monitor — Camera Reliability for Perception

**GitHub About description**

Camera health monitoring for perception pipelines with OpenCV and NumPy: per-camera calibration, persistent fault events, reproducible evaluation, and CPU benchmarking.

Keep the existing repository name, `road-camera-health-monitor`, so application
links continue to identify the same project.

**Suggested topics**

`computer-vision`, `opencv`, `numpy`, `python`, `perception`,
`camera-health`, `image-quality`, `video-analysis`, `fault-detection`,
`traffic-monitoring`, `benchmarking`, `pytest`

These describe the present implementation. Edge-device deployment and learned
models belong in future-work text until implemented and measured.

## First-screen order

1. Title, one-sentence purpose, and the current CI badge.
2. The existing healthy/blur/blocked illustration with its injection disclosure.
3. A demo link once a real recording is published.
4. A compact results table with metric definitions and benchmark scope.
5. Calibration and persistence decisions, followed by quick-start commands.
6. Boundaries and the next validation steps, with detailed reference links.

The repository README now follows this structure, with the recording guide in
place of an unpublished-video link. The About description, topics, social preview,
and profile pin are GitHub settings to apply separately.

## GitHub landing-page actions

| Surface | Change |
|---|---|
| About description | Paste the description above |
| Topics | Apply the implementation topics above |
| Website field | Link the finished demo or a project case study once available; otherwise leave it empty |
| Profile pins | Pin this repository among the projects most relevant to Computer Vision and Perception applications |
| Social preview | Use the title plus the existing three-panel visual and a readable "Illustration: injected degradation" caption |
| README top links | Link to the published demo when it exists; retain direct links to evaluation and quick start |
| Evidence access | Link generated evaluation reports and benchmark metadata from a future tagged release after reproducing them |

For a social preview, keep the title and three camera states readable at thumbnail
size. Do not put a context-free "100% accuracy" or "Edge deployed" label on it.
The existing [hero image](road_camera_health_monitor_hero_final.jpg) can serve as
the visual source; its real-frame/injected-fault disclosure must remain visible.
No new social-preview image is included in this change.

## What each hiring audience can inspect

| Audience | Present engineering evidence | Boundary |
|---|---|---|
| Computer Vision | Sharpness, exposure, noise, tile statistics, registration, and calibration | Fault realism and thresholds have not been validated on held-out real footage |
| Perception | Camera-health signals, temporal events, unavailable checks, and structured outputs | No measured detector/tracker accuracy gain or automatic downstream control |
| Edge AI | CPU pipeline, per-stage profiling, stride/resolution options, and output-cost benchmarking | No reported deployment, power, memory, or sustained throughput on a named Edge device |

## Short portfolio description

Built an OpenCV/NumPy pipeline for camera-health monitoring in recorded road video,
with per-camera calibration, temporal fault persistence, and annotated/CSV/JSON
outputs. Documented a 57.6 source-fps full-pipeline container benchmark at 640×360
with video output, and 100% steady-state recall on a controlled synthetic
development corpus. The repository includes 99 documented tests and CI regression
gates; real-camera validation and target Edge-device measurements remain open.

## Claim-to-evidence guide

| Public wording | Evidence and qualification |
|---|---|
| "Camera health monitoring for perception pipelines" | [Implementation](../src/road_health); emits diagnostics rather than controlling downstream services |
| "57.6 source fps / 2.31× realtime" | [Published benchmark](EVALUATION.md#10-benchmark); full pipeline with annotated video, 640×360 at 25 fps input, OpenCV 4.13, reported container environment |
| "100% steady-state recall on the synthetic development corpus" | [Scoring](EVALUATION.md#3-scoring) and [per-clip results](EVALUATION.md#4-detection-results--file-baseline); excludes ramp and hold-down |
| "0 false-alarm events on the clean clip" | [Clean clip](EVALUATION.md#clean-clip); one synthetic clip, 400 frames / 16 s |
| "File baseline catches startup blur; auto misses it" | [Calibration comparison](EVALUATION.md#6-why-per-camera-calibration-matters); one controlled test case |
| "99 documented tests with CI" | [Tests](../tests) and [workflow](../.github/workflows/ci.yml); use the actual workflow status when describing a specific commit |

## Publishing checklist

- [ ] Apply the About description and topics.
- [ ] Set a social preview that preserves the illustration disclosure.
- [ ] Record the [demo](DEMO.md) from actual output videos.
- [ ] Add its working URL to the README and, if suitable, the Website field.
- [ ] Pin the repository on the profile.
- [ ] For a later evidence release, record the commit, config, dependency versions,
      host details, and generated JSON/Markdown results together.

The checklist is intentionally open: committing this document does not perform
these profile/settings changes or produce a video.
