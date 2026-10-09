# Portfolio presentation

[Project overview](../README.md) · [Demo guide](DEMO.md)

Copy, presentation choices, and publishing status for the GitHub repository and
a short portfolio entry. The [finished 60-second demo](DEMO.md) is linked from
the README with its capture provenance and reproduction commands.

## Positioning

**Display title**

Road Camera Health Monitor — Camera Reliability for Perception

**GitHub About description**

Camera health monitoring for perception pipelines with OpenCV and NumPy: per-camera calibration, persistent fault events, reproducible evaluation, and CPU benchmarking.

Keep the existing repository name, `road-camera-health-monitor`, so application
links continue to identify the same project.

**Repository topics**

`computer-vision`, `opencv`, `numpy`, `python`, `perception`,
`camera-health`, `image-quality`, `video-analysis`, `fault-detection`,
`traffic-monitoring`, `benchmarking`, `pytest`

These describe the present implementation. Edge-device deployment and learned
models belong in future-work text until implemented and measured.

## First-screen order

1. Title, one-sentence purpose, and the current CI badge.
2. The embedded 60-second demo and its synthetic-source disclosure.
3. Direct video, MP4, and reproduction/evidence links.
4. A compact results table with metric definitions and benchmark scope.
5. Calibration and persistence decisions, followed by quick-start commands.
6. Boundaries and the next validation steps, with detailed reference links.

The repository README follows this structure. The About description, topics,
social preview, and profile pin have also been applied through GitHub settings.

## GitHub landing-page actions

| Surface | Change |
|---|---|
| About description | Uses the description above |
| Topics | Uses the implementation topics above |
| Website field | Left empty; the working demo is embedded directly in the README |
| Profile pins | Pinned among the first two projects on the profile |
| Social preview | Uses the project title, perception subtitle, and an abstract camera graphic |
| Sidebar | Empty Releases and Packages sections are hidden |
| README top links | Links to the published demo, evaluation, and quick start |
| Evidence access | Link generated evaluation reports and benchmark metadata from a future tagged release after reproducing them |

The [social preview](social-preview.jpg) uses a 2:1 canvas and wide safe margins
so its title remains readable without cropping. It is an AI-generated branding
graphic made with the built-in imagegen tool, not a camera frame or evaluation
artifact; the [generation prompt](social-preview.prompt.txt) is included.
It contains no performance metrics or deployment claims. The existing
[three-panel illustration](road_camera_health_monitor_hero_final.jpg) remains
separate from the demo, with its real-frame/injected-fault disclosure in the
[demo guide](DEMO.md#separate-illustration).

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

- [x] Apply the About description and topics.
- [x] Set a social preview with no unqualified result claims.
- [x] Record the [demo](DEMO.md) from actual output videos.
- [x] Add the finished video and its evidence/reproduction links to the README.
- [x] Pin the repository among the first two projects on the profile.
- [x] Hide the empty Releases and Packages sidebar sections.
- [ ] For a later evidence release, record the commit, config, dependency versions,
      host details, and generated JSON/Markdown results together.

Settings and the demo were published on 2026-10-09. A versioned full-evaluation
evidence release remains separate future work.
