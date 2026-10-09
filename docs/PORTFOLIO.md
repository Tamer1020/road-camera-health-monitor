# Portfolio presentation

[Project overview](../README.md) · [Demo guide](DEMO.md)

Copy, presentation choices, and publishing status for the GitHub repository and
a short portfolio entry. The demo recording remains a separate deliverable.

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
2. The existing healthy/blur/blocked illustration with its injection disclosure.
3. A demo link once a real recording is published.
4. A compact results table with metric definitions and benchmark scope.
5. Calibration and persistence decisions, followed by quick-start commands.
6. Boundaries and the next validation steps, with detailed reference links.

The repository README follows this structure, with the recording guide in place
of an unpublished-video link. The About description, topics, social preview, and
profile pin have also been applied through GitHub settings.

## GitHub landing-page actions

| Surface | Change |
|---|---|
| About description | Uses the description above |
| Topics | Uses the implementation topics above |
| Website field | Link the finished demo or a project case study once available; otherwise leave it empty |
| Profile pins | Pinned among the first two projects on the profile |
| Social preview | Uses the project title, perception subtitle, and an abstract camera graphic |
| Sidebar | Empty Releases and Packages sections are hidden |
| README top links | Link to the published demo when it exists; retain direct links to evaluation and quick start |
| Evidence access | Link generated evaluation reports and benchmark metadata from a future tagged release after reproducing them |

The [social preview](social-preview.jpg) uses a 2:1 canvas and wide safe margins
so its title remains readable without cropping. It is an AI-generated branding
graphic made with the built-in imagegen tool, not a camera frame or evaluation
artifact; the [generation prompt](social-preview.prompt.txt) is included.
It contains no performance metrics or deployment claims. The existing
[README illustration](road_camera_health_monitor_hero_final.jpg) remains separate,
with its real-frame/injected-fault disclosure.

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
- [ ] Record the [demo](DEMO.md) from actual output videos.
- [ ] Add its working URL to the README and, if suitable, the Website field.
- [x] Pin the repository among the first two projects on the profile.
- [x] Hide the empty Releases and Packages sidebar sections.
- [ ] For a later evidence release, record the commit, config, dependency versions,
      host details, and generated JSON/Markdown results together.

Completed settings were verified on GitHub on 2026-10-09. Recording and publishing
the demo, and producing a versioned evidence release, remain open.
