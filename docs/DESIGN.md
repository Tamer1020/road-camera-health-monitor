# Design reference

[Project overview](../README.md) · [Evaluation](EVALUATION.md) · [Demo guide](DEMO.md)

This reference preserves the implementation detail behind the portfolio overview.
The current interface processes recorded video files, one camera per run.

## Baseline-relative detection

Almost every image-quality metric worth computing is scene dependent. A
Laplacian variance of 90 is excellent for a camera pointed at an empty desert
highway and terrible for one pointed at a tree line. An absolute threshold
therefore cannot be both sensitive and specific across a fleet — you either
miss real defocus on detailed scenes or permanently alarm on plain ones.

So the tool characterises each camera in its healthy state and detects
*deviation from that camera's own normal*:

```
road-health calibrate --input healthy_clip.mp4 --output baseline.json
```

A baseline stores, per camera:

- robust location and scale (median and MAD, not mean and std — the calibration
  clip contains traffic, shadows and the occasional parked truck) for nine
  metrics;
- per-tile texture statistics on a grid, so occlusion can be localised;
- a **temporal-median background image** of the commissioned field of view — a
  deterministic background model that needs no training and no per-pixel state
  at runtime, and which doubles as the geometric reference for "has this camera
  moved".

Thresholds are then expressed as fractions of that camera's own median.
Absolute thresholds survive only for quantities with the same physical meaning
on every sensor (pixel clipping) or as an explicitly-labelled fallback.

## Module boundaries

| Module | Responsibility |
|---|---|
| [config.py](../src/road_health/config.py) | YAML configuration, validation, and conversion from seconds to frames |
| [io.py](../src/road_health/io.py) | Input validation, decoding, and video-writer fallback |
| [metrics.py](../src/road_health/metrics.py) | Frame measurements, with no health decisions |
| [baseline.py](../src/road_health/baseline.py) | Robust calibration statistics, reference view, persistence, and compatibility |
| [health_checks.py](../src/road_health/health_checks.py) | Fault rules and reliability gates |
| [temporal.py](../src/road_health/temporal.py) | Per-condition persistence, status, and event tracking |
| [pipeline.py](../src/road_health/pipeline.py) | Orchestration and timing |
| [reporting.py](../src/road_health/reporting.py) | CSV, event JSON, metadata, and logs |
| [visualization.py](../src/road_health/visualization.py) | Status overlay, active conditions, and unavailable checks |
| [cli.py](../src/road_health/cli.py) | Calibration and inspection commands |

**Measurement and decision are separate.** Keeping rules separate from metrics
supports calibration and offline threshold sweeps.

**Unavailable is explicit.** For example, missing edge structure in an
overexposed frame makes a framing check unreliable. The check stands down and
reports unavailable rather than claiming the camera has not moved. The overall
`OK` status only means no fault is active; consumers must also inspect check
availability. No downstream analytics service is controlled by this repository.

## Fault rules

| Condition | Severity | Primary metric | Core idea |
|---|---|---|---|
| `blur` | DEGRADED | `reblur_ratio` | Re-blur the frame with a known kernel; a sharp frame has high-frequency energy left to destroy, a defocused one does not. Largely cancels scene dependence. |
| `dark` | DEGRADED | `mean_luma`, `clip_low_frac` | Low light is **not** a fault. Fires only when information is actually lost: blacks crushed, or level far below normal *and* usable range collapsed with it. |
| `bright` | DEGRADED | `clip_high_frac` | Saturated pixels carry no information whatever the cause. Sun glare is folded in here on purpose rather than given a separate rule. |
| `low_contrast` | DEGRADED | `contrast_span` (p95−p05) | Fog/haze/filmed-over dome. Gated on normal exposure: haze compresses the histogram toward the middle, exposure faults compress it against a rail. |
| `blocked` | UNUSABLE | `dead_tile_frac` | A tile is occluded when its texture collapsed **more than the frame as a whole did**. That normalisation is what separates a covered lens from a dark or foggy scene. |
| `frozen` | UNUSABLE | `frame_mad` vs noise floor | A live imager always produces temporal noise (MAD ≈ 1.128 σ); a stalled decoder produces none. Self-calibrating per camera and per gain setting. |
| `moved` | UNUSABLE | phase correlation on edge maps | Displacement and structural correlation against the commissioned reference view. |
| `shake` | DEGRADED | std-dev of displacement | Unstable mount. Same measurement as `moved`, different temporal signature: a moved camera settles at a new offset, a shaking one oscillates around the old one. |

Supporting features: ROI masking for burnt-in OSD banners (sharp, static, and
otherwise capable of masking a frozen stream entirely); fixed analysis
resolution so thresholds are portable between a 1080p and a 4CIF camera;
frame-stride for throughput; typed CLI exit codes; codec fallback when `mp4v`
is unavailable.

## Metrics and calibration caveats

Every rule's docstring in `src/road_health/health_checks.py` states what it
measures, how thresholds should be calibrated, and its specific false
positive / false negative modes. Summarised:

| Metric | Known limitation |
|---|---|
| `lap_var` | Strongly scene dependent — never compared across cameras, only to the same camera's baseline. |
| `reblur_ratio` | Degenerate on a featureless frame: 0/0 evaluates to 0, i.e. "maximally defocused". Handled upstream — a uniform frame is classified `blocked`, which suppresses `blur`. Asserted in `test_reblur_ratio_is_degenerate_on_a_featureless_frame`. |
| `edge_density` | Fixed Canny hysteresis is acceptable only because the value is compared to the same camera's baseline, never to a constant. |
| `clip_*_frac` | The only genuinely camera-independent quantities; absolute thresholds on them are defensible. |
| `noise_sigma` | Lossy re-encoding suppresses sensor noise. Measured on the frame itself each time rather than assumed. |
| `frame_mad` | A very-low-bitrate encoder emitting skip-blocks on a static scene mimics a freeze. |
| `shift_frac` / `edge_corr` | Cannot distinguish an operator-commanded PTZ preset from a knock — the VMS has to supply PTZ state, or disable the check on PTZ cameras. |

## Temporal persistence

Per-condition leaky-bucket de-bounce with asymmetric fill and drain:

```
bad frame   level += 1                        (clamped at enter_frames)
good frame  level -= enter_frames/exit_frames
raise  when level >= enter_frames
clear  when level <= 0
```

Chosen over "N consecutive bad frames" because it tolerates intermittent good
frames inside a real fault (a branch swinging in and out of view) while still
requiring a majority of bad frames over the window, so it does not latch on
sparse noise. Raise and clear times are tuned independently.

**All hold-down times are configured in seconds, never frames**, and converted
using `source_fps / analysis.stride`. This preserves the intended hold-down duration across source rates and strides,
subject to frame rounding. Frame skipping can still change sampled evidence and
short-event detection; compare detection behavior when changing stride.

## CLI and exit codes

```text
road-health calibrate --input VIDEO --output BASELINE [--config CFG]
road-health inspect-video --input VIDEO --output DIR [--baseline FILE] [--config CFG]
```

A supplied `--baseline FILE` selects a saved commissioning baseline. Omitting
it uses the mode in the config (`auto` in [default.yaml](../configs/default.yaml)).
The string `auto` is not a special value of `--baseline`; that argument is a file
path. `--no-baseline` disables baseline use.

| Code | Meaning in the current implementation |
|---:|---|
| 0 | Completed without an active fault, or completed with `--no-status-exit` |
| 2 | Argument-parser error or missing input |
| 3 | Unsupported input |
| 4 | Corrupt or undecodable input |
| 5 | Configuration error |
| 6 | Baseline error |
| 10 | Completed with a `DEGRADED` state |
| 11 | Completed with an `UNUSABLE` state |
| 130 | Interrupted by the user |

Expected input/configuration errors come from
[errors.py](../src/road_health/errors.py); status exits and argument parsing are
defined in [cli.py](../src/road_health/cli.py). Unexpected exceptions are not
given a separate typed output-error code.

Additional options include `--stride`, `--no-video`, `--camera-id`, and
`--verbose`. Run `road-health inspect-video --help` for details.

## Failure cases

Known ambiguities, missed faults, false alarms, and expected behavior:

| Situation | Behaviour | Why |
|---|---|---|
| Frozen stream on an already-degraded camera | `frozen` reports **unavailable** | Below the minimum noise floor a stalled stream and a crushed-but-live one are genuinely indistinguishable from pixels alone. Deliberately declines to guess. |
| Lens covered by a *textured* obstruction (leafy branch, spider web) | `blocked` misses it | The rule detects loss of texture; a branch adds edges. |
| Lens covered by opaque black tape | `blocked` fires, `dark` co-fires | Correct status (UNUSABLE), attribution ambiguous. Both are listed in `co_active`. |
| Very-low-bitrate encoder, static night scene | `frozen` may false-fire | The encoder emits skip-blocks; decoded frames really are identical. |
| Operator-commanded PTZ preset change | `moved` fires | No access to VMS state. |
| Heavy snow covering the whole scene | `blocked` may fire | Texture genuinely disappears. |
| Large white box truck parked in front of the camera | `blocked` fires | Arguably a true positive for a monitoring camera. |
| Burnt-in OSD banner, stream stalls | `frozen` misses it | The banner clock keeps changing. Use `analysis.roi`. |
| Camera legitimately at night with a working IR illuminator | No alarm | By design — low light is not a fault unless information is lost. |

## Regression coverage

The documented suite contains 99 tests spanning configuration/default consistency,
known-answer image metrics, reliability gates, temporal behavior, event tracking,
file handling, end-to-end runs, and CLI exits. A throughput regression test
checks that end-to-end throughput counts source frames consumed.

[CI](../.github/workflows/ci.yml) runs the test matrix and generated-video CLI
smoke checks, verifies output artefacts and CSV row counts, and runs a separate
synthetic-evaluation gate. Its smoke commands are maintained in the workflow;
they are not extracted automatically from the README.

## Reading the benchmark

The published results are in [EVALUATION.md](EVALUATION.md#10-benchmark).

- `pipeline_fps_end_to_end` counts source frames consumed per second.
- `analysis_fps_excluding_io` counts analyzed frames excluding decode and video writing.
- `stride: 3` skips analysis of two out of every three source frames. Its
  195.6 source fps is not 195.6 fully analyzed frames per second.
- Profiling motivated summed-area tile statistics and reduced-scale registration.
  The per-stage table identifies metrics as the dominant reported cost.
- The benchmark driver is a sequential pipeline, but OpenCV may use internal
  threads. A reproducible single-core claim requires recorded CPU limits and
  thread settings. The published table does not document these controls.

## Validation priorities

Real-camera validation, multiple lighting regimes, PTZ integration, and textured
occlusions remain open. See the [README boundaries](../README.md#validation-boundaries)
and [real-footage protocol](EVALUATION.md#9-repeating-the-evaluation-on-real-footage).
