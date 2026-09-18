# Evaluation

Everything in this document is reproducible with:

```bash
python scripts/make_sample_data.py -o sample_data/road_clean.mp4 -n 400
python scripts/make_eval_corpus.py
road-health calibrate -i sample_data/eval/normal.mp4 -o sample_data/eval/baseline.json
python scripts/run_evaluation.py --baseline-mode file
python scripts/run_evaluation.py --baseline-mode auto
python scripts/benchmark.py
```

Raw output: `outputs/evaluation/{file,auto}/summary.json` and `report.md`.

---

## 1. Why the corpus is synthetic

Two reasons, neither of them convenience:

1. **Licensing.** Public road and traffic footage is almost never
   redistributable. Committing it to a public repository is a legal problem,
   not a technical one.
2. **Ground truth and controllability.** Scoring precision and recall requires
   a label on *every frame*. With scraped footage that means hand-annotation,
   and even then the onset of a gradual defocus is a judgement call. A
   generated scene also gives a controllable sensor noise floor, which the
   freeze detector is explicitly calibrated against — real footage would not
   let us set it.

The generated scene (`scripts/make_sample_data.py`) models a flat road under
linear perspective, perspective-correct lane-dash spacing, traffic with size
and speed scaling by depth, static roadside structures that give the reference
view stable edges, per-frame Gaussian sensor noise, and a slow auto-exposure
wobble.

**What this costs us** is stated below and in the README: these results do not
establish field accuracy.

## 2. Corpus

11 clips, each 400 frames at 25 fps, 640×360. Every clip is the *same* clean
source with one controlled fault injected over a known window, so the ground
truth is exact.

| Clip | Condition | Fault window | Injection |
|---|---|---|---|
| `normal` | — | none | unmodified |
| `blur` | blur | 150–330 | Gaussian σ ramped 0.4 → 4.6 over 25 frames |
| `dark` | dark | 150–330 | gain → 0.20 with a −14 offset |
| `bright` | bright | 150–330 | gain → 2.6 with a +55 offset |
| `low_contrast` | low_contrast | 150–330 | airlight blend, transmission → 0.28 |
| `blocked_partial` | blocked | 150–330 | opaque polygon over ~43% of the frame |
| `blocked_full` | blocked | 150–330 | near-uniform field at level 58 |
| `frozen` | frozen | 150–330 | frame 149 repeated byte-for-byte |
| `moved` | moved | 200–399 | +74 px, −38 px translation and 4.5° rotation |
| `shake` | shake | 150–330 | integer-pixel oscillation, ±4.5 px, plus jitter |
| `blur_from_start` | blur | 0–399 | permanent defocus — camera already faulted |

Faults ramp in over 12–25 frames rather than appearing instantaneously,
because real faults do.

## 3. Scoring

A single precision/recall pair conflates different effects, so the evaluation
reports them separately.

**Steady-state recall** is computed only after the injected fault ramp and the
configured temporal hold-down have elapsed. Before that point the fault may
still be sub-threshold and the de-bouncer is deliberately designed not to raise
an event.

**Detection latency** is the number of frames between the first genuinely bad
frame and the alarm. It is compared against the configured hold-down, which is
a floor on latency by construction.

**False positives** exclude the designed recovery tail, because the
de-bouncer intentionally takes `exit_seconds` to clear.

Whole-window precision/recall are still reported for completeness, but they
are structurally lower because ramp-in and recovery-tail frames are included.

All scoring uses the **post-temporal** state (`active_*` columns), not raw
per-frame flags. What an operator sees is the de-bounced state.

## 4. Detection results — file baseline

| Clip | Condition | Precision | Whole-window recall | Steady recall | Detect latency | Hold-down | FP frames | Events |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| blur | blur | 0.653 | 0.718 | **1.000** | 51 fr (2.04 s) | 50 | 0 | 1 |
| dark | dark | 0.629 | 0.646 | **1.000** | 64 fr (2.56 s) | 50 | 0 | 1 |
| bright | bright | 0.621 | 0.624 | **1.000** | 68 fr (2.72 s) | 50 | 0 | 1 |
| low_contrast | low_contrast | 0.566 | 0.497 | **1.000** | 91 fr (3.64 s) | 75 | 0 | 1 |
| blocked_partial | blocked | 0.582 | 0.530 | **1.000** | 85 fr (3.40 s) | 75 | 0 | 1 |
| blocked_full | blocked | 0.582 | 0.530 | **1.000** | 85 fr (3.40 s) | 75 | 0 | 1 |
| frozen | frozen | 0.843 | 0.713 | **1.000** | 52 fr (2.08 s) | 50 | 0 | 1 |
| moved | moved | 1.000 | 0.615 | **1.000** | 77 fr (3.08 s) | 75 | 0 | 1 |
| shake | shake | 0.594 | 0.558 | **1.000** | 80 fr (3.20 s) | 75 | 0 | 1 |
| blur_from_start | blur | — | — | **1.000** | 49 fr (1.96 s) | 50 | 0 | 1 |

Every condition is detected exactly once and there are zero false-positive
frames outside fault and recovery windows.

### Clean clip

**0 false-alarm events.** Status is `OK` on 100% of 400 frames. Per-condition
active-frame fraction is 0.0000 for all eight conditions.

## 5. Cross-condition confusion

Fraction of each clip's fault-window frames during which each condition was
active:

| clip | blur | dark | bright | low_contrast | blocked | frozen | moved | shake |
|---|---|---|---|---|---|---|---|---|
| normal | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| blur | **0.72** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dark | 0.00 | **0.65** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| bright | 0.00 | 0.00 | **0.62** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| low_contrast | 0.00 | 0.00 | 0.00 | **0.50** | 0.00 | 0.00 | 0.00 | 0.00 |
| blocked_partial | 0.00 | 0.00 | 0.00 | 0.00 | **0.53** | 0.00 | 0.00 | 0.00 |
| blocked_full | 0.00 | 0.00 | 0.00 | 0.00 | **0.53** | 0.00 | 0.00 | 0.00 |
| frozen | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.71** | 0.00 | 0.00 |
| moved | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.61** | 0.00 |
| blur_from_start | **0.88** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shake | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.56** |

The matrix is strictly diagonal. Diagonal values are below 1.0 because they
include ramp-in and temporal hold-down frames.

### Failure-driven fixes

The first implementation produced cross-condition false positives. The fixes
were based on measured failure modes:

| Symptom | Root cause | Fix |
|---|---|---|
| `blocked` fired on dark/bright/fog | Global image flattening looked like local occlusion | Normalize tile texture collapse by frame-wide contrast change and keep a separate uniform-field path |
| `frozen` fired on dark clips | Crushed signals have tiny inter-frame differences | Compare frame MAD with the frame's measured noise floor and return **unavailable** below a minimum noise level |
| `moved` fired on blur/bright/blocked | Edge correlation collapses whenever structure disappears | Add framing reliability gates; return **unavailable** when edge content/exposure/contrast are unreliable |
| `blur` fired on dark frames | Sharpness is not meaningful on a railed frame | Gate blur on exposure reliability |

General rule: a detector that infers a fault from missing structure must stand
down when another fault already destroyed that structure.

## 6. Why per-camera calibration matters

`blur_from_start` is the decisive test. The camera is already defocused when
monitoring begins.

| Baseline mode | Source of normal | Detected? | Steady recall | Events |
|---|---|---|---:|---:|
| `file` | commissioning-time healthy clip | yes, at 49 frames | **1.000** | 1 |
| `auto` | first 4 s of clip under test | **never** | **0.000** | 0 |

On the other ten clips both modes are identical because every clip begins
healthy — exactly the assumption auto-baselining makes.

Operational conclusion: `auto` is useful for triage when the start of the
clip is believed to be healthy. Production deployment should use
`calibrate` on a known-healthy commissioning clip.

## 7. Threshold sensitivity

The evaluation also sweeps raw-rule thresholds offline from stored metrics.

### blur — `rel_min`

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.15 | 1.000 | 0.993 | 0.997 |
| 0.25 | 1.000 | 0.997 | 0.998 |
| **0.35** | **1.000** | **0.997** | **0.998** |
| 0.50 | 0.998 | 0.998 | 0.998 |
| 0.65 | 0.962 | 0.998 | 0.980 |
| 0.80 | 0.709 | 0.998 | 0.829 |

### dark — `rel_mean_luma`

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.25 | 1.000 | 0.890 | 0.942 |
| 0.35 | 1.000 | 0.901 | 0.948 |
| **0.45** | **1.000** | **0.917** | **0.957** |
| 0.55 | 0.496 | 0.923 | 0.645 |
| 0.70 | 0.491 | 0.923 | 0.641 |

### bright — `clip_high_frac`

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.05 | 1.000 | 0.895 | 0.945 |
| 0.10 | 1.000 | 0.895 | 0.945 |
| **0.20** | **1.000** | **0.895** | **0.945** |
| 0.35 | 1.000 | 0.895 | 0.945 |
| 0.50 | 1.000 | 0.895 | 0.945 |

The bright sweep is flat on this corpus because the secondary
`high_level AND lost_range` clause carries the synthetic bright clip. Genuine
sun-glare validation is still needed on real footage.

### low_contrast — `rel_span`

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.30 | 0.987 | 0.862 | 0.920 |
| 0.40 | 0.982 | 0.884 | 0.930 |
| **0.50** | **0.970** | **0.906** | **0.937** |
| 0.65 | 0.960 | 0.934 | 0.947 |
| 0.80 | 0.926 | 0.961 | 0.943 |

### blocked — `dead_tile_frac`

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.20 | 0.983 | 0.942 | 0.962 |
| 0.30 | 0.983 | 0.939 | 0.960 |
| **0.35** | **1.000** | **0.939** | **0.969** |
| 0.45 | 1.000 | 0.470 | 0.639 |
| 0.60 | 1.000 | 0.470 | 0.639 |
| 0.80 | 1.000 | 0.467 | 0.637 |

The cliff at 0.45 is meaningful: the partial occlusion covers about 43% of the
frame, so a threshold above that misses it.

### frozen — `noise_ratio`

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.10 | 1.000 | 0.994 | 0.997 |
| **0.20** | **1.000** | **0.994** | **0.997** |
| 0.35 | 0.861 | 0.994 | 0.923 |
| 0.50 | 0.571 | 1.000 | 0.727 |
| 0.70 | 0.125 | 1.000 | 0.223 |

### moved — `shift_frac`

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.01 | 0.957 | 0.995 | 0.975 |
| 0.02 | 1.000 | 0.990 | 0.995 |
| **0.04** | **1.000** | **0.985** | **0.992** |
| 0.07 | 1.000 | 0.985 | 0.992 |
| 0.12 | 1.000 | 0.985 | 0.992 |

### shake — `jitter_frac`

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.0004 | 0.541 | 0.983 | 0.698 |
| 0.0008 | 0.540 | 0.967 | 0.693 |
| **0.0012** | **0.572** | **0.967** | **0.719** |
| 0.0020 | 0.508 | 0.702 | 0.589 |
| 0.0040 | 0.000 | 0.000 | 0.000 |

`shake` is the weakest raw rule and should be treated as advisory. Its
post-temporal behavior is much cleaner because of the 3-second hold-down.

The thresholds are still development-set choices; they have not been validated
on held-out real data.

## 8. What this evaluation proves — and what it does not

### Establishes

- Every implemented rule fires on its target synthetic phenomenon.
- Steady-state recall is 1.0 on the controlled corpus.
- Cross-condition activity is diagonal after reliability gating.
- The clean clip produces zero false-alarm events.
- Detection latency is dominated by the configured temporal hold-down.
- Per-camera calibration provides information that auto-baselining cannot
  recover once a fault is already present.
- Reliability gates can turn an unreliable check into **unavailable** instead
  of silently reporting healthy.

### Does not establish

- Field false-alarm rate over days or weeks of real operation.
- Behavior across day/night transitions, rain, glare, wet-road reflection,
  dust, seasonal change, and real optics.
- Robustness across production encoders and bitrates.
- Slow defocus drift over hours.
- Textured occlusion such as spider webs and branches.
- Multi-camera fleet behavior or live RTSP operation.

Injected faults are idealized. Treat these numbers as controlled validation,
not production accuracy.

## 9. Repeating the evaluation on real footage

1. Record 2–5 minutes per camera in a verified healthy state.
2. Keep recordings outside the repository; `.gitignore` excludes `data/`.
3. Calibrate each camera:
   `road-health calibrate -i data/cam01_healthy.mp4 -o baselines/cam01.json`
4. Run `inspect-video` on held-out clips containing known incidents.
5. Annotate true fault windows and reuse the scoring logic in
   `scripts/run_evaluation.py`.
6. Re-run threshold sweeps and verify the operating point lies on a plateau,
   not a narrow optimum.
7. For 24/7 deployment, maintain separate day/dusk/night baselines.

## 10. Benchmark

Single container core, 640×360 @ 25 fps, 400 frames, OpenCV 4.13.

| Configuration | Source fps | Analysis-only fps | × realtime |
|---|---:|---:|---:|
| Full pipeline + annotated video | 57.6 | 64.3 | 2.31 |
| Analysis only | 66.5 | 68.7 | 2.66 |
| Analysis only, stride 3 | 195.6 | 71.1 | 7.82 |
| Analysis only, `max_side: 320` | 87.5 | 91.2 | 3.50 |

Per-stage cost for the full pipeline, ms per analyzed frame:

| Stage | ms |
|---|---:|
| metrics | 15.375 |
| write_video | 0.953 |
| decode | 0.414 |
| annotate | 0.338 |
| preprocess | 0.113 |
| checks | 0.052 |
| temporal | 0.009 |

`pipeline_fps_end_to_end` counts source frames consumed per second.
`analysis_fps_excluding_io` counts analyzed frames excluding decode and video
writing. Both are reported because they answer different questions.

Metrics dominate the runtime. Tile statistics use summed-area tables and
registration runs at reduced scale; both were introduced after profiling
identified the original hot spots.
