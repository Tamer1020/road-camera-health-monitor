"""Configuration model.

Design rules:

* Every threshold lives here, never in code.
* Temporal thresholds are expressed in **seconds**, not frames, so that the
  same config behaves identically on a 10 fps and a 30 fps feed and is not
  silently broken by ``analysis.stride``.
* Detection thresholds are expressed **relative to a per-camera baseline**
  wherever the underlying quantity is scene-dependent (sharpness, edge
  density, texture).  Absolute thresholds are only used for quantities with
  a physical meaning that is the same on every camera (pixel clipping,
  absolute black level) or as an explicit fallback when no baseline exists.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from road_health.errors import ConfigError

SEVERITIES = ("OK", "DEGRADED", "UNUSABLE")


@dataclass
class AnalysisConfig:
    max_side: int = 640
    stride: int = 1
    roi: Optional[List[float]] = None
    tile_rows: int = 6
    tile_cols: int = 8
    fallback_fps: float = 25.0


@dataclass
class BaselineConfig:
    mode: str = "auto"
    path: Optional[str] = None
    warmup_seconds: float = 4.0
    min_warmup_frames: int = 25


@dataclass
class CheckConfig:
    enabled: bool = True
    severity: str = "DEGRADED"
    enter_seconds: float = 2.0
    exit_seconds: float = 4.0
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class OutputConfig:
    write_video: bool = True
    video_codec: str = "mp4v"
    annotate_stride: int = 1
    csv: bool = True
    events: bool = True
    metadata: bool = True


@dataclass
class Config:
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    checks: Dict[str, CheckConfig] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def enter_frames(self, name: str, fps: float) -> int:
        c = self.checks[name]
        eff = fps / max(1, self.analysis.stride)
        return max(1, int(round(c.enter_seconds * eff)))

    def exit_frames(self, name: str, fps: float) -> int:
        c = self.checks[name]
        eff = fps / max(1, self.analysis.stride)
        return max(1, int(round(c.exit_seconds * eff)))


DEFAULT_CHECKS: Dict[str, Dict[str, Any]] = {
    "blur": {
        "severity": "DEGRADED",
        "enter_seconds": 2.0,
        "exit_seconds": 4.0,
        "params": {
            "rel_min": 0.35,
            "abs_min": 6.0,
            "exposure_lo": 0.55,
            "exposure_hi": 1.60,
        },
    },
    "dark": {
        "severity": "DEGRADED",
        "enter_seconds": 2.0,
        "exit_seconds": 4.0,
        "params": {
            "abs_mean_luma": 35.0,
            "rel_mean_luma": 0.45,
            "clip_low_frac": 0.35,
        },
    },
    "bright": {
        "severity": "DEGRADED",
        "enter_seconds": 2.0,
        "exit_seconds": 4.0,
        "params": {
            "abs_mean_luma": 205.0,
            "clip_high_frac": 0.20,
        },
    },
    "low_contrast": {
        "severity": "DEGRADED",
        "enter_seconds": 3.0,
        "exit_seconds": 5.0,
        "params": {
            "rel_span": 0.50,
            "abs_span": 30.0,
            "exposure_lo": 0.55,
            "exposure_hi": 1.60,
            "max_clip_frac": 0.05,
        },
    },
    "blocked": {
        "severity": "UNUSABLE",
        "enter_seconds": 3.0,
        "exit_seconds": 5.0,
        "params": {
            "tile_rel_std": 0.35,
            "tile_rel_edge": 0.30,
            "tile_abs_std": 4.0,
            "gain_floor": 0.25,
            "uniform_std_max": 3.0,
            "exposure_lo": 0.35,
            "exposure_hi": 1.90,
            "dead_tile_frac": 0.35,
        },
    },
    "frozen": {
        "severity": "UNUSABLE",
        "enter_seconds": 2.0,
        "exit_seconds": 1.0,
        "params": {
            "noise_ratio": 0.20,
            "abs_mad": 0.05,
            "min_noise_sigma": 0.25,
        },
    },
    "moved": {
        "severity": "UNUSABLE",
        "enter_seconds": 3.0,
        "exit_seconds": 5.0,
        "params": {
            "shift_frac": 0.04,
            "edge_corr_min": 0.25,
        },
    },
    "shake": {
        "enabled": True,
        "severity": "DEGRADED",
        "enter_seconds": 3.0,
        "exit_seconds": 5.0,
        "params": {
            "jitter_frac": 0.0012,
            "window_seconds": 1.0,
        },
    },
}


def default_config() -> Config:
    cfg = Config()
    cfg.checks = {
        name: CheckConfig(
            enabled=bool(spec.get("enabled", True)),
            severity=str(spec["severity"]),
            enter_seconds=float(spec["enter_seconds"]),
            exit_seconds=float(spec["exit_seconds"]),
            params=dict(spec["params"]),
        )
        for name, spec in DEFAULT_CHECKS.items()
    }
    return cfg


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str | Path] = None) -> Config:
    cfg = default_config()
    if path is None:
        _validate(cfg)
        return cfg

    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    merged = _deep_merge(cfg.to_dict(), raw)

    try:
        cfg = Config(
            analysis=AnalysisConfig(**merged["analysis"]),
            baseline=BaselineConfig(**merged["baseline"]),
            output=OutputConfig(**merged["output"]),
            checks={
                name: CheckConfig(
                    enabled=bool(spec.get("enabled", True)),
                    severity=str(spec.get("severity", "DEGRADED")),
                    enter_seconds=float(spec.get("enter_seconds", 2.0)),
                    exit_seconds=float(spec.get("exit_seconds", 4.0)),
                    params={k: float(v) for k, v in (spec.get("params") or {}).items()},
                )
                for name, spec in merged["checks"].items()
            },
        )
    except (TypeError, KeyError, ValueError) as exc:
        raise ConfigError(f"invalid config: {exc}") from exc

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    a = cfg.analysis
    if a.max_side < 64:
        raise ConfigError("analysis.max_side must be >= 64")
    if a.stride < 1:
        raise ConfigError("analysis.stride must be >= 1")
    if a.tile_rows < 1 or a.tile_cols < 1:
        raise ConfigError("tile grid must be at least 1x1")
    if a.fallback_fps <= 0:
        raise ConfigError("analysis.fallback_fps must be > 0")
    if a.roi is not None:
        if len(a.roi) != 4:
            raise ConfigError("analysis.roi must be [x, y, w, h] as fractions")
        x, y, w, h = a.roi
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ConfigError("analysis.roi values must be fractions in [0, 1]")
        if x + w > 1.0 + 1e-6 or y + h > 1.0 + 1e-6:
            raise ConfigError("analysis.roi extends outside the frame")

    if cfg.baseline.mode not in ("auto", "file", "none"):
        raise ConfigError("baseline.mode must be one of: auto, file, none")
    if cfg.baseline.mode == "file" and not cfg.baseline.path:
        raise ConfigError("baseline.mode='file' requires baseline.path")

    if not cfg.checks:
        raise ConfigError("at least one check must be configured")
    for name, c in cfg.checks.items():
        if c.severity not in SEVERITIES:
            raise ConfigError(
                f"check '{name}': severity must be one of {SEVERITIES}, got {c.severity!r}"
            )
        if c.enter_seconds <= 0 or c.exit_seconds <= 0:
            raise ConfigError(f"check '{name}': enter/exit seconds must be > 0")
