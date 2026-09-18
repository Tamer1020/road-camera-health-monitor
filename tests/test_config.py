"""Configuration loading, validation and the config-drift guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from road_health.config import Config, default_config, load_config
from road_health.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"


def test_default_config_is_valid():
    cfg = default_config()
    assert cfg.checks
    assert all(c.severity in ("OK", "DEGRADED", "UNUSABLE") for c in cfg.checks.values())


def test_shipped_yaml_matches_builtin_defaults():
    assert DEFAULT_YAML.is_file(), "configs/default.yaml is missing"
    shipped = load_config(DEFAULT_YAML).to_dict()
    builtin = default_config().to_dict()
    assert shipped["analysis"] == builtin["analysis"]
    assert shipped["baseline"] == builtin["baseline"]
    assert shipped["output"] == builtin["output"]
    assert set(shipped["checks"]) == set(builtin["checks"])
    for name in builtin["checks"]:
        assert shipped["checks"][name] == builtin["checks"][name], (
            f"configs/default.yaml has drifted from the built-in default for '{name}'"
        )


def test_partial_config_is_deep_merged(tmp_path):
    p = tmp_path / "partial.yaml"
    p.write_text("analysis:\n  stride: 3\nchecks:\n  blur:\n    enter_seconds: 9.5\n")
    cfg = load_config(p)
    assert cfg.analysis.stride == 3
    assert cfg.analysis.max_side == default_config().analysis.max_side
    assert cfg.checks["blur"].enter_seconds == 9.5
    assert cfg.checks["blur"].params == default_config().checks["blur"].params


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_malformed_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("analysis: [unclosed\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_non_mapping_root_raises(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ConfigError):
        load_config(p)


@pytest.mark.parametrize(
    "body",
    [
        "checks:\n  blur:\n    severity: PANIC\n",
        "analysis:\n  stride: 0\n",
        "analysis:\n  max_side: 16\n",
        "analysis:\n  roi: [0.5, 0.0, 0.8, 1.0]\n",
        "analysis:\n  roi: [0.0, 0.0, 1.0]\n",
        "checks:\n  blur:\n    enter_seconds: 0\n",
        "baseline:\n  mode: sometimes\n",
        "baseline:\n  mode: file\n  path: null\n",
    ],
)
def test_invalid_configs_are_rejected(tmp_path, body):
    p = tmp_path / "bad.yaml"
    p.write_text(body)
    with pytest.raises(ConfigError):
        load_config(p)


def test_seconds_to_frames_uses_effective_rate():
    cfg = default_config()
    cfg.checks["blur"].enter_seconds = 2.0
    cfg.analysis.stride = 1
    assert cfg.enter_frames("blur", 25.0) == 50
    cfg.analysis.stride = 5
    assert cfg.enter_frames("blur", 25.0) == 10


def test_enter_frames_never_zero():
    cfg = default_config()
    cfg.checks["blur"].enter_seconds = 0.001
    assert cfg.enter_frames("blur", 25.0) >= 1
