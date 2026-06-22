# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import anemoi.training
from anemoi.training.utils.config import load_config
from anemoi.training.utils.config import load_config2

# Directory of the packaged Anemoi config groups (the "base_config" for load_config2 tests).
_ANEMOI_CONFIG_DIR = Path(anemoi.training.__file__).parent / "config"

# A minimal primary config whose ``defaults:`` list pulls a group from the packaged
# configs (resolved via the auto-discovered AnemoiSearchPathPlugin).
USER_CONFIG = """
defaults:
- training: default
- _self_

custom_key: custom_value
"""


def _write(tmp_path: Path, content: str = USER_CONFIG, name: str = "user.yaml") -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


def test_load_config_pulls_packaged_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path), resolve=False)

    # The group referenced in ``defaults:`` was resolved from the package config...
    assert "training" in cfg
    assert OmegaConf.is_dict(cfg.training)
    # ...and the primary file's own keys are preserved.
    assert cfg.custom_key == "custom_value"


def test_load_config_relative_and_absolute_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    abs_path = _write(tmp_path)
    cfg_abs = load_config(abs_path, resolve=False)

    monkeypatch.chdir(tmp_path)
    cfg_rel = load_config("user.yaml", resolve=False)

    assert OmegaConf.to_container(cfg_abs) == OmegaConf.to_container(cfg_rel)


def test_load_config_applies_overrides(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path), overrides=["custom_key=overridden"], resolve=False)
    assert cfg.custom_key == "overridden"


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# Tests for load_config2
# ---------------------------------------------------------------------------


def _write2(tmp_path: Path, content: str, name: str = "user.yaml") -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


def test_load_config2_no_defaults(tmp_path: Path) -> None:
    """A config with no defaults list is returned as-is."""
    cfg = load_config2(
        _ANEMOI_CONFIG_DIR,
        _write2(tmp_path, "foo: bar\nbaz: 1\n"),
        resolve=False,
    )
    assert cfg.foo == "bar"
    assert cfg.baz == 1
    assert "defaults" not in cfg


def test_load_config2_resolves_group_default(tmp_path: Path) -> None:
    """A ``group: value`` entry loads the file and nests it under group key."""
    content = "defaults:\n- training: single\n\ncustom_key: hello\n"
    cfg = load_config2(_ANEMOI_CONFIG_DIR, _write2(tmp_path, content), resolve=False)

    assert "training" in cfg
    assert OmegaConf.is_dict(cfg.training)
    assert cfg.custom_key == "hello"
    assert "defaults" not in cfg


def test_load_config2_self_at_end_overrides_default(tmp_path: Path) -> None:
    """Without _self_, the primary file's keys override defaults."""
    content = "defaults:\n- training: single\n\ntraining:\n  run_id: overridden_value\n"
    cfg = load_config2(_ANEMOI_CONFIG_DIR, _write2(tmp_path, content), resolve=False)

    assert cfg.training.run_id == "overridden_value"


def test_load_config2_self_before_default(tmp_path: Path) -> None:
    """With _self_ before the group, the default wins over the primary key."""
    content = "defaults:\n- _self_\n- training: single\n\ntraining:\n  run_id: primary_value\n"
    cfg = load_config2(_ANEMOI_CONFIG_DIR, _write2(tmp_path, content), resolve=False)

    # The default (training/single.yaml) is merged last, so its run_id (null) wins.
    assert cfg.training.run_id is None


def test_load_config2_null_default_skipped(tmp_path: Path) -> None:
    """A ``group: null`` entry disables the default — no key is added."""
    content = "defaults:\n- training: single\n\ntraining:\n  weight_averaging: should_stay\n"
    cfg = load_config2(_ANEMOI_CONFIG_DIR, _write2(tmp_path, content), resolve=False)
    # weight_averaging is null in training/single.yaml defaults, so no nested key is added
    # from it. The primary config's value survives.
    assert cfg.training.weight_averaging == "should_stay"


def test_load_config2_recursive_defaults(tmp_path: Path) -> None:
    """Defaults in sub-configs (e.g. training/single.yaml) are also resolved recursively."""
    content = "defaults:\n- training: single\n\ncustom: yes\n"
    cfg = load_config2(_ANEMOI_CONFIG_DIR, _write2(tmp_path, content), resolve=False)

    # training/single.yaml has `- scalers: global` in its own defaults,
    # which should surface as training.scalers in the merged config.
    assert "scalers" in cfg.training
    assert OmegaConf.is_dict(cfg.training.scalers)


def test_load_config2_missing_input_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config2(_ANEMOI_CONFIG_DIR, tmp_path / "does_not_exist.yaml")


def test_load_config2_missing_base_config(tmp_path: Path) -> None:
    cfg_file = _write2(tmp_path, "foo: bar\n")
    with pytest.raises(FileNotFoundError):
        load_config2(tmp_path / "nonexistent_dir", cfg_file)


def test_load_config2_missing_default_file_raises(tmp_path: Path) -> None:
    content = "defaults:\n- training: nonexistent_config\n"
    with pytest.raises(FileNotFoundError):
        load_config2(_ANEMOI_CONFIG_DIR, _write2(tmp_path, content))
