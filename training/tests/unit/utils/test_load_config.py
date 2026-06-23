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

# Directory of the packaged Anemoi config groups (the "base_config").
_ANEMOI_CONFIG_DIR = Path(anemoi.training.__file__).parent / "config"


def _write(tmp_path: Path, content: str, name: str = "user.yaml") -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


def test_load_config_no_defaults(tmp_path: Path) -> None:
    """A config with no defaults list is returned as-is."""
    cfg = load_config(
        _write(tmp_path, "foo: bar\nbaz: 1\n"),
        resolve=False,
    )
    assert cfg.foo == "bar"
    assert cfg.baz == 1
    assert "defaults" not in cfg


def test_load_config_resolves_group_default(tmp_path: Path) -> None:
    """A ``group: value`` entry loads the file and nests it under the group key."""
    content = "defaults:\n- training: single\n\ncustom_key: hello\n"
    cfg = load_config(_write(tmp_path, content), resolve=False)

    assert "training" in cfg
    assert OmegaConf.is_dict(cfg.training)
    assert cfg.custom_key == "hello"
    assert "defaults" not in cfg


def test_load_config_self_at_end_overrides_default(tmp_path: Path) -> None:
    """Without ``_self_``, the primary file's keys override defaults (last write wins)."""
    content = "defaults:\n- training: single\n\ntraining:\n  run_id: overridden_value\n"
    cfg = load_config(_write(tmp_path, content), resolve=False)

    assert cfg.training.run_id == "overridden_value"


def test_load_config_self_before_default(tmp_path: Path) -> None:
    """With ``_self_`` before the group entry, the default is merged last and wins."""
    content = "defaults:\n- _self_\n- training: single\n\ntraining:\n  run_id: primary_value\n"
    cfg = load_config(_write(tmp_path, content), resolve=False)

    # training/single.yaml is merged after _self_, so its run_id (null) wins.
    assert cfg.training.run_id is None


def test_load_config_null_default_skipped(tmp_path: Path) -> None:
    """A ``group: null`` entry disables loading that default entirely."""
    content = "defaults:\n- training: null\n\ntraining:\n  run_id: kept\n"
    cfg = load_config(_write(tmp_path, content), resolve=False)

    # No training defaults were loaded; the primary config's value is preserved.
    assert cfg.training.run_id == "kept"


def test_load_config_recursive_defaults(tmp_path: Path) -> None:
    """Defaults in sub-configs (e.g. training/single.yaml) are resolved recursively."""
    content = "defaults:\n- training: single\n\ncustom: yes\n"
    cfg = load_config(_write(tmp_path, content), resolve=False)

    # training/single.yaml itself has ``- scalers: global`` in its defaults,
    # which should surface as cfg.training.scalers.
    assert "scalers" in cfg.training
    assert OmegaConf.is_dict(cfg.training.scalers)


def test_load_config_relative_input_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative input_config path is resolved against the cwd."""
    content = "defaults:\n- training: single\n\ncustom_key: hello\n"
    abs_path = _write(tmp_path, content)

    monkeypatch.chdir(tmp_path)
    cfg_rel = load_config("user.yaml", resolve=False)
    cfg_abs = load_config(abs_path, base_config=_ANEMOI_CONFIG_DIR, resolve=False)

    assert OmegaConf.to_container(cfg_rel) == OmegaConf.to_container(cfg_abs)


def test_load_config_missing_input_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_missing_base_config(tmp_path: Path) -> None:
    cfg_file = _write(tmp_path, "foo: bar\n")
    with pytest.raises(FileNotFoundError):
        load_config(cfg_file, base_config=tmp_path / "nonexistent_dir")


def test_load_config_missing_default_file_raises(tmp_path: Path) -> None:
    content = "defaults:\n- training: nonexistent_config\n"
    with pytest.raises(FileNotFoundError):
        load_config(_write(tmp_path, content))
