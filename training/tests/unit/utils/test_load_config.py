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

from anemoi.training.utils.config import load_config

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
