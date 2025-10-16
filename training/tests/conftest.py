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
from _pytest.fixtures import SubRequest
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig


def _get_config_path() -> str:
    """Get the config path relative to the project root, working from any directory."""
    # Find the config directory by looking for src/anemoi/training/config
    # This works whether running from training/ or training/tests/
    current = Path.cwd()

    # Try from current directory first (running from training/)
    config_path = current / "src" / "anemoi" / "training" / "config"
    if config_path.exists():
        return str(config_path)

    # Try from parent directory (running from training/tests/)
    config_path = current.parent / "src" / "anemoi" / "training" / "config"
    if config_path.exists():
        return str(config_path)

    # Fallback: use relative path from tests/ directory
    return "../src/anemoi/training/config"


@pytest.fixture
def config(request: SubRequest) -> DictConfig:
    overrides = request.param
    config_path = _get_config_path()
    with initialize(version_base=None, config_path=config_path):
        # config is relative to a module
        return compose(config_name="debug", overrides=overrides)


@pytest.fixture
def datamodule():  # type: ignore[no-untyped-def]
    """Lazy-load AnemoiDatasetsDataModule to avoid expensive import at test collection time."""
    from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

    config_path = _get_config_path()
    with initialize(version_base=None, config_path=config_path):
        # config is relative to a module
        cfg = compose(config_name="config")
    return AnemoiDatasetsDataModule(cfg)
