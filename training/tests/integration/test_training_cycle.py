# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import shutil

import pytest
from omegaconf import DictConfig

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.train import AnemoiTrainer

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners


LOGGER = logging.getLogger(__name__)


@pytest.mark.longtests
def test_training_cycle_architecture_configs(architecture_config: DictConfig) -> None:
    AnemoiTrainer(architecture_config).train()
    shutil.rmtree(architecture_config.hardware.paths.output)


def test_config_validation_architecture_configs(architecture_config: DictConfig) -> None:
    BaseSchema(**architecture_config)


@pytest.mark.longtests
def test_training_cycle_stretched(stretched_config: DictConfig) -> None:
    AnemoiTrainer(stretched_config).train()
    shutil.rmtree(stretched_config.hardware.paths.output)


def test_config_validation_stretched(stretched_config: DictConfig) -> None:
    BaseSchema(**stretched_config)


@pytest.mark.longtests
def test_training_cycle_lam(lam_config: DictConfig) -> None:
    AnemoiTrainer(lam_config).train()
    shutil.rmtree(lam_config.hardware.paths.output)


def test_config_validation_lam(lam_config: DictConfig) -> None:
    BaseSchema(**lam_config)


if __name__ == "__main__":
    from pathlib import Path

    from hydra import compose
    from hydra import initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_stretched"):
        template = compose(config_name="lam")
        use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_lam.yaml")
        testing_modifications = OmegaConf.load(
            Path.cwd() / "training/tests/integration/config/testing_modifications.yaml",
        )
        cfg = OmegaConf.merge(template, testing_modifications, use_case_modifications)
        OmegaConf.resolve(cfg)
        AnemoiTrainer(cfg).train()
