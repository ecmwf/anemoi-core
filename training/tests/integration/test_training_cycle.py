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
def test_training_cycle_grid_configs(stretched_config: DictConfig) -> None:
    AnemoiTrainer(stretched_config).train()
    shutil.rmtree(stretched_config.hardware.paths.output)


def test_config_validation_grid_configs(stretched_config: DictConfig) -> None:
    BaseSchema(**stretched_config)

    
@pytest.mark.longtests
def test_training_cycle_architecture_configs(lam_config: DictConfig) -> None:
    AnemoiTrainer(lam_config).train()
    shutil.rmtree(lam_config.hardware.paths.output)


def test_config_validation_architecture_configs(lam_config: DictConfig) -> None:
    BaseSchema(**lam_config)
