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

import pytest
from omegaconf import DictConfig

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import skip_if_offline

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners


LOGGER = logging.getLogger(__name__)


@skip_if_offline
@pytest.mark.longtests
def test_training_cycle_architecture_configs(architecture_config_with_data: DictConfig) -> None:
    AnemoiTrainer(architecture_config_with_data).train()


def test_config_validation_architecture_configs(architecture_config: DictConfig) -> None:
    BaseSchema(**architecture_config)


@skip_if_offline
@pytest.mark.longtests
def test_training_cycle_stretched(stretched_config_with_data: DictConfig) -> None:
    AnemoiTrainer(stretched_config_with_data).train()


def test_config_validation_stretched(stretched_config: DictConfig) -> None:
    BaseSchema(**stretched_config)


@skip_if_offline
@pytest.mark.longtests
def test_training_cycle_lam(lam_config_with_data: DictConfig) -> None:
    AnemoiTrainer(lam_config_with_data).train()


def test_config_validation_lam(lam_config: DictConfig) -> None:
    BaseSchema(**lam_config)
