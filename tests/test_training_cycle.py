# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import shutil

import pytest
import torch
from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf

from anemoi.training.train.train import AnemoiTrainer

LOGGER = logging.getLogger(__name__)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_training_cycle_debug_config(debug_config) -> None:
    AnemoiTrainer(debug_config).train()
    shutil.rmtree(debug_config.hardware.paths.output)
