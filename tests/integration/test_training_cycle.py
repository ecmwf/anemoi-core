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

from anemoi.training.train.train import AnemoiTrainer

LOGGER = logging.getLogger(__name__)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_training_cycle_architecture_configs(architecture_config) -> None:
    AnemoiTrainer(architecture_config).train()
    shutil.rmtree(architecture_config.hardware.paths.output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_training_cycle_grid_configs(stretched_config) -> None:
    AnemoiTrainer(stretched_config).train()
    shutil.rmtree(stretched_config.hardware.paths.output)


if __name__ == "__main__":
    from hydra import compose
    from hydra import initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="", job_name="test_training"):
        cfg = compose(config_name="stretched_config", overrides=[])
        OmegaConf.resolve(cfg)
        test_training_cycle_architecture_configs(cfg)
