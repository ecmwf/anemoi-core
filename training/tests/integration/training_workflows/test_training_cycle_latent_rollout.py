# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Training cycle with the rollout in latent space."""

import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict

from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_latent_rollout(
    global_config: tuple[DictConfig, str, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url, _ = global_config
    get_test_archive(url)
    with open_dict(cfg):
        cfg.training.method = OmegaConf.create({"_target_": "anemoi.training.train.methods.LatentRolloutTraining"})
        cfg.task.rollout = OmegaConf.create({"start": 3, "epoch_increment": 0, "maximum": 3})
    AnemoiTrainer(cfg).train()
