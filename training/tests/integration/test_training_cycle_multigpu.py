# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

import pytest
from omegaconf import DictConfig

from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners


def _configure_rollout_gt1(cfg: DictConfig) -> None:
    cfg.training.rollout.start = 2
    cfg.training.rollout.max = 2
    cfg.training.rollout.epoch_increment = 0
    cfg.training.max_epochs = 1
    cfg.training.num_sanity_val_steps = 0
    cfg.dataloader.limit_batches.training = 1
    cfg.dataloader.limit_batches.validation = 1


@skip_if_offline
@pytest.mark.slow
@pytest.mark.multigpu
@pytest.mark.parametrize(
    "keep_batch_sharded",
    [True, False],
    ids=["keep_batch_sharded", "allgather_batch"],
)
def test_training_cycle_model_sharding(
    multigpu_config: tuple[DictConfig, list[str], str],
    get_test_archive: GetTestArchive,
    keep_batch_sharded: bool,
) -> None:
    """Test multigpu model-sharding training for all model types."""
    cfg, urls, _model_type = multigpu_config
    cfg.model.keep_batch_sharded = keep_batch_sharded
    for url in urls:
        get_test_archive(url)
    AnemoiTrainer(cfg).train()


@skip_if_offline
@pytest.mark.slow
@pytest.mark.multigpu
def test_training_cycle_model_sharding_rollout2(
    multigpu_rollout_config: tuple[DictConfig, list[str], str],
    get_test_archive: GetTestArchive,
) -> None:
    """Test multigpu model-sharding training with rollout > 1."""
    cfg, urls, _model_type = multigpu_rollout_config
    cfg.model.keep_batch_sharded = True
    _configure_rollout_gt1(cfg)
    for url in urls:
        get_test_archive(url)
    AnemoiTrainer(cfg).train()
