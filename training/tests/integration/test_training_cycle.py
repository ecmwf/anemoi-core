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
from pathlib import Path

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
def test_training_cycle_without_config_validation(gnn_config_with_data: DictConfig) -> None:
    gnn_config_with_data.config_validation = False
    gnn_config_with_data.hardware.files.graph = "dummpy.pt"  # Mandatory input when running without config validation
    AnemoiTrainer(gnn_config_with_data).train()


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


@skip_if_offline
@pytest.mark.longtests
def test_training_cycle_lam_with_existing_graph(lam_config_with_data_and_graph: DictConfig) -> None:
    AnemoiTrainer(lam_config_with_data_and_graph).train()


def test_config_validation_lam(lam_config: DictConfig) -> None:
    BaseSchema(**lam_config)


@skip_if_offline
@pytest.mark.longtests
def test_training_cycle_ensemble(ensemble_config_with_data: DictConfig) -> None:
    AnemoiTrainer(ensemble_config_with_data).train()


def test_config_validation_ensemble(ensemble_config: DictConfig) -> None:
    BaseSchema(**ensemble_config)


@skip_if_offline
@pytest.mark.longtests
def test_restart_training(gnn_config_with_data: DictConfig) -> None:

    AnemoiTrainer(gnn_config_with_data).train()

    cfg = gnn_config_with_data
    output_dir = Path(cfg.hardware.paths.output + "checkpoint")

    assert output_dir.exists(), f"Checkpoint directory not found at: {output_dir}"

    run_dirs = [item for item in output_dir.iterdir() if item.is_dir()]
    assert (
        len(run_dirs) == 1
    ), f"Expected exactly one run_id directory, found {len(run_dirs)}: {[d.name for d in run_dirs]}"

    checkpoint_dir = run_dirs[0]
    assert len(list(checkpoint_dir.glob("anemoi-by_epoch-*.ckpt"))) == 2, "Expected 2 checkpoints after first run"

    cfg.training.run_id = checkpoint_dir.name
    cfg.training.max_epochs = 3
    AnemoiTrainer(cfg).train()

    assert len(list(checkpoint_dir.glob("anemoi-by_epoch-*.ckpt"))) == 3, "Expected 3 checkpoints after second run"


@skip_if_offline
@pytest.mark.longtests
def test_restart_from_existing_checkpoint(gnn_config_with_checkpoint: DictConfig) -> None:
    AnemoiTrainer(gnn_config_with_checkpoint).train()
