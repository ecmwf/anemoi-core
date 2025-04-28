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


@skip_if_offline
@pytest.mark.longtests
def test_restart_training(gnn_config_with_data: DictConfig) -> None:

    AnemoiTrainer(gnn_config_with_data).train()

    cfg = gnn_config_with_data
    output_dir = Path(cfg.hardware.paths.output + "checkpoint")

    if not output_dir.exists():
        msg = f"Checkpoint directory not found at: {output_dir}"
        raise FileNotFoundError(msg)

    run_dirs = [item for item in output_dir.iterdir() if item.is_dir()]
    if len(run_dirs) != 1:
        found_dirs = [d.name for d in run_dirs]
        msg = f"Expected exactly one run_id directory, found {len(run_dirs)}: {found_dirs}"
        raise RuntimeError(msg)

    checkpoint_dir = run_dirs[0]
    run_id = checkpoint_dir.name

    assert len(list(checkpoint_dir.glob("anemoi-by_epoch-*.ckpt"))) == 2
    assert len(list(checkpoint_dir.glob("inference-anemoi-by_epoch-*.ckpt"))) == 2
    assert (checkpoint_dir / "last.ckpt").exists()
    assert (checkpoint_dir / "inference-last.ckpt").exists()

    cfg.training.run_id = run_id
    cfg.training.max_epochs = 3
    AnemoiTrainer(cfg).train()

    assert len(list(checkpoint_dir.glob("anemoi-by_epoch-*.ckpt"))) == 3
    assert len(list(checkpoint_dir.glob("inference-anemoi-by_epoch-*.ckpt"))) == 3
