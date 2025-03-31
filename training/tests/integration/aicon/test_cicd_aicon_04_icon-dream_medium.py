# (C) Copyright 2025 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
#
# Various tests of the Anemoi components using a sample data set.
#
# This script is not part of a productive ML workflow, but is
# used for CI/CD!
import os
import pathlib
import tempfile
from typing import Optional

import matplotlib as mpl
import pytest
import torch
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from typeguard import typechecked

import anemoi.training
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.train import AnemoiTrainer

os.environ["ANEMOI_BASE_SEED"] = "42"
os.environ["ANEMOI_CONFIG_PATH"] = str(pathlib.Path(anemoi.training.__file__).parent / "config")
mpl.use("agg")


@typechecked
def aicon_config(output_dir: Optional[str] = None) -> DictConfig:
    with initialize(version_base=None, config_path="./"):
        config = compose(config_name="test_cicd_aicon_04_icon-dream_medium")

    if output_dir is not None:
        config.hardware.paths.output = output_dir
        config.hardware.paths.graph = output_dir

    return config


@typechecked
def trainer(config: DictConfig) -> AnemoiTrainer:
    grid_filename = config.graph.nodes.icon_mesh.node_builder.grid_filename
    with tempfile.NamedTemporaryFile(suffix=".nc") as grid_fp:
        if grid_filename.startswith(("http://", "https://")):
            import urllib.request

            urllib.request.urlretrieve(grid_filename, grid_fp.name)  # noqa: S310
            config.graph.nodes.icon_mesh.node_builder.grid_filename = grid_fp.name

        trainer = AnemoiTrainer(config)
        initial_sum = torch.tensor(list(map(torch.sum, trainer.model.parameters()))).sum()
        trainer.train()
        final_sum = torch.tensor(list(map(torch.sum, trainer.model.parameters()))).sum()
    return trainer, initial_sum, final_sum


@pytest.fixture
def get_trainer() -> tuple:
    with tempfile.TemporaryDirectory() as output_dir:
        return trainer(aicon_config(output_dir=output_dir))


def test_config_validation_aicon() -> None:
    BaseSchema(**aicon_config())


@pytest.mark.longtests
def test_main(get_trainer: tuple) -> None:
    trainer, initial_sum, final_sum = get_trainer
    assert trainer
    assert initial_sum != final_sum
