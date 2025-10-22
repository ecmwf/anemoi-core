# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from pytorch_lightning.demos.boring_classes import BoringModel

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.utils.config import DotDict


@pytest.fixture
def config(request: SubRequest) -> DictConfig:
    overrides = request.param
    with initialize(version_base=None, config_path="../src/anemoi/training/config"):
        # config is relative to a module
        return compose(config_name="debug", overrides=overrides)


@pytest.fixture
def datamodule() -> AnemoiDatasetsDataModule:
    with initialize(version_base=None, config_path="../src/anemoi/training/config"):
        # config is relative to a module
        cfg = compose(config_name="config")
    return AnemoiDatasetsDataModule(cfg)


class DummyModel(torch.nn.Module):
    """Dummy pytorch model for testing."""

    def __init__(self, *, config: DotDict, metadata: dict):
        super().__init__()

        self.config = config
        self.metadata = metadata
        self.supporting_arrays = {}
        self.fc1 = nn.Linear(32, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class DummyModule(BoringModel):
    """Dummy lightning module for testing."""

    def __init__(self, *, config: DotDict, metadata: dict) -> None:
        super().__init__()
        self.model = DummyModel(config=config, metadata=metadata)
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@pytest.fixture
def model(metadata: dict, config: DictConfig) -> DummyModule:
    kwargs = {
        "metadata": metadata,
        "config": config,
    }
    return DummyModule(**kwargs)


@pytest.fixture
def metadata(config: DictConfig) -> dict:
    return map_config_to_primitives(
        {
            "config": config,
        },
    )
