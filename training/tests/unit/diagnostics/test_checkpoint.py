# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import shutil
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from torch import nn

from anemoi.training.diagnostics.callbacks import AnemoiCheckpoint
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.utils.checkpoints import load_metadata
from anemoi.utils.config import DotDict


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
def tmp_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    # returns a session-scoped temporary directory
    return str(tmp_path_factory.mktemp("test_checkpoint"))


@pytest.fixture
def config() -> DictConfig:
    config = DictConfig({"diagnostics": {"log": {"wandb": {"enabled": False}, "mlflow": {"enabled": False}}}})
    return DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True)))


@pytest.fixture
def checkpoint_settings(tmp_path: str) -> dict[str : str | bool]:
    return {
        "dirpath": tmp_path,
        "verbose": False,
        "save_weights_only": False,
        "auto_insert_metric_name": False,
        "save_on_train_epoch_end": False,
        "enable_version_counter": False,
    }


@pytest.fixture
def callback(tmp_path: str, config: DictConfig, checkpoint_settings: dict) -> AnemoiCheckpoint:
    callback = AnemoiCheckpoint(
        config=config,
        filename="{step}",
        save_last=True,
        train_time_interval=datetime.timedelta(seconds=0.1),
        save_top_k=3,
        monitor="step",
        mode="max",
        **checkpoint_settings,
    )
    callback.dirpath = tmp_path
    return callback


@pytest.fixture
def metadata(config: DictConfig) -> dict:
    return map_config_to_primitives(
        {
            "config": config,
        },
    )


@pytest.fixture
def model(metadata: dict, config: DictConfig) -> DummyModule:
    kwargs = {
        "metadata": metadata,
        "config": config,
    }
    return DummyModule(**kwargs)


@pytest.mark.data_dependent
def test_same_uuid(tmp_path: str, callback: AnemoiCheckpoint, model: DummyModule) -> None:
    """Test if the inference checkpoints and lightning checkpoints store same uuid.

    Args:
        tmp_path (str): path to checkpoint dir
        callback (AnemoiCheckpoint): callback to store checkpoints
        model (DummyModule): dummy lightning module
    """
    trainer = Trainer(default_root_dir=tmp_path, accelerator="cpu", callbacks=[callback], max_epochs=2, logger=False)
    trainer.fit(model)

    for ckpt_path in Path(tmp_path).iterdir():
        if str(ckpt_path.name).startswith("inference"):
            pl_ckpt_name = (ckpt_path.name).replace("inference-", "")

            if Path(tmp_path + "/" + pl_ckpt_name).exists():
                uuid = load_metadata(ckpt_path)["uuid"]

                pl_model = DummyModule.load_from_checkpoint(tmp_path + "/" + pl_ckpt_name)

                assert uuid == pl_model.hparams["metadata"]["uuid"]

    shutil.rmtree(tmp_path)
