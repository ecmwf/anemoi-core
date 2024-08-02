import datetime
import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from torch import nn

from anemoi.training.diagnostics.callbacks import AnemoiCheckpoint
from anemoi.training.utils.config import DotConfig
from anemoi.training.utils.jsonify import map_config_to_primitives


def read_zipped_model_metadata(path: Path) -> dict:
    with ZipFile(path) as myzip:
        base = Path(path).stem
        with myzip.open(f"{base}/ai-models.json") as myfile:
            data = myfile.read()
    return json.loads(data.decode("utf-8"))


class DummyModel(torch.nn.Module):
    """Dummy pytorch model for testing."""

    def __init__(self, *, config=DotConfig, metadata=dict):
        super().__init__()

        self.config = config
        self.metadata = metadata
        self.fc1 = nn.Linear(32, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DummyModule(BoringModel):
    """Dummy lightning module for testing."""

    def __init__(self, *, config=DotConfig, metadata=dict) -> None:
        super().__init__()
        self.model = DummyModel(config=config, metadata=metadata)
        # self.hparams.update({"metadata": dict()})
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@pytest.fixture()
def tmp_path() -> str:
    return os.environ["HPCPERM"] + "/aifs/checkpoint_test"


@pytest.fixture()
def config() -> DictConfig:
    config = DictConfig({"diagnostics": {"log": {"wandb": {"enabled": False}, "mlflow": {"enabled": False}}}})
    return DotConfig(map_config_to_primitives(OmegaConf.to_container(config, resolve=True)))


@pytest.fixture()
def checkpoint_settings(tmp_path: str):
    return {
        "dirpath": tmp_path,
        "verbose": False,
        "save_weights_only": False,
        "auto_insert_metric_name": False,
        "save_on_train_epoch_end": False,
        "enable_version_counter": False,
    }


@pytest.fixture()
def callback(tmp_path: str, config: DictConfig, checkpoint_settings: dict):
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


@pytest.fixture()
def metadata(config: DictConfig) -> dict:
    return map_config_to_primitives(
        {
            "config": config,
        },
    )


@pytest.fixture()
def model(metadata: dict, config: DictConfig) -> DummyModule:
    kwargs = {
        "metadata": metadata,
        "config": config,
    }
    return DummyModule(**kwargs)


@pytest.mark.data_dependent()
def test_same_uuid(tmp_path: str, callback: AnemoiCheckpoint, model: DummyModule) -> None:
    """Test if the inference checkpoints and lightning checkpoints store same uuid.

    Args:
        tmp_path (str): path to checkpoint dir
        callback (AnemoiCheckpoint): callback to store checkpoints
        model (DummyModule): dummy lightning module
    """
    trainer = Trainer(default_root_dir=tmp_path, callbacks=[callback], max_epochs=2, logger=False)
    trainer.fit(model)

    for ckpt_path in Path(tmp_path).iterdir():
        if str(ckpt_path.name).startswith("inference"):
            pl_ckpt_name = (ckpt_path.name).replace("inference-", "")

            if Path(tmp_path + "/" + pl_ckpt_name).exists():
                uuid = read_zipped_model_metadata(ckpt_path)["uuid"]

                pl_model = DummyModule.load_from_checkpoint(tmp_path + "/" + pl_ckpt_name)

                assert uuid == pl_model.hparams["metadata"]["uuid"]

    shutil.rmtree(tmp_path)
