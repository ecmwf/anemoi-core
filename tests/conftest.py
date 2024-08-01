import pytest
from hydra import compose
from hydra import initialize

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule


@pytest.fixture()
def config(request):
    overrides = request.param
    with initialize(version_base=None, config_path="../src/anemoi/training/config"):
        # config is relative to a module
        config = compose(config_name="debug", overrides=overrides)
    return config


@pytest.fixture()
def datamodule():
    with initialize(version_base=None, config_path="../src/anemoi/training/config"):
        # config is relative to a module
        cfg = compose(config_name="config")
    return AnemoiDatasetsDataModule(cfg)
