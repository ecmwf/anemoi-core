import pytest
from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf


@pytest.fixture()
def hydra_overrides() -> None:
    overrides = ["model=transformer"]
    return overrides


@pytest.fixture()
@pytest.mark.parametrize("hydra_overrides", ["model=gnn"])  # , 'model=transformer', 'model=graph_transformer'])
def debug_config(hydra_overrides) -> None:
    initialize(version_base=None, config_path="../training/src/anemoi/training/config", job_name="test_training")
    cfg = compose(config_name="debug", overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    return cfg
