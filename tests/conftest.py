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
    initialize(version_base=None, config_path="", job_name="test_training")
    print(hydra_overrides)
    cfg = compose(config_name="basic_config", overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    return cfg
