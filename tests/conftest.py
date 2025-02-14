import pytest
from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf


@pytest.fixture(params= [["model=gnn"], ["model=graphtransformer"], ["model=transformer", "graph=encoder_decoder_only"]])
def debug_config(request) -> None:
    overrides = request.param
    with initialize(version_base=None, config_path="", job_name="test_training"):
        print(overrides)
        cfg = compose(config_name="basic_config", overrides=overrides)
        OmegaConf.resolve(cfg)
        return cfg
