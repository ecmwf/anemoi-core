import pytest
from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf


@pytest.fixture(
    params=[
        ["model=gnn"],
        ["model=graphtransformer"],
        [
            "model=transformer",
            "graph=encoder_decoder_only",
            "model.processor.attention_implementation=scaled_dot_product_attention",
        ],
    ]
)
def architecture_config(request) -> None:
    overrides = request.param
    with initialize(version_base=None, config_path="", job_name="test_training"):
        cfg = compose(config_name="basic_config", overrides=overrides)
        OmegaConf.resolve(cfg)
        return cfg


@pytest.fixture()
def stretched_config() -> None:
    with initialize(version_base=None, config_path="", job_name="test_stretched"):
        cfg = compose(config_name="stretched_config")
        OmegaConf.resolve(cfg)
        return cfg
