from pathlib import Path

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
    with initialize(version_base=None, config_path="../../training/src/anemoi/training/config", job_name="test_basic"):
        template = compose(
            config_name="debug", overrides=overrides
        )  # apply architecture overrides to template since they override a default
        global_modifications = OmegaConf.load(Path.cwd() / "tests/integration/test_training_cycle.yaml")
        specific_modifications = OmegaConf.load(Path.cwd() / "tests/integration/test_basic.yaml")
        cfg = OmegaConf.merge(template, global_modifications, specific_modifications)
        OmegaConf.resolve(cfg)
        return cfg


@pytest.fixture()
def stretched_config() -> None:
    with initialize(version_base=None, config_path="", job_name="test_stretched"):
        template = compose(config_name="stretched")
        global_modifications = OmegaConf.load(Path.cwd() / "tests/integration/test_training_cycle.yaml")
        specific_modifications = OmegaConf.load(Path.cwd() / "tests/integration/test_stretched.yaml")
        cfg = OmegaConf.merge(template, global_modifications, specific_modifications)
        OmegaConf.resolve(cfg)
        return cfg
