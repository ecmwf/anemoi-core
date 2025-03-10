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


def pytest_addoption(parser):
    parser.addoption(
        "--longtests", action="store_true", dest="longtests", default=False, help="enable longrundecorated tests"
    )
