# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


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
    ],
)
def architecture_config(request: pytest.FixtureRequest) -> None:
    overrides = request.param
    with initialize(version_base=None, config_path="../../training/src/anemoi/training/config", job_name="test_basic"):
        template = compose(
            config_name="debug",
            overrides=overrides,
        )  # apply architecture overrides to template since they override a default
        global_modifications = OmegaConf.load(Path.cwd() / "tests/integration/test_training_cycle.yaml")
        specific_modifications = OmegaConf.load(Path.cwd() / "tests/integration/test_basic.yaml")
        cfg = OmegaConf.merge(template, global_modifications, specific_modifications)
        OmegaConf.resolve(cfg)
        return cfg


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--longtests",
        action="store_true",
        dest="longtests",
        default=False,
        help="enable longrundecorated tests",
    )
