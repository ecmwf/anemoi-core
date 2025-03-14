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
    ],
)
def architecture_config(request: pytest.FixtureRequest, testing_modifications_with_temp_dir: OmegaConf) -> None:
    overrides = request.param
    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_basic"):
        template = compose(
            config_name="debug",
            overrides=overrides,
        )  # apply architecture overrides to template since they override a default
        use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_basic.yaml")
        cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
        OmegaConf.resolve(cfg)
        return cfg


@pytest.fixture
def testing_modifications_with_temp_dir(tmp_path) -> OmegaConf:
    testing_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/testing_modifications.yaml")
    temp_dir = str(tmp_path)
    testing_modifications.hardware.paths.output = temp_dir
    return testing_modifications


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--longtests",
        action="store_true",
        dest="longtests",
        default=False,
        help="enable longrundecorated tests",
    )
