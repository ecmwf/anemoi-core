# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import shutil

import pytest
from omegaconf import DictConfig

from anemoi.training.train.train import AnemoiTrainer

longtests = pytest.mark.skipif("not config.getoption('longtests')", reason="need --longtests option to run")

LOGGER = logging.getLogger(__name__)


@longtests
def test_training_cycle_architecture_configs(architecture_config: DictConfig) -> None:
    AnemoiTrainer(architecture_config).train()
    shutil.rmtree(architecture_config.hardware.paths.output)


if __name__ == "__main__":
    from pathlib import Path

    from hydra import compose
    from hydra import initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="../../training/src/anemoi/training/config", job_name="test_basic"):
        template = compose(
            config_name="debug",
        )  # apply architecture overrides to template since they override a default
        global_modifications = OmegaConf.load(Path.cwd() / "tests/integration/test_training_cycle.yaml")
        specific_modifications = OmegaConf.load(Path.cwd() / "tests/integration/test_basic.yaml")
        cfg = OmegaConf.merge(template, global_modifications, specific_modifications)
        OmegaConf.resolve(cfg)

    test_training_cycle_architecture_configs(cfg)
