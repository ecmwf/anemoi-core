# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import shutil

import pytest
from omegaconf import DictConfig

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.train import AnemoiTrainer

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners


LOGGER = logging.getLogger(__name__)


# @pytest.mark.longtests
# def test_training_cycle_architecture_configs(architecture_config: DictConfig) -> None:
#     AnemoiTrainer(architecture_config).train()
#     shutil.rmtree(architecture_config.hardware.paths.output)


# def test_config_validation_architecture_configs(architecture_config: DictConfig) -> None:
#     BaseSchema(**architecture_config)


@pytest.mark.longtests
def test_training_cycle_stretched(stretched_config: DictConfig) -> None:
    from pathlib import Path

    from hydra import compose
    from hydra import initialize
    from omegaconf import OmegaConf

    from anemoi.utils.testing import get_test_archive

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_stretched"):

        template = compose(
            config_name="stretched",
        )  # apply architecture overrides to template since they override a default
        use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_stretched.yaml")
        testing_modifications = OmegaConf.load(
            Path.cwd() / "training/tests/integration/config/testing_modifications.yaml",
        )

    name_dataset = use_case_modifications.hardware.files.dataset
    url_dataset = "anemoi-integration-tests/regional-use-cases/" + name_dataset + ".tgz"
    tmp_path_dataset = get_test_archive(url_dataset)

    name_forcing_dataset = use_case_modifications.hardware.files.forcing_dataset
    url_forcing_dataset = os.path.join("anemoi-integration-tests/regional-use-cases/", name_forcing_dataset + ".tgz")
    tmp_path_forcing_dataset = get_test_archive(url_forcing_dataset)

    tmp_dir = os.path.commonprefix([tmp_path_dataset, tmp_path_forcing_dataset])

    use_case_modifications.hardware.paths.data = tmp_dir
    use_case_modifications.hardware.files.dataset = os.path.join(os.path.basename(tmp_path_dataset), name_dataset)
    use_case_modifications.hardware.files.forcing_dataset = os.path.join(
        os.path.basename(tmp_path_forcing_dataset), name_forcing_dataset,
    )

    cfg = OmegaConf.merge(template, testing_modifications, use_case_modifications)
    OmegaConf.resolve(cfg)

    AnemoiTrainer(cfg).train()
    shutil.rmtree(cfg.hardware.paths.output)
    shutil.rmtree(cfg.hardware.paths.data)


def test_config_validation_stretched(stretched_config: DictConfig) -> None:
    BaseSchema(**stretched_config)


# @pytest.mark.longtests
# def test_training_cycle_lam(lam_config: DictConfig) -> None:
#     AnemoiTrainer(lam_config).train()
#     shutil.rmtree(lam_config.hardware.paths.output)


# def test_config_validation_lam(lam_config: DictConfig) -> None:
#     BaseSchema(**lam_config)


if __name__ == "__main__":
    from pathlib import Path

    from hydra import compose
    from hydra import initialize
    from omegaconf import OmegaConf

    from anemoi.utils.testing import get_test_archive

    with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_stretched"):

        template = compose(
            config_name="stretched",
        )  # apply architecture overrides to template since they override a default
        use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_stretched.yaml")
        testing_modifications = OmegaConf.load(
            Path.cwd() / "training/tests/integration/config/testing_modifications.yaml",
        )

        name_dataset = use_case_modifications.hardware.files.dataset
        url_dataset = "anemoi-integration-tests/regional-use-cases/" + name_dataset + ".tgz"
        tmp_path_dataset = get_test_archive(url_dataset)

        name_forcing_dataset = use_case_modifications.hardware.files.forcing_dataset
        url_forcing_dataset = os.path.join(
            "anemoi-integration-tests/regional-use-cases/", name_forcing_dataset + ".tgz",
        )
        tmp_path_forcing_dataset = get_test_archive(url_forcing_dataset)

        tmp_dir = os.path.commonprefix([tmp_path_dataset, tmp_path_forcing_dataset])

        use_case_modifications.hardware.paths.data = tmp_dir
        use_case_modifications.hardware.files.dataset = os.path.join(os.path.basename(tmp_path_dataset), name_dataset)
        use_case_modifications.hardware.files.forcing_dataset = os.path.join(
            os.path.basename(tmp_path_forcing_dataset), name_forcing_dataset,
        )

        cfg = OmegaConf.merge(template, testing_modifications, use_case_modifications)
        OmegaConf.resolve(cfg)

        AnemoiTrainer(cfg).train()
        shutil.rmtree(cfg.hardware.paths.output)
        shutil.rmtree(cfg.hardware.paths.data)
