# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import os
from pathlib import Path
from typing import Callable

import pytest
from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf

from anemoi.utils.testing import get_test_archive


@pytest.fixture(autouse=True)
def set_working_directory() -> None:
    """Automatically set the working directory to the repo root."""
    repo_root = Path(__file__).resolve().parent
    while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent

    os.chdir(repo_root)


@pytest.fixture
def testing_modifications_with_temp_dir(tmp_path: Path) -> OmegaConf:
    testing_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/testing_modifications.yaml")
    temp_dir = str(tmp_path)
    testing_modifications.hardware.paths.output = temp_dir
    return testing_modifications


@pytest.fixture(
    params=[
        ["model=gnn"],
        ["model=graphtransformer"],
    ],
)
def architecture_config(
    request: pytest.FixtureRequest,
    testing_modifications_with_temp_dir: OmegaConf,
) -> Callable[[bool], OmegaConf]:
    overrides = request.param

    def _create_config(download_data: bool = False) -> Callable[[bool], OmegaConf]:
        with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_config"):
            template = compose(
                config_name="config",
                overrides=overrides,
            )  # apply architecture overrides to template since they override a default

        use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_config.yaml")

        if download_data:
            tmp_dir, dataset = _download_datasets(use_case_modifications)
            use_case_modifications.hardware.paths.data = tmp_dir
            use_case_modifications.hardware.files.dataset = dataset

        cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
        OmegaConf.resolve(cfg)
        return cfg

    return _create_config


@pytest.fixture
def stretched_config(testing_modifications_with_temp_dir: OmegaConf) -> Callable[[bool], OmegaConf]:
    def _create_config(download_data: bool = False) -> Callable[[bool], OmegaConf]:
        with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_stretched"):
            template = compose(config_name="stretched")

        use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_stretched.yaml")

        if download_data:
            tmp_dir, dataset, forcing_dataset = _download_datasets(use_case_modifications, forcing_dataset=True)
            use_case_modifications.hardware.paths.data = tmp_dir
            use_case_modifications.hardware.files.dataset = dataset
            use_case_modifications.hardware.files.forcing_dataset = forcing_dataset

        cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
        OmegaConf.resolve(cfg)
        return cfg

    return _create_config


@pytest.fixture
def lam_config(testing_modifications_with_temp_dir: OmegaConf) -> Callable[[bool], OmegaConf]:
    def _create_config(download_data: bool = False) -> Callable[[bool], OmegaConf]:
        with initialize(version_base=None, config_path="../../src/anemoi/training/config", job_name="test_lam"):
            template = compose(config_name="lam")

        use_case_modifications = OmegaConf.load(Path.cwd() / "training/tests/integration/config/test_lam.yaml")

        if download_data:
            tmp_dir, dataset, forcing_dataset = _download_datasets(use_case_modifications, forcing_dataset=True)
            use_case_modifications.hardware.paths.data = tmp_dir
            use_case_modifications.hardware.files.dataset = dataset
            use_case_modifications.hardware.files.forcing_dataset = forcing_dataset

        cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
        OmegaConf.resolve(cfg)
        return cfg

    return _create_config


def _download_datasets(config: OmegaConf, forcing_dataset: bool = False) -> list[str]:
    url_dataset = config.hardware.files.dataset + ".tgz"
    name_dataset = Path(config.hardware.files.dataset).name
    tmp_path_dataset = get_test_archive(url_dataset)

    if forcing_dataset:
        url_forcing_dataset = config.hardware.files.forcing_dataset + ".tgz"
        name_forcing_dataset = Path(config.hardware.files.forcing_dataset).name
        tmp_path_forcing_dataset = get_test_archive(url_forcing_dataset)

        tmp_dir = os.path.commonprefix([tmp_path_dataset, tmp_path_forcing_dataset])[:-1]  # remove trailing slash
        rel_path_dataset = Path(tmp_path_dataset).name + "/" + name_dataset
        rel_path_forcing_dataset = Path(tmp_path_forcing_dataset).name + "/" + name_forcing_dataset

        return tmp_dir, rel_path_dataset, rel_path_forcing_dataset

    return tmp_path_dataset, name_dataset
