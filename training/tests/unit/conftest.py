# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.utils.testing import TemporaryDirectoryForTestData


@pytest.fixture
def config(request: SubRequest) -> DictConfig:
    overrides = request.param
    with initialize(version_base=None, config_path="../../src/anemoi/training/config"):
        # config is relative to a module
        return compose(config_name="debug", overrides=overrides)


@pytest.fixture(scope="module")
def dataset_path(temporary_directory_for_test_data: TemporaryDirectoryForTestData) -> str:
    """Get path to test dataset."""
    test_ds = "anemoi-integration-tests/training/datasets/aifs-ea-an-oper-0001-mars-o96-2017-2017-6h-v8-testing.zarr"
    name_dataset = Path(test_ds).name
    url_archive = test_ds + ".tgz"
    tmp_path = temporary_directory_for_test_data(url_archive, archive=True)
    return str(Path(tmp_path, name_dataset))


@pytest.fixture
def datamodule() -> AnemoiDatasetsDataModule:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config"):
        # config is relative to a module
        cfg = compose(config_name="config")
    return AnemoiDatasetsDataModule(cfg)
