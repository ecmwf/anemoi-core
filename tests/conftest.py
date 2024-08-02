# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from hydra import compose
from hydra import initialize

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule


@pytest.fixture()
def config(request):
    overrides = request.param
    with initialize(version_base=None, config_path="../src/anemoi/training/config"):
        # config is relative to a module
        config = compose(config_name="debug", overrides=overrides)
    return config


@pytest.fixture()
def datamodule():
    with initialize(version_base=None, config_path="../src/anemoi/training/config"):
        # config is relative to a module
        cfg = compose(config_name="config")
    return AnemoiDatasetsDataModule(cfg)
