# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
from _pytest.fixtures import SubRequest
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.training.builders.grid_indices import build_grid_indices_from_config
from anemoi.training.data.datamodule import AnemoiDatasetsDataModule


@pytest.fixture
def config(request: SubRequest) -> DictConfig:
    overrides = request.param
    with initialize(version_base=None, config_path="../../src/anemoi/training/config"):
        # config is relative to a module
        return compose(config_name="debug", overrides=overrides)


@pytest.fixture
def datamodule() -> AnemoiDatasetsDataModule:
    with initialize(version_base=None, config_path="../../src/anemoi/training/config"):
        # config is relative to a module
        cfg = compose(config_name="config")
    graph = HeteroData()
    graph["data"].num_nodes = 1
    grid_indices = build_grid_indices_from_config(cfg, graph_data={"data": graph})
    return AnemoiDatasetsDataModule(cfg, {"data": graph}, grid_indices)
