# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

pytest_plugins = "anemoi.utils.testing"


@pytest.fixture
def config(request: SubRequest) -> DictConfig:
    overrides = request.param
    with initialize(version_base=None, config_path="../src/anemoi/training/config"):
        # config is relative to a module
        return compose(config_name="debug", overrides=overrides)


@pytest.fixture
def datamodule() -> AnemoiDatasetsDataModule:
    with initialize(version_base=None, config_path="../src/anemoi/training/config"):
        # config is relative to a module
        cfg = compose(config_name="config")
    return AnemoiDatasetsDataModule(cfg)


@pytest.fixture
def graph_with_nodes() -> HeteroData:
    """Graph with 12 nodes."""
    lats = [-0.15, 0, 0.15]
    lons = [0, 0.25, 0.5, 0.75]
    coords = np.array([[lat, lon] for lat in lats for lon in lons])
    graph = HeteroData()
    graph["test_nodes"].x = 2 * torch.pi * torch.tensor(coords)
    graph["test_nodes"].test_attr = (torch.tensor(coords) ** 2).sum(1)
    graph["test_nodes"].mask = torch.tensor([True] * len(coords))
    return graph


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--multigpu",
        action="store_true",
        dest="multigpu",
        default=False,
        help="enable tests marked as requiring multiple GPUs",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the 'multigpu' marker to avoid warnings."""
    config.addinivalue_line("markers", "multigpu: mark tests as requiring multiple GPUs")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically skip @pytest.mark.multigpu tests unless --multigpu is used."""
    if not config.getoption("--multigpu"):
        skip_marker = pytest.mark.skip(reason="Skipping tests requiring multipe GPUs, use --multigpu to enable")
        for item in items:
            if item.get_closest_marker("multigpu"):
                item.add_marker(skip_marker)
