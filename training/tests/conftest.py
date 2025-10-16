# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from torch_geometric.data import HeteroData


def _get_config_path() -> str:
    """Get the config path relative to the project root, working from any directory."""
    # Find the config directory by looking for src/anemoi/training/config
    # This works whether running from training/ or training/tests/
    current = Path.cwd()

    # Try from current directory first (running from training/)
    config_path = current / "src" / "anemoi" / "training" / "config"
    if config_path.exists():
        return str(config_path)

    # Try from parent directory (running from training/tests/)
    config_path = current.parent / "src" / "anemoi" / "training" / "config"
    if config_path.exists():
        return str(config_path)

    # Fallback: use relative path from tests/ directory
    return "../src/anemoi/training/config"

pytest_plugins = "anemoi.utils.testing"

PYTEST_MARKED_TESTS = [
    "multigpu",
    "mlflow",
]


@pytest.fixture
def config(request: SubRequest) -> DictConfig:
    overrides = request.param
    config_path = _get_config_path()
    with initialize(version_base=None, config_path=config_path):
        # config is relative to a module
        return compose(config_name="debug", overrides=overrides)


@pytest.fixture
def datamodule():  # type: ignore[no-untyped-def]
    """Lazy-load AnemoiDatasetsDataModule to avoid expensive import at test collection time."""
    from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

    config_path = _get_config_path()
    with initialize(version_base=None, config_path=config_path):
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
    parser.addoption(
        "--mlflow",
        action="store_true",
        dest="mlflow",
        default=False,
        help="enable tests marked as requiring MLFlow test server",
    )
    parser.addoption(
        "--mlflow-server",
        dest="mlflow_server",
        default=None,
        help="MLFlow server for tests requiring MLFlow (only if --mlflow is set)",
    )


@pytest.fixture
def mlflow_server(pytestconfig: Any) -> str:
    mlflow_server = pytestconfig.getoption("mlflow_server")
    if pytestconfig.getoption("mlflow") and mlflow_server is None:
        e = ValueError("MLFlow server must be provided via --mlflow-server when using --mlflow")
        raise e
    return mlflow_server


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically skip PYTEST_MARKED_TESTS unless options are used in CLI."""
    for option_name in PYTEST_MARKED_TESTS:
        if not config.getoption(f"--{option_name}"):
            skip_marker = pytest.mark.skip(
                reason=f"Skipping tests requiring {option_name}, use --{option_name} to enable",
            )
            for item in items:
                if item.get_closest_marker(option_name):
                    item.add_marker(skip_marker)
