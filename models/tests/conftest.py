# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--distributed",
        action="store_true",
        default=False,
        help="enable tests marked as requiring torch.distributed processes",
    )
    parser.addoption(
        "--distributed-backend",
        choices=("gloo", "nccl", "all"),
        default="gloo",
        help="backend for distributed tests when --distributed is set",
    )
    parser.addoption(
        "--distributed-world-size",
        type=int,
        default=2,
        help="world size for distributed tests when --distributed is set",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--distributed"):
        _skip_distributed_tests(items)


def _skip_distributed_tests(items: list[pytest.Item]) -> None:
    skip_distributed = pytest.mark.skip(reason="Skipping distributed tests, use --distributed to enable")
    for item in items:
        if item.get_closest_marker("distributed"):
            item.add_marker(skip_distributed)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "distributed_backend" in metafunc.fixturenames:
        _parametrize_distributed_backend(metafunc)


def _parametrize_distributed_backend(metafunc: pytest.Metafunc) -> None:
    backend = metafunc.config.getoption("distributed_backend")
    backends = ["gloo", "nccl"] if backend == "all" else [backend]
    metafunc.parametrize("distributed_backend", backends, indirect=True)


@pytest.fixture
def distributed_backend(request: pytest.FixtureRequest) -> str:
    backend = request.param
    if backend == "nccl" and not torch.cuda.is_available():
        pytest.skip("NCCL backend requested but CUDA is not available.")
    return backend


@pytest.fixture
def distributed_world_size(pytestconfig: pytest.Config) -> int:
    world_size = pytestconfig.getoption("distributed_world_size")
    if world_size < 2:
        msg = f"--distributed-world-size must be >= 2, got {world_size}"
        raise ValueError(msg)
    return world_size


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
