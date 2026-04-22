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

PYTEST_MARKED_TESTS = [
    "multigpu",
]


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--multigpu",
        action="store_true",
        dest="multigpu",
        default=False,
        help="enable tests marked as requiring multiple GPUs",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically skip marked tests unless options are used in CLI."""
    for option_name in PYTEST_MARKED_TESTS:
        if not config.getoption(f"--{option_name}"):
            skip_marker = pytest.mark.skip(
                reason=f"Skipping tests requiring {option_name}, use --{option_name} to enable",
            )
            for item in items:
                if item.get_closest_marker(option_name):
                    item.add_marker(skip_marker)
