# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--longtests",
        action="store_true",
        dest="longtests",
        default=False,
        help="enable tests marked as longtests",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the 'longtests' marker to avoid warnings."""
    config.addinivalue_line("markers", "longtests: mark tests as long-running")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically skip @pytest.mark.longtests tests unless --longtests is used."""
    if not config.getoption("--longtests"):
        skip_marker = pytest.mark.skip(reason="Skipping long test, use --longtests to enable")
        for item in items:
            if item.get_closest_marker("longtests"):
                item.add_marker(skip_marker)
