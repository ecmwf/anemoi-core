# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import TYPE_CHECKING

import pytest

from anemoi.utils.testing import cli_testing

if TYPE_CHECKING:
    from pytest_mock import MockerFixture
    from pytest_mock import MockType


@pytest.fixture
def mock_auth(mocker: "MockerFixture") -> "MockType":
    mock_auth = mocker.patch("anemoi.utils.mlflow.auth.TokenAuth")
    mock_auth.get_servers.return_value = [("http://server-2", 2), ("http://server-1", 1)]
    return mock_auth


def test_mlflow_login(mocker: "MockerFixture", mock_auth: "MockType") -> None:
    cli_testing("anemoi-training", "mlflow", "login", "--url", "http://localhost:5000")
    mock_auth.assert_called_once_with(url="http://localhost:5000")
    mock_auth.return_value.login.assert_called_once()
    mock_auth.reset_mock()

    cli_testing("anemoi-training", "mlflow", "login")
    mock_auth.get_servers.assert_called_once()
    mock_auth.assert_called_once_with(url="http://server-2")
    mock_auth.return_value.login.assert_called_once()
    mock_auth.reset_mock()

    cli_testing("anemoi-training", "mlflow", "login", "--list")
    mock_auth.get_servers.assert_called_once()
    mock_auth.return_value.login.assert_not_called()
    mock_auth.reset_mock()

    cli_testing("anemoi-training", "mlflow", "login", "--all")
    mock_auth.get_servers.assert_called_once()
    assert mock_auth.call_args_list == [mocker.call(url="http://server-2"), mocker.call(url="http://server-1")]
    assert mock_auth.return_value.login.call_count == 2
    mock_auth.reset_mock()
