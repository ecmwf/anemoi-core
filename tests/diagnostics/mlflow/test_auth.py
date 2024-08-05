# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import time

import pytest

from anemoi.training.diagnostics.mlflow.auth import TokenAuth


def mocks(mocker, token_request={}, load_config={}):
    response = {
        "access_token": "access_token",
        "expires_in": 3600,
        "refresh_token": "new_refresh_token",
    }
    response.update(token_request)

    config = {
        "refresh_token": "old_refresh_token",
        "refresh_expires": time.time() + 3600,
    }
    config.update(load_config)

    mock_token_request = mocker.patch(
        "anemoi.training.diagnostics.mlflow.auth.TokenAuth._token_request",
        return_value=response,
    )
    mocker.patch(
        "anemoi.training.diagnostics.mlflow.auth.load_config",
        return_value=config,
    )
    mocker.patch(
        "anemoi.training.diagnostics.mlflow.auth.save_config",
    )
    mocker.patch(
        "anemoi.training.diagnostics.mlflow.auth.input",
        return_value="username",
    )
    mocker.patch(
        "anemoi.training.diagnostics.mlflow.auth.getpass",
        return_value="password",
    )
    mocker.patch("os.environ")

    return mock_token_request


def test_auth(mocker):
    mock_token_request = mocks(mocker)

    auth = TokenAuth("https://test.url")

    assert auth.access_token is None
    assert auth.refresh_token == "old_refresh_token"

    auth.authenticate()
    # test that no new token is requested the second time
    auth.authenticate()

    mock_token_request.assert_called_once()

    assert auth.access_token == "access_token"
    assert auth.access_expires > time.time()
    assert auth.refresh_token == "new_refresh_token"


def test_not_logged_in(mocker):
    # no refresh token
    mocks(mocker, load_config={"refresh_token": None})
    auth = TokenAuth("https://test.url")
    pytest.raises(RuntimeError, auth.authenticate)

    # expired refresh token
    mocks(mocker, load_config={"refresh_expires": time.time() - 1})
    auth = TokenAuth("https://test.url")
    pytest.raises(RuntimeError, auth.authenticate)


def test_login(mocker):
    # normal login
    mock_token_request = mocks(mocker)
    auth = TokenAuth("https://test.url")
    auth.login()

    mock_token_request.assert_called_once()

    # normal credential login
    mock_token_request = mocks(mocker, load_config={"refresh_token": None})
    auth = TokenAuth("https://test.url")
    auth.login()

    mock_token_request.assert_called_once_with(username="username", password="password")

    # forced credential login
    mock_token_request = mocks(mocker)
    auth = TokenAuth("https://test.url")
    auth.login(force_credentials=True)

    mock_token_request.assert_called_once_with(username="username", password="password")

    # failed login
    mock_token_request = mocks(mocker, token_request={"refresh_token": None})
    auth = TokenAuth("https://test.url")
    pytest.raises(RuntimeError, auth.login)

    mock_token_request.assert_called_with(username="username", password="password")
    assert mock_token_request.call_count == 2


def test_enabled(mocker):
    mock_token_request = mocks(mocker)
    auth = TokenAuth("https://test.url", enabled=False)
    auth.authenticate()

    mock_token_request.assert_not_called()


def test_api(mocker):
    mocks(mocker)
    auth = TokenAuth("https://test.url")
    mock_post = mocker.patch("requests.post")

    # successful request
    response_json = {"status": "OK", "response": {}}
    mocker.patch("requests.post.return_value.json", return_value=response_json)
    response = auth._request("path", {"key": "value"})

    assert response == response_json["response"]
    mock_post.assert_called_once_with(
        "https://test.url/path",
        json={"key": "value"},
        headers=mocker.ANY,
        timeout=mocker.ANY,
    )

    # api error
    error_response = {"status": "ERROR", "response": {}}
    mocker.patch("requests.post.return_value.json", return_value=error_response)

    with pytest.raises(RuntimeError):
        auth._request("path", {"key": "value"})
