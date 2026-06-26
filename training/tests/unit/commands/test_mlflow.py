from pathlib import Path
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


@pytest.fixture
def mock_sync(mocker: "MockerFixture") -> "MockType":
    mock = mocker.patch("anemoi.training.utils.mlflow_sync.MlFlowSync")
    mocker.patch("anemoi.utils.mlflow.utils.health_check")
    return mock


def test_sync_sqlite_source(mocker: "MockerFixture", mock_sync: "MockType", tmp_path: Path) -> None:
    (tmp_path / "mlflow.db").touch()
    cli_testing(
        "anemoi-training", "mlflow", "sync",
        "-s", str(tmp_path),
        "-d", "http://server.com",
        "-r", "abc123",
    )
    source_used = mock_sync.call_args[0][0]
    extra_tags = mock_sync.call_args[1]["extra_tags"]
    assert source_used == f"sqlite:///{tmp_path.resolve() / 'mlflow.db'}"
    assert extra_tags.get("sync.offline_store") == "sqlite"


def test_sync_explicit_sqlite_uri(mocker: "MockerFixture", mock_sync: "MockType", tmp_path: Path) -> None:
    db = tmp_path / "mlflow.db"
    db.touch()
    uri = f"sqlite:///{db}"
    cli_testing(
        "anemoi-training", "mlflow", "sync",
        "-s", uri,
        "-d", "http://server.com",
        "-r", "abc123",
    )
    source_used = mock_sync.call_args[0][0]
    extra_tags = mock_sync.call_args[1]["extra_tags"]
    assert source_used == uri
    assert extra_tags.get("sync.offline_store") == "sqlite"


def test_sync_filesystem_source(mocker: "MockerFixture", mock_sync: "MockType", tmp_path: Path) -> None:
    (tmp_path / "0").mkdir()  # numeric subdir = legacy filesystem store
    cli_testing(
        "anemoi-training", "mlflow", "sync",
        "-s", str(tmp_path),
        "-d", "http://server.com",
        "-r", "abc123",
    )
    source_used = mock_sync.call_args[0][0]
    extra_tags = mock_sync.call_args[1]["extra_tags"]
    assert source_used == str(tmp_path)
    assert extra_tags.get("sync.offline_store") == "filesystem"


def test_sync_invalid_source(mocker: "MockerFixture", mock_sync: "MockType", tmp_path: Path) -> None:
    # Empty directory — neither mlflow.db nor numeric subdirs
    with pytest.raises((SystemExit, ValueError)):
        cli_testing(
            "anemoi-training", "mlflow", "sync",
            "-s", str(tmp_path),
            "-d", "http://server.com",
            "-r", "abc123",
        )
