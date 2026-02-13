# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_mlflow_temp_log_file_with_tmpdir(mocker: "MockerFixture", tmp_path: Path) -> None:
    """Test that mlflow_temp_log_file context manager works when TMPDIR is set."""
    from anemoi.training.utils.mlflow_sync import mlflow_temp_log_file

    test_user = "testuser"
    mocker.patch.dict(os.environ, {"TMPDIR": str(tmp_path), "USER": test_user}, clear=False)

    with mlflow_temp_log_file() as temp_file:
        # Verify it returns a temp file
        assert temp_file is not None
        assert hasattr(temp_file, "name")
        assert Path(temp_file.name).exists()

        # Verify environment variables were set
        assert os.environ["MLFLOW_EXPORT_IMPORT_LOG_OUTPUT_FILE"] == temp_file.name
        assert os.environ["MLFLOW_EXPORT_IMPORT_TMP_DIRECTORY"] == str(tmp_path)

        # Verify the temp file is in the correct directory
        assert temp_file.name.startswith(str(tmp_path))
        assert test_user in temp_file.name

    # After exiting context, verify cleanup happened
    assert "MLFLOW_EXPORT_IMPORT_LOG_OUTPUT_FILE" not in os.environ
    assert "MLFLOW_EXPORT_IMPORT_TMP_DIRECTORY" not in os.environ


def test_mlflow_temp_log_file_with_scratch(mocker: "MockerFixture", tmp_path: Path) -> None:
    """Test that mlflow_temp_log_file works when SCRATCH is set (TMPDIR not set)."""
    from anemoi.training.utils.mlflow_sync import mlflow_temp_log_file

    test_user = "testuser"
    # Clear TMPDIR, only set SCRATCH
    env_vars = {"SCRATCH": str(tmp_path), "USER": test_user}
    if "TMPDIR" in os.environ:
        del os.environ["TMPDIR"]
    mocker.patch.dict(os.environ, env_vars, clear=False)

    with mlflow_temp_log_file() as temp_file:
        assert temp_file is not None
        assert Path(temp_file.name).exists()
        assert temp_file.name.startswith(str(tmp_path))

    # Verify cleanup
    assert "MLFLOW_EXPORT_IMPORT_LOG_OUTPUT_FILE" not in os.environ
    assert "MLFLOW_EXPORT_IMPORT_TMP_DIRECTORY" not in os.environ


def test_mlflow_temp_log_file_missing_env_vars_raises_error(mocker: "MockerFixture") -> None:
    """Test that mlflow_temp_log_file raises ValueError when env vars are missing.

    This test validates the fix for issue #622 where the error message itself
    would throw a KeyError when trying to access missing environment variables.
    """
    from anemoi.training.utils.mlflow_sync import mlflow_temp_log_file

    # Mock both environment variables as not set
    mocker.patch("os.getenv", side_effect=lambda k, d=None: None if k in ["TMPDIR", "SCRATCH"] else os.getenv(k, d))

    # Should raise ValueError with a proper error message (not KeyError)
    with pytest.raises(ValueError, match="Please set one of those variables"), mlflow_temp_log_file():
        pass


def test_mlflow_temp_log_file_cleanup_on_exception(mocker: "MockerFixture", tmp_path: Path) -> None:
    """Test that context manager cleans up even when exception occurs."""
    from anemoi.training.utils.mlflow_sync import mlflow_temp_log_file

    test_user = "testuser"
    mocker.patch.dict(os.environ, {"TMPDIR": str(tmp_path), "USER": test_user}, clear=False)

    with pytest.raises(RuntimeError, match="Test exception"), mlflow_temp_log_file() as temp_file:
        # Verify environment variables are set
        assert os.environ["MLFLOW_EXPORT_IMPORT_LOG_OUTPUT_FILE"] == temp_file.name
        # Raise an exception to test cleanup
        raise RuntimeError("Test exception")  # noqa: EM101, TRY003

    # Even after exception, cleanup should have happened
    assert "MLFLOW_EXPORT_IMPORT_LOG_OUTPUT_FILE" not in os.environ
    assert "MLFLOW_EXPORT_IMPORT_TMP_DIRECTORY" not in os.environ


def test_cleanup_temp_env_vars() -> None:
    """Test that _cleanup_temp_env_vars removes environment variables."""
    from anemoi.training.utils.mlflow_sync import _cleanup_temp_env_vars

    # Set up environment variables
    os.environ["MLFLOW_EXPORT_IMPORT_LOG_OUTPUT_FILE"] = "/tmp/test"  # noqa: S108
    os.environ["MLFLOW_EXPORT_IMPORT_TMP_DIRECTORY"] = "/tmp"  # noqa: S108

    # Call cleanup
    _cleanup_temp_env_vars()

    # Verify they were removed
    assert "MLFLOW_EXPORT_IMPORT_LOG_OUTPUT_FILE" not in os.environ
    assert "MLFLOW_EXPORT_IMPORT_TMP_DIRECTORY" not in os.environ


def test_close_and_clean_temp_with_server2server(tmp_path: Path) -> None:
    """Test that close_and_clean_temp removes artifacts when server2server=True."""
    from anemoi.training.utils.mlflow_sync import close_and_clean_temp

    # Create an artifact path
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    (artifact_path / "test.txt").write_text("test")

    # Call close_and_clean_temp with server2server=True
    close_and_clean_temp(server2server=True, artifact_path=artifact_path)

    # Verify artifact path was removed (server2server=True)
    assert not artifact_path.exists()


def test_close_and_clean_temp_no_artifact_removal(tmp_path: Path) -> None:
    """Test that close_and_clean_temp doesn't remove artifacts when server2server=False."""
    from anemoi.training.utils.mlflow_sync import close_and_clean_temp

    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()

    # Call with server2server=False
    close_and_clean_temp(server2server=False, artifact_path=artifact_path)

    # Verify artifact path still exists (server2server=False)
    assert artifact_path.exists()
