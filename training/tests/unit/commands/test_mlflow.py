# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from omegaconf import OmegaConf

from anemoi.training.checkpoint.sources.run import RunIdSource
from anemoi.training.commands.mlflow import prepare_mlflow_run_id
from anemoi.utils.testing import cli_testing

if TYPE_CHECKING:
    from omegaconf import DictConfig
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


_RUN_SOURCE = "anemoi.training.checkpoint.sources.run.RunIdSource"


class SiteRunSource(RunIdSource):
    """A site-specific subclass whose ``_target_`` does not end in ``RunIdSource``."""


_SITE_RUN_SOURCE = f"{__name__}.SiteRunSource"


def _prepare_config(source: dict | None) -> "DictConfig":
    checkpoint = {"source": source} if source is not None else {}
    return OmegaConf.create(
        {
            "diagnostics": {
                "log": {
                    "mlflow": {
                        "tracking_uri": "http://localhost:5000",
                        "experiment_name": "anemoi-tests",
                        "run_name": "a-run",
                        "expand_hyperparams": [],
                    },
                },
            },
            "training": {"checkpoint": checkpoint} if checkpoint else {},
        },
    )


def _mock_client(mocker: "MockerFixture") -> "MockType":
    client = mocker.MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="exp-7")
    mocker.patch("anemoi.utils.mlflow.client.AnemoiMlflowClient", return_value=client)
    return client


def test_prepare_mlflow_run_id_returns_the_run_it_attached_to(mocker: "MockerFixture") -> None:
    """Attaching to an existing run returns that run, not None.

    The caller unpacks two values and writes them to the metadata file the launch
    script reads, so returning None raised ``TypeError: cannot unpack non-sequence
    NoneType object`` and left that file unwritten.
    """
    client = _mock_client(mocker)
    config = _prepare_config({"_target_": _RUN_SOURCE, "run_id": "run-abc", "fork": False})

    run_id, experiment_id = prepare_mlflow_run_id(config=config)

    assert (run_id, experiment_id) == ("run-abc", "exp-7")
    client.get_run.assert_called_once_with("run-abc")
    # Attaching must not mint a second run.
    client.create_run.assert_not_called()


def test_prepare_mlflow_run_id_forking_mints_a_new_run(mocker: "MockerFixture") -> None:
    """A fork starts a new run, so it falls through to creating one.

    Positive control for the test above: proves the attach path is selected by the
    source's ``fork`` flag rather than being taken unconditionally.
    """
    client = _mock_client(mocker)
    client.create_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run-new"))
    config = _prepare_config({"_target_": _RUN_SOURCE, "run_id": "run-abc", "fork": True})

    run_id, experiment_id = prepare_mlflow_run_id(config=config)

    assert (run_id, experiment_id) == ("run-new", "exp-7")
    client.create_run.assert_called_once()


def test_prepare_mlflow_run_id_attaches_through_a_run_source_subclass(mocker: "MockerFixture") -> None:
    """Run identity comes from the source class, so a subclass attaches like its parent.

    A match on the end of the ``_target_`` string treated ``SiteRunSource`` as an
    ordinary source and minted a fresh run for a config that named an existing one.
    """
    client = _mock_client(mocker)
    config = _prepare_config({"_target_": _SITE_RUN_SOURCE, "run_id": "run-abc", "fork": False})

    run_id, experiment_id = prepare_mlflow_run_id(config=config)

    assert (run_id, experiment_id) == ("run-abc", "exp-7")
    client.get_run.assert_called_once_with("run-abc")
    client.create_run.assert_not_called()
