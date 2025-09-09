# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

if TYPE_CHECKING:
    from argparse import Namespace

    from mlflow.tracking import MlflowClient

try:
    if TYPE_CHECKING:
        from azure.ai.ml.entities import Workspace

    from azure.ai.ml import MLClient
    from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
    from azure.identity import DefaultAzureCredential
    from azure.identity import ManagedIdentityCredential

    # NOTE: Lightweight dependency for Python 3.10.
    # Replace with `from enum import StrEnum` when deprecating 3.10.
    from strenum import StrEnum

except ModuleNotFoundError as e:
    msg = (
        "Use of MLFlow logging in Azure requires the modules `azure-ai-ml` and `azure-identity`. You can install these"
        "via the Azure optional extra in Anemoi training: `pip install anemoi-training[azure]`."
    )
    raise ModuleNotFoundError(msg) from e

from pytorch_lightning.loggers.mlflow import _convert_params

from anemoi.training.diagnostics.mlflow import LOG_MODEL
from anemoi.training.diagnostics.mlflow import MAX_PARAMS_LENGTH
from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger
from anemoi.utils.mlflow.auth import NoAuth

LOGGER = logging.getLogger(__name__)


class AzureIdentity(StrEnum):
    USER = "user_identity"
    MANAGED = "managed"
    DEFAULT = "default"


def get_azure_workspace(
    auth_type: str,
    subscription_id: str | None = None,
    resource_group: str | None = None,
    workspace_name: str | None = None,
) -> Workspace:
    # validate auth_type
    try:
        azure_id_type = AzureIdentity(auth_type.strip().casefold())
    except ValueError as e:
        valid_options = ", ".join(x.value for x in AzureIdentity)
        msg = f"AzureML auth type needs to be one of: {valid_options}. Recieved {auth_type}"
        raise ValueError(msg) from e

    # these env variables are usually attatched to azure ml jobs,
    # so use them if they exist.
    sub = subscription_id or os.getenv("AZUREML_ARM_SUBSCRIPTION")
    rg = resource_group or os.getenv("AZUREML_ARM_RESOURCEGROUP")
    wsname = workspace_name or os.getenv("AZUREML_ARM_WORKSPACE_NAME")

    if sub and rg and wsname:
        LOGGER.info("Attempting Azure authentication with the following details:")
        LOGGER.info("Subscription: %s", sub)
        LOGGER.info("Resource group: %s", rg)
        LOGGER.info("Workspace: %s", wsname)

        match azure_id_type:
            case AzureIdentity.MANAGED:
                client_id = os.getenv("DEFAULT_IDENTITY_CLIENT_ID")
                credential = (
                    ManagedIdentityCredential(client_id=client_id) if client_id else ManagedIdentityCredential()
                )
            case AzureIdentity.USER:
                credential = AzureMLOnBehalfOfCredential()
            case AzureIdentity.DEFAULT:
                credential = DefaultAzureCredential()

        ml_client = MLClient(
            credential,
            subscription_id=sub,
            resource_group_name=rg,
            workspace_name=wsname,
        )
        LOGGER.info("Successfully authenticated with Azure.")
    else:
        msg = (
            "Azure environment incorrectly configured; tried to use \n  "
            f"- subscription: {sub}\n  - resource_group: {rg}\n  - workspace: {wsname}.\n"
            "Try explicitly setting your subscription details via `diagnostics.mlflow.subscription_id`,"
            "`diagnostics.mlflow.resource_group`, `diagnostics.mlflow.workspace_name`."
        )
        raise ValueError(msg)
    LOGGER.info("Attempting to get Workspace object...")
    ws = ml_client.workspaces.get(ml_client.workspace_name)
    LOGGER.info("Succeeded getting the current workspace!")
    return ws


class AnemoiAzureMLflowLogger(AnemoiMLflowLogger):
    """A custom MLflow logger that logs terminal output."""

    def __init__(
        self,
        identity: AzureIdentity,
        subscription_id: str | None = None,
        resource_group: str | None = None,
        workspace_name: str | None = None,
        azure_log_level: str = "WARNING",
        experiment_name: str = "lightning_logs",
        project_name: str = "anemoi",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        save_dir: str | None = "./mlruns",
        log_model: Literal["all"] | bool = LOG_MODEL,
        prefix: str = "",
        run_id: str | None = None,
        fork_run_id: str | None = None,
        log_hyperparams: bool | None = True,
        on_resume_create_child: bool | None = True,
        max_params_length: int | None = MAX_PARAMS_LENGTH,
        http_max_retries: int | None = 35,
    ) -> None:
        """Initialize the AnemoiAzureMLflowLogger.

        Parameters
        ----------
        identity: str | None, optional
            Type of authentication to fall back on for accessing the AzureML workspace.
        subscription_id: str | None, optional
            The Azure subscription id
        resource_group: str | None, optional
            Name of the Azure ML resource group
        workspace: str | None, optional
            Name of the Azure ML workspace
        azure_log_level: str, optional
            Log level for all azure packages (azure-identity, azure-core, etc)
        """
        # Set azure logging to warning, since otherwise it's way too much
        azure_logger = logging.getLogger("azure")
        numeric_level = getattr(logging, azure_log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {azure_log_level.upper()}")
        azure_logger.setLevel(numeric_level)

        # Azure ML jobs (should) set this for us:
        tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

        # fall back to subscription-based method if not
        if not tracking_uri:
            LOGGER.warning(
                "Could not retrieve Azure MLFlow uri automatically; trying to retrieve from subscription...",
            )
            tracking_uri = get_azure_workspace(
                identity or "default",
                subscription_id,
                resource_group,
                workspace_name,
            ).mlflow_tracking_uri

        # Azure sets the mlflow run id when the user runs az ml job create,
        # so unless we are forking/resuming etc, we want to grab this environment variable
        run_id = run_id or os.getenv("MLFLOW_RUN_ID")

        super().__init__(
            experiment_name=experiment_name,
            project_name=project_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            save_dir=None,
            log_model=log_model,
            prefix=prefix,
            run_id=run_id,
            fork_run_id=fork_run_id,
            offline=False,
            authentication=False,
            log_hyperparams=log_hyperparams,
            on_resume_create_child=on_resume_create_child,
            max_params_length=max_params_length,
        )

    def _init_authentication(
        self,
        tracking_uri: str,
        authentication: bool | None,
        offline: bool,
    ) -> None:
        """No need to authenticate with Azure ML flavor of MLFlow logger"""
        self.auth = NoAuth()

    @staticmethod
    def log_hyperparams_in_mlflow(
        client: MlflowClient,
        run_id: str,
        params: dict[str, Any] | Namespace,
        *,
        expand_keys: list[str] | None = None,
        log_hyperparams: bool | None = True,
        clean_params: bool = True,
        max_params_length: int | None = MAX_PARAMS_LENGTH,
    ) -> None:
        """Log hyperparameters to MLflow server.

        - flatten config params using '.'.
        - expand keys within params to avoid truncation.
        - log hyperparameters as an artifact.

        Parameters
        ----------
        client : MlflowClient
            MLflow client.
        run_id : str
            Run ID.
        params : dict[str, Any] | Namespace
            params to log.
        expand_keys : list[str] | None, optional
            keys to expand within params. Any key being expanded will
            have lists converted according to `expand_iterables`,
            by default None.
        log_hyperparams : bool | None, optional
            Whether to log hyperparameters, by default True.
        max_params_length: int | None, optional
            Maximum number of params to be logged to Mlflow
        """
        if log_hyperparams:
            params = _convert_params(params)

            # this is needed to resolve optional missing config values to a string, instead of raising a missing error
            if config := params.get("config"):
                params["config"] = config.model_dump(by_alias=True)

            import mlflow

            try:  # Check maximum param value length is available and use it
                truncation_length = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
            except AttributeError:  # Fallback (in case of MAX_PARAM_VAL_LENGTH not available)
                truncation_length = 250  # Historical default value

            AnemoiAzureMLflowLogger.log_hyperparams_as_mlflow_artifact(client=client, run_id=run_id, params=params)

    @staticmethod
    def log_hyperparams_as_mlflow_artifact(
        client: MlflowClient,
        run_id: str,
        params: dict[str, Any] | Namespace,
    ) -> None:
        """Log hyperparameters as an artifact."""
        import datetime
        import json
        import tempfile
        from json import JSONEncoder

        class StrEncoder(JSONEncoder):
            def default(self, o: Any) -> str:
                return str(o)

        now = str(datetime.datetime.now()).replace(" ", "T")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / f"config.{now}.json"
            with Path.open(path, "w") as f:
                json.dump(params, f, cls=StrEncoder)
            client.log_artifact(run_id=run_id, local_path=path)
