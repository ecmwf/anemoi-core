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

from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.loggers.mlflow import _convert_params
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger
from anemoi.training.diagnostics.mlflow.utils import FixedLengthSet
from anemoi.utils.mlflow.auth import NoAuth

LOGGER = logging.getLogger(__name__)
MAX_PARAMS_LENGTH = 2000


class AzureIdentity(StrEnum):
    USER = "user_identity"
    MANAGED = "managed"
    DEFAULT = "default"


def get_azure_workspace(
    auth_type: str,
    aml_subscription_id: str | None = None,
    aml_resource_group: str | None = None,
    aml_workspace_name: str | None = None,
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
    sub = aml_subscription_id or os.getenv("AZUREML_ARM_SUBSCRIPTION")
    rg = aml_resource_group or os.getenv("AZUREML_ARM_RESOURCEGROUP")
    wsname = aml_workspace_name or os.getenv("AZUREML_ARM_WORKSPACE_NAME")

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
            "Try explicitly setting your subscription details via `diagnostics.mlflow.aml_subscription_id`,"
            "`diagnostics.mlflow.aml_resource_group`, `diagnostics.mlflow.aml_workspace_name`."
        )
        raise ValueError(msg)
    LOGGER.info("Attempting to get Workspace object...")
    ws = ml_client.workspaces.get(ml_client.workspace_name)
    LOGGER.info("Succeeded getting the current workspace!")
    return ws


class AnemoiAzureMLflowLogger(AnemoiMLflowLogger):
    """A custom MLflow logger that logs terminal output."""

    # By default, Azure sets a different 16 character (or so) run_id as the display name
    # we may as well set this to the run_id that mlflow/anemoi creates so we don't have two
    # of these to deal with
    # However, it has to be done after we've already logged an artifact, otherwise we may get an error
    _display_name_is_run_id = False

    def __init__(
        self,
        aml_identity: AzureIdentity,
        aml_subscription_id: str | None = None,
        aml_resource_group: str | None = None,
        aml_workspace_name: str | None = None,
        experiment_name: str = "lightning_logs",
        project_name: str = "anemoi",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        save_dir: str | None = "./mlruns",
        log_model: Literal["all"] | bool = False,
        prefix: str = "",
        resumed: bool | None = False,
        forked: bool | None = False,
        run_id: str | None = None,
        fork_run_id: str | None = None,
        offline: bool | None = False,
        authentication: bool | None = None,
        log_hyperparams: bool | None = True,
        on_resume_create_child: bool | None = True,
        max_params_length: int | None = MAX_PARAMS_LENGTH,
    ) -> None:
        """Initialize the AnemoiMLflowLogger.

        Parameters
        ----------
        aml_identity: str | None, optional
            Type of authentication to fall back on for accessing the AzureML workspace.
        aml_subscription_id: str | None, optional
            The Azure subscription id
        aml_resource_group: str | None, optional
            Name of the Azure ML resource group
        aml_workspace: str | None, optional
            Name of the Azure ML workspace
        experiment_name : str, optional
            Name of experiment, by default "lightning_logs"
        project_name : str, optional
            Name of the project, by default "anemoi"
        run_name : str | None, optional
            Name of run, by default None
        tracking_uri : str | None, optional
            Tracking URI of server, by default os.getenv("MLFLOW_TRACKING_URI")
        save_dir : str | None, optional
            Directory to save logs to, by default "./mlruns"
        log_model : Literal[True, False, "all"], optional
            Log model checkpoints to server (expensive), by default False
        prefix : str, optional
            Prefix for experiments, by default ""
        resumed : bool | None, optional
            Whether the run was resumed or not, by default False
        forked : bool | None, optional
            Whether the run was forked or not, by default False
        run_id : str | None, optional
            Run id of current run, by default None
        fork_run_id : str | None, optional
            Fork Run id from parent run, by default None
        offline : bool | None, optional
            Whether to run offline or not, by default False
        authentication : bool | None, optional
            Whether to authenticate with server or not, by default None
        log_hyperparams : bool | None, optional
            Whether to log hyperparameters, by default True
        on_resume_create_child: bool | None, optional
            Whether to create a child run when resuming a run, by default False
        max_params_length: int | None, optional
            Maximum number of params to be logged to Mlflow
        """
        import mlflow

        self._resumed = resumed
        self._forked = forked
        self._flag_log_hparams = log_hyperparams
        self._max_params_length = max_params_length

        self._fork_run_server2server = None
        self._parent_run_server2server = None
        self._parent_dry_run = False

        # <-- Azure specific stuff -->
        # we don't need authenticate, this just lets us easily subclass the logger
        self.auth = NoAuth()

        # Azure ML jobs (should) set this for us:
        tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

        # fall back to subscription-based method if not
        if not tracking_uri:
            LOGGER.warning(
                "Could not retrieve Azure MLFlow uri automatically;trying to retrieve from subscription...",
            )
            tracking_uri = get_azure_workspace(
                aml_identity or "default",
                aml_subscription_id,
                aml_resource_group,
                aml_workspace_name,
            ).mlflow_tracking_uri

        mlflow.set_tracking_uri(tracking_uri)

        if offline:
            msg = (
                "Logging with AzureML and saving offline simultaneously is currently unsupported."
                "If you need this functionality, please open an issue on the `anemoi-core` repo."
            )
            raise ValueError(msg)
        # <--/ End of Azure specific stuff -->
        if (rank_zero_only.rank == 0) and offline:
            LOGGER.info("MLflow is logging offline.")

        run_id = run_id or os.getenv("MLFLOW_RUN_ID")
        run_id, run_name, tags = self._get_mlflow_run_params(
            project_name=project_name,
            run_name=run_name,
            config_run_id=run_id,
            fork_run_id=fork_run_id,
            tracking_uri=tracking_uri,
            on_resume_create_child=on_resume_create_child,
        )
        # Before creating the run we need to overwrite the tracking_uri and save_dir if offline
        if offline:
            # OFFLINE - When we run offline we can pass a save_dir pointing to a local path
            tracking_uri = None

        else:
            # ONLINE - When we pass a tracking_uri to mlflow then it will ignore the
            # saving dir and save all artifacts/metrics to the remote server database
            save_dir = None

        # Track logged metrics to prevent duplicate logs
        # 2000 has been chosen as this should contain metrics form many steps
        self._logged_metrics = FixedLengthSet(maxlen=2000)  # Track (key, step)

        MLFlowLogger.__init__(
            self,
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=save_dir,
            log_model=log_model,
            prefix=prefix,
            run_id=run_id,
        )

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any] | Namespace, *, expand_keys: list[str] | None = None) -> None:
        super().log_hyperparams(params=params, expand_keys=expand_keys)
        # if not self._display_name_is_run_id:
        #     # now set Azure display name to be equal to the run name anemoi sees
        #     # this apparently should happen after the logger is initialized
        #     from azureml.core import Run as AzureMLRun
        #     from azureml.core import Workspace
        #     from azureml.core.authentication import ServicePrincipalAuthentication
        #
        #     sp_auth = ServicePrincipalAuthentication(
        #         tenant_id=os.environ["AZURE_TENANT_ID"],
        #         service_principal_id=os.environ["AZURE_CLIENT_ID"],
        #         service_principal_password=os.environ["AZURE_CLIENT_SECRET"],
        #     )
        #     ws = Workspace(
        #         subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        #         resource_group=self.aml_resource_group,
        #         workspace_name=self.aml_workspace_name,
        #         auth=sp_auth,
        #     )
        #     aml_run = AzureMLRun.get(ws, run_id=self.run_id)
        #     aml_run.display_name = self.run_id
        #     aml_run.flush()
        #     self._display_name_is_run_id = True

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
