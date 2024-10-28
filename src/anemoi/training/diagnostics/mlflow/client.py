# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from typing import Any

from mlflow import MlflowClient

from anemoi.training.diagnostics.mlflow.auth import TokenAuth
from anemoi.training.diagnostics.mlflow.utils import health_check


class AnemoiMlflowClient(MlflowClient):
    """Anemoi extension of the MLflow client with token authentication support."""

    def __init__(
        self,
        tracking_uri: str,
        *args,
        authentication: bool = False,
        check_health: bool = True,
        **kwargs,
    ) -> None:
        """Behaves like a normal `mlflow.MlflowClient` but with token authentication injected on every call.

        Parameters
        ----------
        tracking_uri : str
            The URI of the MLflow tracking server.
        authentication : bool, optional
            Enable token authentication, by default False
        check_health : bool, optional
            Check the health of the MLflow server on init, by default True
        *args : Any
            Additional arguments to pass to the MLflow client.
        **kwargs : Any
            Additional keyword arguments to pass to the MLflow client.

        """
        self.anemoi_auth = TokenAuth(tracking_uri, enabled=authentication)
        if check_health:
            super().__getattribute__("anemoi_auth").authenticate()
            health_check(tracking_uri)
        super().__init__(tracking_uri, *args, **kwargs)

    def __getattribute__(self, name: str) -> Any:
        """Intercept attribute access and inject authentication."""
        attr = super().__getattribute__(name)
        if callable(attr) and name != "anemoi_auth":
            super().__getattribute__("anemoi_auth").authenticate()
        return attr
