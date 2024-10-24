# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import os

import requests


def health_check(tracking_uri: str) -> None:
    """Query the health endpoint of an MLflow server.

    If the server is not reachable, raise an error and remind the user that authentication may be required.

    Raises
    ------
    ConnectionError
        If the server is not reachable.

    """
    token = os.getenv("MLFLOW_TRACKING_TOKEN")

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{tracking_uri}/health", headers=headers, timeout=60)

    if response.text == "OK":
        return

    error_msg = f"Could not connect to MLflow server at {tracking_uri}. "
    if not token:
        error_msg += "The server may require authentication, did you forget to turn it on?"
    raise ConnectionError(error_msg)
