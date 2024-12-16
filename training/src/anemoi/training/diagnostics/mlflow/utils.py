# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import functools
import os
from typing import Any

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


def expand_iterables(
    params: dict[str, Any],
    *,
    size_threshold: int | None = None,
    recursive: bool = True,
    delimiter: str = ".",
) -> dict[str, Any]:
    """Expand any iterable values to the form {key.i: value_i}.

    If expanded will also add {key.all: [value_0, value_1, ...], key.length: len([value_0, value_1, ...])}.

    If `size_threshold` is not None, expand the iterable only if the length of str(value) is
    greater than `size_threshold`.

    Parameters
    ----------
    params : dict[str, Any]
        Parameters to be expanded.
    size_threshold : int | None, optional
        Threshold of str(value) to expand iterable at.
        Default is None.
    recursive : bool, optional
        Expand nested dictionaries.
        Default is True.
    delimiter: str, optional
        Delimiter to use for keys.
        Default is ".".

    Returns
    -------
    dict[str, Any]
        Dictionary with all iterable values expanded.

    Examples
    --------
        >>> expand_iterables({'a': ['a', 'b', 'c']})
        {'a.0': 'a', 'a.1': 'b', 'a.2': 'c', 'a.all': ['a', 'b', 'c'], 'a.length': 3}
        >>> expand_iterables({'a': {'b': ['a', 'b', 'c']}})
        {'a': {'b.0': 'a', 'b.1': 'b', 'b.2': 'c', 'b.all': ['a', 'b', 'c'], 'b.length': 3}}
        >>> expand_iterables({'a': ['a', 'b', 'c']}, size_threshold=100)
        {'a': ['a', 'b', 'c']}
        >>> expand_iterables({'a': [[0,1,2], 'b', 'c']})
        {'a.0': {0: 0, 1: 1, 2: 2}, 'a.1': 'b', 'a.2': 'c', 'a.all': [[0, 1, 2], 'b', 'c'], 'a.length': 3}
    """

    def should_be_expanded(x: Any) -> bool:
        return size_threshold is None or len(str(x)) > size_threshold

    nested_func = functools.partial(expand_iterables, size_threshold=size_threshold, recursive=recursive)

    def expand(val: dict | list) -> dict[str, Any]:
        if not recursive:
            return val
        if isinstance(val, dict):
            return nested_func(val)
        if isinstance(val, list):
            return nested_func(dict(enumerate(val)))
        return val

    expanded_params = {}

    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            if should_be_expanded(value):
                for i, v in enumerate(value):
                    expanded_params[f"{key}{delimiter}{i}"] = expand(v)

                expanded_params[f"{key}{delimiter}all"] = value
                expanded_params[f"{key}{delimiter}length"] = len(value)
            else:
                expanded_params[key] = value
        else:
            expanded_params[key] = expand(value)
    return expanded_params
