# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import deque
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig
from omegaconf import ListConfig

from anemoi.training.diagnostics.mlflow import MAX_PARAMS_LENGTH

LOGGER = logging.getLogger(__name__)

# Flattened param under which a run logs its own `diagnostics.log.mlflow.max_params_length`
MAX_PARAMS_LENGTH_PARAM = "config.diagnostics.log.mlflow.max_params_length"


class FixedLengthSet:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._deque = deque(maxlen=maxlen)
        self._set = set()

    def add(self, item: float) -> None:
        if item in self._set:
            return  # Already present, do nothing
        if len(self._deque) == self.maxlen:
            oldest = self._deque.popleft()
            self._set.remove(oldest)
        self._deque.append(item)
        self._set.add(item)

    def __contains__(self, item: float):
        return item in self._set

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        return iter(self._deque)

    def __repr__(self):
        return f"{list(self._deque)}"


def expand_iterables(
    params: Any,
    *,
    recursive: bool = True,
) -> Any:
    """Enumerate list-like iterables of non-primitive elements as dicts.

    Converts lists, tuples, and ListConfigs into individual keyed dicts with
    numeric indices (e.g., 0, 1, ...) and additional summary keys ('all',
    'length') when the iterable contains nested structures. DictConfigs are
    converted to dicts. Dicts are copied into new dicts. Inputs of other
    types are returned without conversion.

    Parameters
    ----------
    params : Any
        Parameter dictionary (dict | DictConfig) to copy to a new dict,
        list (list | tuple | ListConfig) to expand, or
        Any type to be returned as is.
    recursive : bool, optional
        Expand nested dictionaries.
        Default is True.

    Returns
    -------
    Any
        Dictionary with all iterable values expanded, list/tuple of primitive
        types, or the plain `params` if it is neither a list nor a dict.

    Examples
    --------
        >>> expand_iterables({'a': ['a', 'b', 'c']})
        {'a': ['a', 'b', 'c']}
        >>> expand_iterables({'a': {'b': {'c': 123}}})
        {'a': {'b': {'c': 123}}}
        >>> expand_iterables({'a': [['a1', 'a2']]})
        {'a': {0: ['a1', 'a2'], 'length': 1, 'all': [['a1', 'a2']]}}
        >>> expand_iterables({'a': [[0, 1, 2], 'b', 'c']})
        {'a': {0: [0, 1, 2], 1: 'b', 2: 'c'},
        'a.length': 3,
        'a.all': [[0, 1, 2], 'b', 'c']}
    """
    list_types = list | tuple | ListConfig
    dict_types = dict | DictConfig
    expandable_types = dict_types | list_types

    def has_expandable_items(value: list_types) -> bool:
        return any(isinstance(item, expandable_types) for item in value)

    additional = {}
    if isinstance(params, list_types):
        if not has_expandable_items(params):
            return params
        additional["length"] = len(params)
        additional["all"] = params
        params = dict(enumerate(params))

    if not isinstance(params, dict_types):
        return params

    if recursive:
        return {key: expand_iterables(value) for key, value in params.items()} | additional
    return dict(params) | additional


def clean_config_params(params: dict[str, Any]) -> dict[str, Any]:
    """Clean up params to avoid issues with mlflow.

    Too many logged params will make the server take longer to render the
    experiment.

    Parameters
    ----------
    params : dict[str, Any]
        Parameters to clean up.

    Returns
    -------
    dict[str, Any]
        Cleaned up params ready for MlFlow.
    """
    prefixes_to_remove = [
        "system",
        "data",
        "dataloader",
        "model",
        "training",
        "diagnostics",
        "graph",
        "metadata.config",
        "config.dataset.sourcesmetadata.dataset.variables_metadata",
        "metadata.dataset.",
    ]

    keys_to_remove = [key for key in params if any(key.startswith(prefix) for prefix in prefixes_to_remove)]
    for key in keys_to_remove:
        del params[key]
    return params


def get_max_params_length(params: Mapping[str, Any]) -> int:
    """Get the maximum number of params a run was allowed to log.

    The value is read from the run's own logged config so that the limit
    applied when syncing a run is the one applied when training it.

    Parameters
    ----------
    params : Mapping[str, Any]
        Flattened params of the run, as logged to MLflow.

    Returns
    -------
    int
        The run's `max_params_length`, or `MAX_PARAMS_LENGTH` if it was not logged.
    """
    value = params.get(MAX_PARAMS_LENGTH_PARAM)
    if value is None:
        LOGGER.info("%s not found in run params, using default %s", MAX_PARAMS_LENGTH_PARAM, MAX_PARAMS_LENGTH)
        return MAX_PARAMS_LENGTH
    try:
        max_params_length = int(value)
    except (TypeError, ValueError):
        LOGGER.warning(
            "Invalid %s=%r in run params, using default %s",
            MAX_PARAMS_LENGTH_PARAM,
            value,
            MAX_PARAMS_LENGTH,
        )
        return MAX_PARAMS_LENGTH
    LOGGER.info("Using max_params_length=%s from run params", max_params_length)
    return max_params_length


def check_params_length(num_params: int, max_params_length: int) -> None:
    """Raise if there are more params than allowed.

    Parameters
    ----------
    num_params : int
        Number of params to log, after `clean_config_params`.
    max_params_length : int
        Maximum number of params allowed.

    Raises
    ------
    ValueError
        If `num_params` exceeds `max_params_length`.
    """
    if num_params > max_params_length:
        msg = (
            f"Too many params: {num_params} > {max_params_length}. "
            "Please revisit the fields being logged and add redundant or irrelevant ones to the "
            "clean_config_params function, or increase `diagnostics.log.mlflow.max_params_length`."
        )
        raise ValueError(msg)
