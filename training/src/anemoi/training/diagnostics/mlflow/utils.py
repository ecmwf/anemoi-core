# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
from collections import deque
from typing import Any

from omegaconf import DictConfig
from omegaconf import ListConfig


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
    delimiter: str = ".",
) -> Any:
    """Enumerate list-like iterables of non-primitve elements

    Converts lists, tuples, and ListConfigs into individual keyed dicts with
    numeric indices (e.g., 0, 1, ...) and additional summary keys ('all',
    'length') when the iterable contains nested structures. Non-iterable
    values and iterables of primitive types are kept as-is.

    DictConfig are converted to dicts. Dicts are copied into new dicts. Inputs
    of other types are returned without conversion

    Parameters
    ----------
    params : dict[str, Any] | DictConfig[str, Any] | list | tuple | ListConfig
        Parameter dictionary, list, tuple, or configuration object to expand.
    recursive : bool, optional
        Expand nested dictionaries.
        Default is True.
    delimiter: str, optional
        Delimiter to use for keys.
        Default is ".".

    Returns
    -------
    Any
        Dictionary with all iterable values expanded or list/tuple of primitive types.

    Examples
    --------
        >>> expand_iterables({'a': ['a', 'b', 'c']})
        {'a': ['a', 'b', 'c']}
        >>> expand_iterables({'a': {'b': {'c': 123}}})
        {'a': {'b': {'c': 123}}}
        >>> expand_iterables({'a': [['a1', 'a2']]})
        {'a': {0: ['a1', 'a2']}, 'a.length': 1, 'a.all': [['a1', 'a2']]}
        >>> expand_iterables({'a': [[0, 1, 2], 'b', 'c'])
        {'a': {0: [0, 1, 2], 1: 'b', 2: 'c'},
        'a.length': 3,
        'a.all': [[0, 1, 2], 'b', 'c']}
    """

    list_types = list | tuple | ListConfig
    dict_types = dict | DictConfig
    expandable_types = dict_types | list_types

    def should_be_expanded(value: list_types) -> bool:
        return any(isinstance(item, expandable_types) for item in value)


    if isinstance(params, list_types):
        if not should_be_expanded(params):
            return params
        kv_iterable = enumerate(params)
    elif isinstance(params, dict_types):
        kv_iterable = params.items()
    else:
        return params

    expanded_params = {}

    for key, value in kv_iterable:
        if recursive:
            expanded = expand_iterables(value, recursive=recursive, delimiter=delimiter)
        else:
            expanded = value

        expanded_params[key] = expanded

        if isinstance(value, list_types) and expanded is not value:
            # add summary keys for expanded lists
            expanded_params[f"{key}{delimiter}length"] = len(value)
            expanded_params[f"{key}{delimiter}all"] = value

    return expanded_params


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
