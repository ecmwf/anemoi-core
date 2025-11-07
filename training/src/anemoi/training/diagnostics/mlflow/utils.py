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
        if isinstance(value, list | tuple):
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
        "hardware",
        "data",
        "dataloader",
        "model",
        "training",
        "diagnostics",
        "graph",
        "metadata.config",
        "config.dataset.sourcesmetadata.dataset.variables_metadata",
        "metadata.dataset.sources",
        "metadata.dataset.specific",
        "metadata.dataset.variables_metadata",
    ]

    keys_to_remove = [key for key in params if any(key.startswith(prefix) for prefix in prefixes_to_remove)]
    for key in keys_to_remove:
        del params[key]
    return params
