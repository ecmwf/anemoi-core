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
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig
from omegaconf import ListConfig

LOGGER = logging.getLogger(__name__)


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


def update_run_id(params: dict, key: str, new_run_id: str, src_run_id: str, run_type: str) -> dict:
    """Point the run-id param ``key`` at ``new_run_id``, keeping ``src_run_id`` as ``<run_type>_<key>``."""
    params[f"config.training.{key}"] = new_run_id
    params[f"config.training.{run_type}_{key}"] = src_run_id

    if key == "run_id":
        params[f"metadata.{run_type}_{key}"] = src_run_id
        params[f"metadata.{key}"] = new_run_id
    return params


def rewrite_lineage(
    params: dict,
    tags: dict,
    dst_run_id: str,
    src_run_id: str,
    resolve: Callable[[str], str | None],
    run_type: str = "offline",
) -> tuple[dict, dict]:
    """Rewrite the run ids stored in a synced run's params and tags to destination ids.

    A run synced to another tracking server gets a new run id there, but anemoi stores run
    ids inside the run (its own, its parent's for a resumed run, its source's for a forked
    run). This maps them to the destination ids and keeps the source ids under
    ``<run_type>`` keys, so lineage still holds on the destination.

    Parameters
    ----------
    params : dict
        The source run's params, updated in place.
    tags : dict
        The source run's tags, updated in place.
    dst_run_id : str
        The run id on the destination server.
    src_run_id : str
        The run id on the source (offline store or source server).
    resolve : Callable[[str], str | None]
        Maps a source run id (parent or fork source) to its destination run id, or None
        when it is not on the destination; ``dst_run_id`` is used then.
    run_type : str, optional
        ``offline`` or ``server2server``, by default ``offline``.

    Returns
    -------
    tuple[dict, dict]
        The rewritten params and tags.
    """

    def remap_tag(tag_key: str, tag_dest: str) -> None:
        dst_related_run_id = resolve(tags[tag_key]) or dst_run_id
        tags[tag_dest] = tags[tag_key]  # keep the source run id
        tags[tag_key] = dst_related_run_id  # point at the destination run id

    if (params["config.training.fork_run_id"] == "None") and (params["metadata.run_id"] == src_run_id):
        params = update_run_id(
            params,
            "run_id",
            new_run_id=dst_run_id,
            src_run_id=src_run_id,
            run_type=run_type,
        )

    elif "forkedRun" in tags:
        try:
            remap_tag("forkedRunId", f"{run_type}.forkedRunId")
            params = update_run_id(
                params,
                "fork_run_id",
                new_run_id=tags["forkedRunId"],
                src_run_id=tags[f"{run_type}.forkedRunId"],
                run_type=run_type,
            )
            params = update_run_id(
                params,
                "run_id",
                new_run_id=dst_run_id,
                src_run_id=src_run_id,
                run_type=run_type,
            )

        except AttributeError:
            LOGGER.warning("No forked run parent found")

    elif "resumedRun" in tags:
        try:
            remap_tag("mlflow.parentRunId", f"mlflow.{run_type}.parentRunId")
            params = update_run_id(
                params,
                "run_id",
                new_run_id=tags["mlflow.parentRunId"],
                src_run_id=tags[f"mlflow.{run_type}.parentRunId"],
                run_type=run_type,
            )

            # in the offline case that's the local folder name for the resumed run
            # in the server2server case that's the source server run_id of the resumed run
            params[f"config.training.{run_type}_self_run_id"] = src_run_id

        except AttributeError:
            LOGGER.warning("No parent run found")

    return params, tags
