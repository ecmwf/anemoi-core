# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Builders for constructing already-collated batches in tests.

Production code builds a :class:`~anemoi.models.data.batch.Batch` through
:meth:`Batch.collate` from reader :class:`~anemoi.models.data.sample.BaseSample`
objects. Tests frequently need the *result* of collation directly - a batch whose
tensors already carry a batch axis. :func:`build_batch` takes per-dataset dicts
and builds a :class:`TabularSource` for list payloads and a :class:`GriddedSource` otherwise.

An identical copy lives under each package's ``tests/`` directory, since the two
suites are collected in separate pytest processes and neither package's tests are
importable from the other.
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch

from anemoi.models.data.batch import Batch
from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import Source
from anemoi.models.data.sources import TabularSource

LOGGER = logging.getLogger(__name__)


def build_batch(
    data: Mapping[str, torch.Tensor | list[torch.Tensor]],
    coordinates: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    timedeltas: Mapping[str, Any] | None = None,
    shard_sizes: Mapping[str, Any] | None = None,
    layouts: Mapping[str, TensorLayout] | None = None,
    variables: Mapping[str, list[str]] | None = None,
    statistics: Mapping[str, Any] | None = None,
    boundaries: Mapping[str, Any] | None = None,
) -> Batch:
    """Build an already-collated batch from per-dataset dicts.

    ``layouts`` and ``variables`` must cover every dataset in ``data``: a source
    cannot be described without them.
    """
    coordinates = coordinates or {}
    metadata = metadata or {}
    timedeltas = timedeltas or {}
    shard_sizes = shard_sizes or {}
    layouts = layouts or {}
    variables = variables or {}
    statistics = statistics or {}
    boundaries = boundaries or {}

    sources: dict[str, Source] = {}
    for name, payload in data.items():
        if name not in layouts or name not in variables:
            missing = "layout" if name not in layouts else "variables"
            msg = f"build_batch() needs a {missing} for dataset {name!r}."
            raise ValueError(msg)

        per_dataset_meta = metadata.get(name) if isinstance(metadata.get(name), dict) else None
        common = {
            "name": name,
            "variables": variables[name],
            "layout": layouts[name],
            "statistics": statistics.get(name, {}),
            "data": payload,
            "coordinates": coordinates.get(name),
            "shard_sizes": shard_sizes.get(name),
        }
        if isinstance(payload, list):
            sources[name] = TabularSource(
                **common,
                timedeltas=timedeltas.get(name),
                boundaries=boundaries.get(name) or (per_dataset_meta or {}).get("boundaries"),
            )
        else:
            sources[name] = GriddedSource(**common)

    return Batch(sources)
