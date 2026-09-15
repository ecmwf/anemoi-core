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
:meth:`Batch.collate` from reader :class:`~anemoi.models.data.sample.SourceSample`
objects. Tests frequently need the *result* of collation directly - a batch whose
tensors already carry a batch axis - which would otherwise mean spelling out a
:class:`~anemoi.models.data.spec.SourceSpec` per dataset at every call site.

These helpers take the spec's fields flat and assemble the sources. They are a
test convenience only; they are not a compatibility layer and production code
should not use them.
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch

from anemoi.models.data.batch import Batch
from anemoi.models.data.spec import SourceSpec
from anemoi.models.data.tensor_layout import TensorLayout
from anemoi.models.data.views import SourceView
from anemoi.models.data.views import create_source_view

LOGGER = logging.getLogger(__name__)

_SPEC_FIELDS = ("name", "variables", "layout", "statistics", "grid_size", "coordinates_are_static", "metadata")


def make_source(**kwargs) -> SourceView:
    """Build one source, taking the spec's fields flat alongside the payload.

    >>> make_source(name="era5", data=x, variables=["t"], layout=layout)
    """
    spec_kwargs = {key: kwargs.pop(key) for key in _SPEC_FIELDS if key in kwargs}
    return create_source_view(SourceSpec(**spec_kwargs), **kwargs)


def make_batch(
    data: Mapping[str, torch.Tensor | list[torch.Tensor]],
    coordinates: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    grid_sizes: Mapping[str, int] | None = None,
    timedeltas: Mapping[str, Any] | None = None,
    shard_sizes: Mapping[str, Any] | None = None,
    layouts: Mapping[str, TensorLayout] | None = None,
    variables: Mapping[str, list[str]] | None = None,
    statistics: Mapping[str, Any] | None = None,
    boundaries: Mapping[str, Any] | None = None,
    static_coords: frozenset[str] | set[str] | tuple[str, ...] = (),
) -> Batch:
    """Build an already-collated batch from per-dataset dicts.

    ``layouts`` and ``variables`` must cover every dataset in ``data``: a source
    cannot be described without them.
    """
    coordinates = coordinates or {}
    metadata = metadata or {}
    grid_sizes = grid_sizes or {}
    timedeltas = timedeltas or {}
    shard_sizes = shard_sizes or {}
    layouts = layouts or {}
    variables = variables or {}
    statistics = statistics or {}
    boundaries = boundaries or {}
    static = frozenset(static_coords)

    sources: dict[str, SourceView] = {}
    for name, payload in data.items():
        if name not in layouts or name not in variables:
            missing = "layout" if name not in layouts else "variables"
            msg = f"make_batch() needs a {missing} for dataset {name!r}."
            raise ValueError(msg)

        per_dataset_meta = metadata.get(name) if isinstance(metadata.get(name), dict) else None
        sources[name] = create_source_view(
            SourceSpec(
                name=name,
                variables=variables[name],
                layout=layouts[name],
                statistics=statistics.get(name, {}),
                grid_size=grid_sizes.get(name),
                coordinates_are_static=name in static,
            ),
            data=payload,
            coordinates=coordinates.get(name),
            timedeltas=timedeltas.get(name),
            boundaries=boundaries.get(name) or (per_dataset_meta or {}).get("boundaries"),
            shard_sizes=shard_sizes.get(name),
        )
    return Batch(sources)
