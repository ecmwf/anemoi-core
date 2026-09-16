# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Marshalling between a :class:`Source` and the encoder's node-row tensors.

This is the *model's* view of the data, not the data's own. Flattening folds the
time axis into the feature axis and the ensemble axis into the node axis, which is
the convention the encoder/decoder graphs expect - one node set per
``(sample, member)`` - and is emphatically not a property of the batch. Keeping it
here is what lets ``anemoi.models.data`` stay a plain data package.

Dispatch is by :func:`functools.singledispatch` rather than a method, because the
source classes live in a package that must not know the encoder exists. That also
means a new source kind can register its own implementation without editing this
module.
"""

import logging
from dataclasses import dataclass
from functools import singledispatch

import einops
import numpy as np
import torch

from anemoi.models.data.source import GriddedSource
from anemoi.models.data.source import _Source
from anemoi.models.data.source import TabularSource
from anemoi.models.distributed.shapes import ShardSizes

LOGGER = logging.getLogger(__name__)

#: How a gridded source's axes collapse into ``(nodes, features)``. Time joins the
#: feature axis; batch and ensemble join the node axis.
FLATTEN_2D = "(batch ensemble grid) (time variables)"


@dataclass(frozen=True, slots=True)
class FlatSource:
    """A source flattened to node rows, ready for the encoder.

    Parameters
    ----------
    data : torch.Tensor
        ``(nodes, features)``.
    coordinates : torch.Tensor
        ``(nodes, 2)``, repeated to line up row-for-row with :attr:`data`.
    device : torch.device or None
        Device the tensors live on.
    shard_sizes : ShardSizes
        Per-rank node counts, or ``None`` when the source is replicated.
    batch_sizes : tuple[int, ...] or None
        Node count per ``(sample, member)``, so :func:`unflatten` can split the rows
        back up. ``None`` for a static grid, where every entry would be equal.
    timedeltas : torch.Tensor or None
        ``(nodes,)`` per-node time offsets, for sources that carry them.
    """

    data: torch.Tensor
    coordinates: torch.Tensor
    device: torch.device | None
    shard_sizes: ShardSizes
    batch_sizes: tuple[int, ...] | None = None
    timedeltas: torch.Tensor | None = None

    def to(self, device: torch.device) -> "FlatSource":
        """Return a copy of this view with all tensors moved to the given device."""
        return FlatSource(
            data=self.data.to(device),
            coordinates=self.coordinates.to(device),
            timedeltas=None if self.timedeltas is None else self.timedeltas.to(device),
            device=device,
            shard_sizes=self.shard_sizes,
            batch_sizes=self.batch_sizes,
        )


@singledispatch
def flatten(source: _Source) -> FlatSource:
    """Flatten a source to ``(nodes, features)`` for the encoder."""
    msg = f"No flatten() implementation registered for {type(source).__name__}."
    raise TypeError(msg)


@singledispatch
def unflatten(source: _Source, data: torch.Tensor, **kwargs) -> _Source:
    """Reshape encoder output back into ``source``'s own layout.

    ``source`` supplies the shape to restore; only ``data`` is taken from the
    flattened tensor. Extra keyword arguments are passed to
    :meth:`Source.clone` (e.g. a replaced ``spec``).
    """
    msg = f"No unflatten() implementation registered for {type(source).__name__}."
    raise TypeError(msg)


# -- gridded -----------------------------------------------------------------


@flatten.register
def _flatten_gridded(source: GriddedSource) -> FlatSource:
    layout, data = source.layout, source.data
    current_pattern = layout.normalized(data.ndim).pattern
    flattened_data = einops.rearrange(data, f"{current_pattern} -> {FLATTEN_2D}")
    device = data.device

    coordinates = source.coordinates
    if coordinates is None:
        msg = f"Source {source.name!r} requires coordinates for flattening."
        raise ValueError(msg)
    if isinstance(coordinates, list):
        msg = f"Source {source.name!r} coordinates must be a tensor, not a list."
        raise TypeError(msg)

    batch_size = data.shape[layout.axis("batch", ndim=data.ndim)]
    ensemble_size = data.shape[layout.axis("ensemble", ndim=data.ndim)]
    grid_size = data.shape[layout.axis("grid", ndim=data.ndim)]
    if coordinates.ndim not in (2, 3):
        msg = (
            f"Source {source.name!r} coordinates must have shape (grid, 2) "
            f"or (batch, grid, 2), got {tuple(coordinates.shape)}."
        )
        raise ValueError(msg)
    expected_shape = (grid_size, 2) if coordinates.ndim == 2 else (batch_size, grid_size, 2)
    if tuple(coordinates.shape) != expected_shape:
        raise ValueError(f"Source {source.name!r} coordinates must have shape {expected_shape}.")

    if coordinates.ndim == 2:
        repeated = einops.repeat(
            coordinates,
            "grid latlon -> (batch ensemble grid) latlon",
            batch=batch_size,
            ensemble=ensemble_size,
        )
    else:
        repeated = einops.repeat(
            coordinates,
            "batch grid latlon -> (batch ensemble grid) latlon",
            ensemble=ensemble_size,
        )

    return FlatSource(
        data=flattened_data,
        coordinates=repeated,  # already on device; see Batch.to
        timedeltas=None,
        device=device,
        shard_sizes=source.shard_sizes,
        batch_sizes=(
            (data.shape[layout.grid],) * (batch_size * ensemble_size)
            if not source.coordinates_are_static
            else None
        ),
    )


@unflatten.register
def _unflatten_gridded(source: GriddedSource, data: torch.Tensor, **kwargs) -> GriddedSource:
    layout, reference = source.layout, source.data
    new_data = einops.rearrange(
        data,
        f"{FLATTEN_2D} -> {layout.normalized(reference.ndim).pattern}",
        batch=reference.shape[layout.batch],
        ensemble=reference.shape[layout.ensemble],
        time=reference.shape[layout.time],
    )
    return source.clone(data=new_data, **kwargs)


# -- tabular -----------------------------------------------------------------


def _fold_members(source: TabularSource, sample: torch.Tensor) -> torch.Tensor:
    """Fold one sample's ensemble axis into its node axis, as the gridded path does.

    A gridded source flattens to ``(batch ensemble grid)``; doing the same here keeps
    the two kinds interchangeable downstream - the encoder and decoder graphs see one
    node set per ``(sample, member)`` either way.
    """
    layout = source.layout
    if layout.ensemble is None:
        return sample
    ensemble_axis = layout.axis("ensemble", ndim=sample.ndim)
    grid_axis = layout.axis("grid", ndim=sample.ndim)
    if ensemble_axis > grid_axis:
        msg = (
            f"Source {source.name!r} expects the ensemble axis before the grid axis so that "
            f"folding yields (ensemble grid) order; got {layout!r}."
        )
        raise ValueError(msg)
    return sample.flatten(ensemble_axis, grid_axis)


@flatten.register
def _flatten_tabular(source: TabularSource) -> FlatSource:
    layout, samples, coords = source.layout, source.data, source.coordinates

    if not isinstance(coords, list) or len(coords) != len(samples):
        raise ValueError(f"Source {source.name!r} requires one coordinate tensor per sample for flattening.")
    for sample, sample_coords in zip(samples, coords, strict=True):
        if tuple(sample_coords.shape) != (sample.shape[layout.grid], 2):
            raise ValueError(f"Source {source.name!r} requires one latitude/longitude pair per node.")
    if not samples:
        msg = f"Source {source.name!r} cannot flatten an empty batch."
        raise ValueError(msg)

    members = source.ensemble_size
    folded = [_fold_members(source, sample) for sample in samples]
    # coordinates and timedeltas are repeated per member to line up with the folded data
    repeated_coords = [c.repeat(members, 1) for c in coords]
    repeated_timedeltas = None if source.timedeltas is None else [td.repeat(members) for td in source.timedeltas]

    if len(folded) > 1:
        data = torch.cat(folded, dim=0)
        coordinates = torch.cat(repeated_coords, dim=0)
        timedeltas = None if repeated_timedeltas is None else torch.cat(repeated_timedeltas, dim=0)
    else:
        data = folded[0]
        coordinates = repeated_coords[0]
        timedeltas = None if repeated_timedeltas is None else repeated_timedeltas[0]

    if timedeltas is not None and timedeltas.shape[0] != coordinates.shape[0]:
        msg = (
            f"Source {source.name!r} timedeltas and coordinates must contain the same number of nodes, "
            f"got {timedeltas.shape[0]} and {coordinates.shape[0]}."
        )
        raise ValueError(msg)

    # Flatten per-window shard sizes into one list for the concatenated data.
    # NOTE this changes the order of observations when gathering:
    #   GPU0  GPU1   GPU0  GPU1            GPU0        GPU1
    #   w1_0, w1_1 | w2_0, w2_1  becomes  w1_0, w2_0, w1_1, w2_1
    flat_shard_sizes = None
    if source.shard_sizes is not None:
        if len(source.shard_sizes) != 1:
            msg = (
                f"Source {source.name!r}: a sharded tabular source is supported only at batch size 1, "
                f"but this batch has {len(source.shard_sizes)} samples."
            )
            raise NotImplementedError(msg)
        window_shard_sizes = source.shard_sizes[0]
        # sum per-rank shard sizes across all windows to get totals for the concatenated data
        flat_shard_sizes = [
            sum(sizes[rank] for sizes in window_shard_sizes) for rank in range(len(window_shard_sizes[0]))
        ]

    device = data.device
    return FlatSource(
        data=data,
        coordinates=coordinates.to(device),
        timedeltas=None if timedeltas is None else timedeltas.to(device),
        device=device,
        shard_sizes=flat_shard_sizes,
        batch_sizes=tuple(sample.shape[layout.grid] for sample in samples for _ in range(members)),
    )


@unflatten.register
def _unflatten_tabular(source: TabularSource, data: torch.Tensor, **kwargs) -> TabularSource:
    layout, samples = source.layout, source.data
    members = source.ensemble_size
    node_counts = [sample.shape[layout.grid] for sample in samples]
    row_counts = [count * members for count in node_counts]
    row_starts = np.cumsum([0] + row_counts[:-1])

    new_data = []
    for sample_index, rows in enumerate(row_counts):
        chunk = data.narrow(0, int(row_starts[sample_index]), rows)
        if layout.ensemble is not None:
            chunk = chunk.unflatten(0, (members, node_counts[sample_index]))
        new_data.append(chunk)
    return source.clone(data=new_data, **kwargs)
