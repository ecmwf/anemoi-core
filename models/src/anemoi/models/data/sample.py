# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The contract between a data reader and :meth:`Batch.collate`."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch

from anemoi.models.data.tensor_layout import TensorLayout
from anemoi.models.distributed.shapes import ShardSizes

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceSample:
    """One dataset's contribution to one sample, as produced by a data reader.

    This is what ``reader.get_sample()`` returns and what :meth:`Batch.collate`
    consumes. Everything needed to build the collated
    :class:`~anemoi.models.data.spec.SourceSpec` travels with the payload, so
    collation needs no side channel.

    Parameters
    ----------
    data : torch.Tensor
        The sample's data, laid out per :attr:`layout`. For gridded datasets every
        sample has the same shape and they are stacked along a new batch axis; for
        tabular (sparse observation) datasets the leading grid extent varies per
        sample and they are kept as a list.
    variables : list[str]
        Variable names along the layout's ``variables`` axis, in order.
    layout : TensorLayout
        The **per-sample** layout. For gridded datasets :meth:`Batch.collate` shifts
        it with :meth:`TensorLayout.with_batch_dim` once the samples are stacked;
        for tabular datasets the batch axis is the outer list, so it stands as-is.
    statistics : Mapping[str, Any], optional
        Per-statistic arrays over the variable axis, as produced by
        ``anemoi-datasets``.
    grid_size : int or None, optional
        Full grid size before any distributed sharding. ``None`` for observation
        datasets, which have no static grid.
    coordinates_are_static : bool, optional
        Whether this dataset's grid is fixed for the whole run (the reader's
        ``is_static_grid``). Static coordinates are shared by reference across the
        batch instead of being stacked, and skipped by pinning.
    coordinates : torch.Tensor, optional
        ``(N, 2)`` stacking ``(latitudes, longitudes)`` in **radians**.
    timedeltas : torch.Tensor, optional
        ``(N,)`` per-point time offsets (sparse observation datasets only).
    boundaries : tuple[slice, ...], optional
        Per-time-window slices into the grid axis (sparse observation datasets
        only), which is how those datasets carry a time axis at all.
    shard_sizes : ShardSizes or list[ShardSizes], optional
        Read-time sharding descriptor. A single ``ShardSizes`` over the grid axis
        for gridded readers; one per window boundary for tabular readers.
    """

    data: torch.Tensor
    variables: list[str]
    layout: TensorLayout
    statistics: Mapping[str, Any] = field(default_factory=dict)
    grid_size: int | None = None
    coordinates_are_static: bool = False
    coordinates: torch.Tensor | None = None
    timedeltas: torch.Tensor | None = None
    boundaries: tuple[slice, ...] | None = None
    shard_sizes: ShardSizes | list[ShardSizes] | None = None

    def __post_init__(self) -> None:
        """Reject the combinations no reader should ever produce."""
        if self.is_tabular:
            if self.boundaries is None:
                msg = (
                    "A tabular sample (layout.time_in_grid=True) must carry 'boundaries'; "
                    "they are the only record of its time axis."
                )
                raise ValueError(msg)
            if self.coordinates_are_static:
                msg = (
                    "A tabular sample cannot have static coordinates: observation grids "
                    "vary per sample. Set coordinates_are_static=False."
                )
                raise ValueError(msg)
        elif self.boundaries is not None:
            msg = "'boundaries' is only meaningful for tabular samples (layout.time_in_grid=True)."
            raise ValueError(msg)

    @property
    def is_tabular(self) -> bool:
        """Whether time is folded into the grid axis (sparse observation sources)."""
        return self.layout.time_in_grid
