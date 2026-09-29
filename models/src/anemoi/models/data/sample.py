# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The contract between a data reader and :meth:`Batch.collate`.

A reader returns one :class:`SourceSample` per dataset and sample. The concrete class
says what kind of data it is, and knows how to collate a list of its own kind into
the matching :class:`~anemoi.models.data.sources.Source`.
"""

import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Self

import torch
from torch.utils.data import default_collate

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import Source
from anemoi.models.data.sources import TabularSource
from anemoi.models.distributed.shapes import ShardSizes

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceSample(ABC):
    """One dataset's contribution to one sample, as produced by a data reader.

    Everything needed to build the collated
    :class:`~anemoi.models.data.sources.Source` travels with the sample, so
    collation needs no side channel. Use :class:`GriddedSourceSample` or
    :class:`TabularSourceSample`.

    Parameters
    ----------
    data : torch.Tensor
        The sample's data, laid out per :attr:`layout`.
    variables : list[str]
        Variable names along the layout's ``variables`` axis, in order.
    layout : TensorLayout
        The **per-sample** layout, without a batch axis.
    coordinates : torch.Tensor
        ``(N, 2)`` stacking ``(latitudes, longitudes)`` in **radians**.
    statistics : Mapping[str, Any], optional
        Per-statistic arrays over the variable axis, as produced by
        ``anemoi-datasets``.
    """

    data: torch.Tensor
    variables: list[str]
    layout: TensorLayout
    coordinates: torch.Tensor
    statistics: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    @abstractmethod
    def collated_layout(cls, layout: TensorLayout) -> TensorLayout:
        """Return the layout of the collated source, given the per-sample ``layout``."""
        ...

    @classmethod
    @abstractmethod
    def collate(cls, name: str, samples: Sequence[Self]) -> Source:
        """Collate the samples of one dataset into a batched source."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} shape={tuple(self.data.shape)} dtype={self.data.dtype}>"


@dataclass(frozen=True, slots=True, kw_only=True)
class GriddedSourceSample(SourceSample):
    """A sample on a grid that every sample of the dataset shares.

    ``data`` is ``(time, ensemble, grid, variables)``, and every sample has the same shape.

    Parameters
    ----------
    grid_size : int or None, optional
        Full grid size before any distributed sharding.
    coordinates_are_static : bool, optional
        Whether this dataset's grid is fixed for the whole run (the reader's
        ``is_static_grid``). Static coordinates are shared by reference across the
        batch instead of being stacked, and skipped by pinning.
    shard_sizes : ShardSizes, optional
        Read-time sharding descriptor over the grid axis.
    """

    grid_size: int | None = None
    coordinates_are_static: bool = False
    shard_sizes: ShardSizes | None = None

    def __post_init__(self) -> None:
        if not self.layout.has_axis("time"):
            msg = f"{self.__class__.__name__} requires a layout with a time axis; got {self.layout!r}."
            raise ValueError(msg)

    @classmethod
    def collated_layout(cls, layout: TensorLayout) -> TensorLayout:
        """Samples are stacked along a new leading batch axis."""
        return layout.with_batch_dim()

    @classmethod
    def collate(cls, name: str, samples: Sequence[Self]) -> GriddedSource:
        """Stack the samples along a new leading batch axis.

        The coordinates of the first sample are used for the whole batch.
        """
        head = samples[0]
        layout = cls.collated_layout(head.layout)
        data = default_collate([s.data for s in samples])
        _validate_layout_against(name, layout, data)

        return GriddedSource(
            name=name,
            variables=head.variables,
            layout=layout,
            statistics=head.statistics,
            coordinates_are_static=head.coordinates_are_static,
            data=data,
            coordinates=head.coordinates,
            shard_sizes=head.shard_sizes,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TabularSourceSample(SourceSample):
    """A sample of points that change from sample to sample (e.g. observations).

    ``data`` is ``(ensemble, grid, variables)``. The time windows are stacked along
    the grid axis and described by :attr:`boundaries`.

    Parameters
    ----------
    timedeltas : torch.Tensor
        ``(N,)`` time offset of each point.
    boundaries : tuple[slice, ...]
        One slice per time window along the grid axis. This is the only record of
        the sample's time axis.
    shard_sizes : list[ShardSizes], optional
        Read-time sharding descriptor, one per time window.
    """

    timedeltas: torch.Tensor
    boundaries: tuple[slice, ...]
    shard_sizes: list[ShardSizes] | None = None

    def __post_init__(self) -> None:
        if self.layout.has_axis("time") or self.layout.has_axis("batch"):
            msg = (
                f"{self.__class__.__name__} requires a layout without time and batch axes; the time windows are "
                f"given by 'boundaries' and the batch is a list. Got {self.layout!r}."
            )
            raise ValueError(msg)

    @classmethod
    def collated_layout(cls, layout: TensorLayout) -> TensorLayout:
        """The samples are kept as a list, so the layout does not change."""
        return layout

    @classmethod
    def collate(cls, name: str, samples: Sequence[Self]) -> TabularSource:
        """Keep the samples as lists of length ``B``, since their grid sizes differ."""
        head = samples[0]
        layout = cls.collated_layout(head.layout)
        data = [s.data for s in samples]
        _validate_layout_against(name, layout, data)

        return TabularSource(
            name=name,
            variables=head.variables,
            layout=layout,
            statistics=head.statistics,
            data=data,
            coordinates=[s.coordinates for s in samples],
            timedeltas=[s.timedeltas for s in samples],
            boundaries=[s.boundaries for s in samples],
            shard_sizes=_collate_tabular_shard_sizes(name, samples),
        )


def _collate_tabular_shard_sizes(name: str, samples: Sequence[TabularSourceSample]) -> list[Any] | None:
    """Collate per-sample shard sizes, or ``None`` when no sample is sharded.

    Sources treat ``shard_sizes is None`` as replicated, so a list of ``None``
    entries would wrongly read as sharded. A mix of sharded and unsharded samples
    cannot be gathered consistently and is rejected.
    """
    shard_sizes = [s.shard_sizes for s in samples]
    n_unsharded = sum(sizes is None for sizes in shard_sizes)
    if n_unsharded == len(shard_sizes):
        return None
    if n_unsharded:
        msg = f"Dataset {name!r} mixes sharded and unsharded samples ({n_unsharded} of {len(shard_sizes)} unsharded)."
        raise ValueError(msg)
    return shard_sizes


def _validate_layout_against(name: str, layout: TensorLayout, data: torch.Tensor | list[torch.Tensor]) -> None:
    """Check every non-None axis position is a valid axis of the collated tensor.

    Catches reader-side mistakes early instead of letting them surface as cryptic
    errors deep inside model code.
    """
    ref = data[0] if isinstance(data, list) else data
    ndim = ref.ndim
    for axis_name in TensorLayout.AXES:
        pos = getattr(layout, axis_name)
        if pos is None:
            continue
        if not (-ndim <= pos < ndim):
            msg = (
                f"TensorLayout for dataset {name!r} declares {axis_name}={pos} but the "
                f"collated tensor only has {ndim} dimensions (shape={tuple(ref.shape)}). "
                f"Layout: {layout!r}."
            )
            raise ValueError(msg)
