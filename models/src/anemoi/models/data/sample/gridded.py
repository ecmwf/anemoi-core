# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import ClassVar
from typing import Self

import torch
from torch.utils.data import default_collate

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sample import sample_registry
from anemoi.models.data.sample.base import BaseSample
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import GriddedTemplate
from anemoi.models.distributed.shapes import ShardSizes


@sample_registry.register("gridded")
@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class GriddedSample(BaseSample):
    """One gridded dataset's contribution to one sample.

    A gridded dataset has a single, fixed grid: the grid belongs to the dataset, not to
    the sample. Every sample a reader returns for that dataset holds the fields at the
    sample's time steps on that same grid, so the samples share their ``coordinates``,
    ``variables``, ``statistics`` and ``shard_sizes``, differ only in ``data``, and stack
    into the single tensor of one :class:`~anemoi.models.data.sources.GriddedSource`.
    Collation checks that the shared fields agree. Points that change from sample to
    sample are tabular data instead (:class:`TabularSample`).

    ``data`` has a ``time``, an ``ensemble``, a ``grid`` and a ``variables`` axis, e.g.
    ``(time, ensemble, grid, variables)`` in the order given by :attr:`layout`.

    When the grid is sharded across a reader group, each rank's sample holds only its
    shard: ``data`` and ``coordinates`` cover the rank's points, and ``shard_sizes``
    gives every rank's share of the full grid.

    Parameters
    ----------
    grid_size : int or None, optional
        Number of points in the dataset's full grid, before sharding.
        :func:`~anemoi.models.data.sample.create_batched_struct` defaults it to the
        number of coordinates, or the sum of ``shard_sizes``, and rejects any other value.
    shard_sizes : ShardSizes, optional
        Number of grid points held by each rank of the reader group, one entry per rank;
        ``None`` when the sample holds the whole grid.
    """

    grid_size: int | None = None
    shard_sizes: ShardSizes | None = None

    source_type: ClassVar[type[GriddedSource]] = GriddedSource
    template_type: ClassVar[type[GriddedTemplate]] = GriddedTemplate
    _METADATA_ATTRS: ClassVar[tuple[str, ...]] = ("variables", "statistics", "coordinates", "shard_sizes")
    _DYNAMIC_ATTRS: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        if not self.layout.has_axis("time"):
            msg = f"{self.__class__.__name__} requires a layout with a time axis; got {self.layout!r}."
            raise ValueError(msg)

    @classmethod
    def _collate_layout(cls, layout: TensorLayout) -> TensorLayout:
        """Return the layout of the collated source, given the per-sample ``layout``."""
        return layout.with_batch_dim()

    @classmethod
    def _collate_data(cls, samples: Sequence[Self]) -> torch.Tensor:
        """Collate the per-sample data tensors into a single tensor."""
        return default_collate([s.data for s in samples])

    @classmethod
    def from_validated(
        cls,
        *,
        n_points: int,
        device: torch.device | None,
        grid_size: int | None = None,
        shard_sizes: ShardSizes | None = None,
        **common: Any,
    ) -> Self:
        """Default ``grid_size`` to the full grid, and check it against the sharding.

        Without sharding the sample holds the full grid of ``n_points``. With sharding
        it holds one shard, and the full grid is the sum of ``shard_sizes``.
        """
        del device
        full_size = n_points if shard_sizes is None else sum(shard_sizes)
        if grid_size is None:
            grid_size = full_size

        if grid_size != full_size:
            msg = f"grid_size {grid_size} does not match the full grid of {full_size} points."
            raise ValueError(msg)

        if shard_sizes is not None and n_points not in shard_sizes:
            msg = f"The sample's {n_points} points do not match any of the shard sizes {list(shard_sizes)}."
            raise ValueError(msg)

        return cls(**common, grid_size=grid_size, shard_sizes=shard_sizes)
