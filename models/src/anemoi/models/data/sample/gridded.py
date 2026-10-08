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
from anemoi.models.data.sample import source_sample_registry
from anemoi.models.data.sample.base import SourceSample
from anemoi.models.data.sources import GriddedSource
from anemoi.models.distributed.shapes import ShardSizes


@source_sample_registry.register("gridded")
@dataclass(frozen=True, eq=False, slots=True, kw_only=True)
class GriddedSourceSample(SourceSample):
    """A sample on a grid that every sample of the dataset shares.

    ``data`` is ``(time, ensemble, grid, variables)``, and every sample has the same shape.

    Parameters
    ----------
    grid_size : int or None, optional
        Full grid size before any distributed sharding.
    shard_sizes : ShardSizes, optional
        Read-time sharding descriptor over the grid axis.
    """

    grid_size: int | None = None
    shard_sizes: ShardSizes | None = None

    source_type: ClassVar[type[GriddedSource]] = GriddedSource
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
