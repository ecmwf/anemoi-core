# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import torch

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.losses.base import BaseLoss


@dataclass(frozen=True, slots=True)
class DatasetContextStatic:
    """Container for dataset-specific objects that are static across steps."""

    name: str
    loss: BaseLoss | torch.nn.Module
    metrics: Any
    val_metric_ranges: dict[str, Any]
    pre_processor: Any
    pre_processor_tendencies: Any | None
    post_processor: Any
    post_processor_tendencies: Any | None
    data_indices: IndexCollection
    output_mask: Any
    grid_indices: Any
    latlons_data: torch.Tensor


@dataclass(frozen=True, slots=True)
class DatasetContextDynamic:
    """Container for dataset-specific objects that change per batch."""

    batch_grid_shard_slice: slice | None
    effective_grid_shard_slice: slice | None
    grid_shard_shapes: Any


@dataclass(frozen=True, slots=True)
class DatasetContext:
    """Container for dataset-specific objects used during training steps."""

    static: DatasetContextStatic
    dynamic: DatasetContextDynamic

    @classmethod
    def from_static_dynamic(
        cls,
        static: DatasetContextStatic,
        dynamic: DatasetContextDynamic,
    ) -> DatasetContext:
        return cls(static=static, dynamic=dynamic)

    def with_effective_grid_shard_slice(self, effective_slice: slice | None) -> DatasetContext:
        return replace(
            self,
            dynamic=replace(self.dynamic, effective_grid_shard_slice=effective_slice),
        )
