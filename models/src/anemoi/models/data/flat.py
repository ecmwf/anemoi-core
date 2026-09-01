# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from dataclasses import dataclass
import torch

from anemoi.models.distributed.shapes import ShardSizes


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FlatView:
    """A flattened view of the data, coordinates and timedeltas for a single sample.

    This is used as an intermediate representation when applying functions or
    losses to the data, before unflattening back to a SourceView.
    """

    data: torch.Tensor
    coordinates: torch.Tensor
    device: torch.device | None
    shard_sizes: ShardSizes
    batch_sizes: tuple[int, ...] | None = None
    timedeltas: torch.Tensor | None = None

    def to(self, device: torch.device) -> "FlatView":
        """Return a copy of this view with all tensors moved to the given device."""
        return FlatView(
            data=self.data.to(device),
            coordinates=self.coordinates.to(device),
            timedeltas=None if self.timedeltas is None else self.timedeltas.to(device),
            device=device,
            shard_sizes=self.shard_sizes,
            batch_sizes=self.batch_sizes,
        )

