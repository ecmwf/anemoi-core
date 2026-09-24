# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""This is the *model's* view of the data, not the data's own. Flattening folds the
time axis into the feature axis and the ensemble axis into the node axis, which is
the convention the encoder/decoder graphs expect.
"""

import logging
from dataclasses import dataclass

import torch

from anemoi.models.distributed.shapes import ShardSizes

LOGGER = logging.getLogger(__name__)


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

    data: torch.Tensor | None
    coordinates: torch.Tensor
    device: torch.device | None
    shard_sizes: ShardSizes
    batch_sizes: tuple[int, ...] | None = None
    timedeltas: torch.Tensor | None = None

    def __post_init__(self):
        if self.data is not None and self.data.ndim != 2:
            raise ValueError(f"{self.__class__} data must be 2-dimensional, got {self.data.ndim} dimensions.")

        if self.coordinates.ndim != 2:
            raise ValueError(
                f"{self.__class__.__name__} coordinates must be 2-dimensional, got {self.coordinates.ndim} dimensions."
            )

        if self.coordinates.shape[1] != 2:
            raise ValueError(
                f"{self.__class__.__name__} coordinates must have a shape of (nodes, 2), got {self.coordinates.shape}."
            )

        if self.data is not None and self.data.device != self.device:
            raise ValueError(
                f"{self.__class__.__name__} data must be on the same device as the source, got {self.data.device} and {self.device}."
            )

        if self.coordinates.device != self.device:
            raise ValueError(
                f"{self.__class__.__name__} coordinates must be on the same device as the source, got {self.coordinates.device} and {self.device}."
            )

    def to(self, device: torch.device) -> "FlatSource":
        """Return a copy of this view with all tensors moved to the given device."""
        return FlatSource(
            data=None if self.data is None else self.data.to(device),
            coordinates=self.coordinates.to(device),
            timedeltas=None if self.timedeltas is None else self.timedeltas.to(device),
            device=device,
            shard_sizes=self.shard_sizes,
            batch_sizes=self.batch_sizes,
        )
