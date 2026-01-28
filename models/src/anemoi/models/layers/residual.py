# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from abc import abstractmethod

import einops
import torch
from torch import nn

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.graph_provider_registry import GraphProviderRegistry
from anemoi.models.layers.sparse_projector import SparseProjector


class BaseResidualConnection(nn.Module, ABC):
    """Base class for residual connection modules."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, grid_shard_shapes=None, model_comm_group=None) -> torch.Tensor:
        """Define the residual connection operation.

        Should be overridden by subclasses.
        """
        pass


class SkipConnection(BaseResidualConnection):
    """Skip connection module

    This layer returns the most recent timestep from the input sequence.

    This module is used to bypass processing layers and directly pass the latest input forward.
    """

    def __init__(self, step: int = -1, **_) -> None:
        super().__init__()
        self.step = step

    def forward(self, x: torch.Tensor, grid_shard_shapes=None, model_comm_group=None) -> torch.Tensor:
        """Return the last timestep of the input sequence."""
        return x[:, self.step, ...]  # x shape: (batch, time, ens, nodes, features)


class TruncatedConnection(BaseResidualConnection):
    """Truncated skip connection using projection graph providers."""

    def __init__(
        self,
        down_provider: str,
        up_provider: str,
        graph_providers: GraphProviderRegistry,
        dataset_name: str,
        autocast: bool = False,
        **_,
    ) -> None:
        super().__init__()
        self.provider_down = graph_providers.get(down_provider, dataset_name)
        self.provider_up = graph_providers.get(up_provider, dataset_name)
        if not isinstance(self.provider_down, ProjectionGraphProvider) or not isinstance(
            self.provider_up, ProjectionGraphProvider
        ):
            raise TypeError("TruncatedConnection requires ProjectionGraphProvider instances.")

        self.projector = SparseProjector(autocast=autocast)

    def forward(self, x: torch.Tensor, grid_shard_shapes=None, model_comm_group=None) -> torch.Tensor:
        """Apply truncated skip connection."""
        batch_size = x.shape[0]
        x = x[:, -1, ...]  # pick latest step
        shard_shapes = apply_shard_shapes(x, 0, grid_shard_shapes) if grid_shard_shapes is not None else None

        x = einops.rearrange(x, "batch ensemble grid features -> (batch ensemble) grid features")
        x = self._to_channel_shards(x, shard_shapes, model_comm_group)
        x = self.projector(x, self.provider_down.get_edges(device=x.device))
        x = self.projector(x, self.provider_up.get_edges(device=x.device))
        x = self._to_grid_shards(x, shard_shapes, model_comm_group)
        x = einops.rearrange(x, "(batch ensemble) grid features -> batch ensemble grid features", batch=batch_size)

        return x

    def _to_channel_shards(self, x, shard_shapes=None, model_comm_group=None):
        return self._reshard(x, shard_channels, shard_shapes, model_comm_group)

    def _to_grid_shards(self, x, shard_shapes=None, model_comm_group=None):
        return self._reshard(x, gather_channels, shard_shapes, model_comm_group)

    def _reshard(self, x, fn, shard_shapes=None, model_comm_group=None):
        if shard_shapes is not None:
            x = fn(x, shard_shapes, model_comm_group)
        return x
