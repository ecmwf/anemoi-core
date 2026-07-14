# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC
from abc import abstractmethod

import torch
from torch import Tensor
from torch import nn


class BaseLatentAggregator(nn.Module, ABC):
    """Base class for combining latent representations from multiple encoders.

    Subclasses implement the strategy for merging a dictionary of per-dataset
    latent tensors into a single latent tensor that is fed to the processor.
    """

    def __init__(self, num_channels: dict[str, int]) -> None:
        super().__init__()
        self.num_channels = num_channels

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Return the channel dimension of the aggregated latent tensor."""

    @abstractmethod
    def forward(self, latents: dict[str, Tensor]) -> Tensor:
        """Aggregate per-dataset latent tensors.

        Parameters
        ----------
        latents : dict[str, Tensor]
            Mapping from dataset name to latent tensor.
            Each tensor has shape ``(nodes, channels)``.

        Returns
        -------
        Tensor
            Aggregated latent tensor with shape ``(nodes, channels)``.
        """


class SumAggregator(BaseLatentAggregator):
    """Element-wise sum of latent representations.

    This is a zero-parameter, zero-overhead aggregator: when a single dataset
    is provided the tensor is returned as-is without any copy or computation.
    """

    def __init__(self, num_channels: dict[str, int]) -> None:
        super().__init__(num_channels)
        self._hidden_dim = list(num_channels.values())[0]
        assert all(ch == self._hidden_dim for ch in num_channels.values()), (
            f"All latent tensors must have the same channel dimension for {self.__class__.__name__}."
        )
        
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(self, latents: dict[str, Tensor]) -> Tensor:
        values = list(latents.values())
        if len(values) == 1:
            return values[0]
        return torch.stack(values).sum(dim=0)


class MeanAggregator(BaseLatentAggregator):
    """Element-wise mean of latent representations."""

    def __init__(self, num_channels: dict[str, int]) -> None:
        super().__init__(num_channels)
        self._hidden_dim = list(num_channels.values())[0]
        assert all(ch == self._hidden_dim for ch in num_channels.values()), (
            f"All latent tensors must have the same channel dimension for {self.__class__.__name__}."
        )
        
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(self, latents: dict[str, Tensor]) -> Tensor:
        values = list(latents.values())
        if len(values) == 1:
            return values[0]
        return torch.stack(values).mean(dim=0)


class ConcatAggregator(BaseLatentAggregator):
    """Concatenate latent representations and project back to the original channel dimension."""

    def __init__(self, num_channels: dict[str, int]) -> None:
        super().__init__(num_channels)
        self.expected_datasets = set(num_channels.keys())
        self._hidden_dim = sum(num_channels.values())

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(self, latents: dict[str, Tensor]) -> Tensor:
        if set(latents.keys()) != self.expected_datasets:
            raise ValueError(f"Latent tensors must match the expected datasets: {self.expected_datasets}")

        values = list(latents.values())
        if len(values) == 1:
            return values[0]

        return torch.cat(values, dim=-1)
