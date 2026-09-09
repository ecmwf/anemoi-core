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
from collections.abc import Mapping
from collections.abc import Sequence

import torch
from torch import Tensor
from torch import nn

from anemoi.models.layers.utils import maybe_checkpoint


class BaseLatentAggregator(nn.Module, ABC):
    """Combine named dataset latents for the processor."""

    def __init__(
        self,
        *,
        input_channels: int,
        source_channels: Mapping[str, int],
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError(f"{self.__class__.__name__}: input_channels must be positive, got {input_channels}.")

        if not source_channels:
            raise ValueError(f"{self.__class__.__name__}: At least one latent source is required.")

        self.input_channels = input_channels
        self.source_channels = dict(source_channels)
        self.source_names = tuple(source_channels)
        self.gradient_checkpointing = gradient_checkpointing

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Return the channel dimension of the aggregated latent tensor."""

    def forward(self, hidden_latent: Tensor, latents: Mapping[str, Tensor]) -> Tensor:
        """Aggregate dataset latents in configured source order."""
        if hidden_latent.shape[-1] != self.input_channels:
            raise ValueError(
                f"Hidden latent must have {self.input_channels} channels, got {hidden_latent.shape[-1]}.",
            )
        if not latents:
            raise ValueError("At least one latent tensor is required.")

        unknown_sources = set(latents).difference(self.source_channels)
        if unknown_sources:
            raise ValueError(f"Unknown latent sources: {sorted(unknown_sources)}.")

        source_names = tuple(name for name in self.source_names if name in latents)
        source_latents = tuple(latents[name] for name in source_names)
        for source_name, latent in zip(source_names, source_latents, strict=True):
            expected_channels = self.source_channels[source_name]
            if latent.shape[-1] != expected_channels:
                raise ValueError(
                    f"Latent source '{source_name}' must have {expected_channels} channels, got {latent.shape[-1]}.",
                )
            if latent.shape[:-1] != hidden_latent.shape[:-1]:
                raise ValueError(
                    f"Latent source '{source_name}' and the hidden latent must have matching leading dimensions, "
                    f"got {latent.shape[:-1]} and {hidden_latent.shape[:-1]}.",
                )

        return maybe_checkpoint(
            self._forward,
            self.gradient_checkpointing,
            hidden_latent,
            source_names,
            source_latents,
        )

    @abstractmethod
    def _forward(
        self,
        hidden_latent: Tensor,
        source_names: Sequence[str],
        source_latents: Sequence[Tensor],
    ) -> Tensor:
        """Aggregate dataset latents."""


class SumAggregator(BaseLatentAggregator):
    """Sum latents element-wise."""

    def __init__(self, *, input_channels: int, source_channels: Mapping[str, int]) -> None:
        super().__init__(input_channels=input_channels, source_channels=source_channels)
        self._hidden_dim = next(iter(self.source_channels.values()))
        if any(channels != self._hidden_dim for channels in self.source_channels.values()):
            raise ValueError(
                f"All latent sources must have the same channel dimension for {self.__class__.__name__}, "
                f"got {self.source_channels}.",
            )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def _forward(
        self,
        hidden_latent: Tensor,
        source_names: Sequence[str],
        source_latents: Sequence[Tensor],
    ) -> Tensor:
        if len(source_latents) == 1:
            return source_latents[0]
        return torch.stack(tuple(source_latents), dim=0).sum(dim=0)


class MeanAggregator(BaseLatentAggregator):
    """Average latents element-wise."""

    def __init__(self, *, input_channels: int, source_channels: Mapping[str, int]) -> None:
        super().__init__(input_channels=input_channels, source_channels=source_channels)
        self._hidden_dim = next(iter(self.source_channels.values()))
        if any(channels != self._hidden_dim for channels in self.source_channels.values()):
            raise ValueError(
                f"All latent sources must have the same channel dimension for {self.__class__.__name__}, "
                f"got {self.source_channels}.",
            )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def _forward(
        self,
        hidden_latent: Tensor,
        source_names: Sequence[str],
        source_latents: Sequence[Tensor],
    ) -> Tensor:
        if len(source_latents) == 1:
            return source_latents[0]
        return torch.stack(tuple(source_latents), dim=0).mean(dim=0)


class ConcatAggregator(BaseLatentAggregator):
    """Concatenate dataset latents in source order."""

    @property
    def hidden_dim(self) -> int:
        return sum(self.source_channels.values())

    def _forward(
        self,
        hidden_latent: Tensor,
        source_names: Sequence[str],
        source_latents: Sequence[Tensor],
    ) -> Tensor:
        if tuple(source_names) != self.source_names:
            missing_sources = set(self.source_names).difference(source_names)
            raise ValueError(
                f"{self.__class__.__name__} requires every configured latent source; "
                f"missing {sorted(missing_sources)}.",
            )
        return torch.cat(tuple(source_latents), dim=-1)
