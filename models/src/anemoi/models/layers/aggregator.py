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

from anemoi.models.layers.attention import PointwiseMultiHeadCrossAttention
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.layers.utils import maybe_checkpoint
from anemoi.utils.config import DotDict


class BaseLatentAggregator(nn.Module, ABC):
    """Combine named dataset latents into the latent consumed by the processor."""

    def __init__(
        self,
        *,
        input_channels: int,
        source_channels: Mapping[str, int],
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}.")
        if not source_channels:
            raise ValueError("At least one latent source is required.")

        invalid_source_channels = {name: channels for name, channels in source_channels.items() if channels <= 0}
        if invalid_source_channels:
            raise ValueError(f"Source channels must be positive, got {invalid_source_channels}.")

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
        """Aggregate validated dataset latents."""


class SumAggregator(BaseLatentAggregator):
    """Sum latent representations elementwise."""

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


class MeanAggregator(SumAggregator):
    """Average latent representations elementwise."""

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
    """Concatenate every configured dataset latent in configured source order."""

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


class CrossAttentionAggregator(BaseLatentAggregator):
    """Fuse dataset latents with pointwise cross-attention over named sources."""

    def __init__(
        self,
        *,
        input_channels: int,
        source_channels: Mapping[str, int],
        num_channels: int,
        num_heads: int,
        layer_kernels: DotDict | None = None,
        attn_channels: int | None = None,
        dropout_p: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attention_implementation: str = "scaled_dot_product_attention",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            source_channels=source_channels,
            gradient_checkpointing=gradient_checkpointing,
        )
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}.")

        self.num_channels = num_channels
        self.layer_factory = load_layer_kernels(layer_kernels)
        self.hidden_projection = self.layer_factory.Linear(input_channels, num_channels)
        self.hidden_norm = self.layer_factory.LayerNorm(normalized_shape=num_channels)
        self.source_norm = self.layer_factory.LayerNorm(normalized_shape=num_channels)
        self.source_projections = nn.ModuleDict(
            {
                source_name: self.layer_factory.Linear(channels, num_channels)
                for source_name, channels in self.source_channels.items()
            },
        )
        self.source_embeddings = nn.ParameterDict(
            {source_name: nn.Parameter(torch.empty(num_channels)) for source_name in self.source_names},
        )
        self.attention = PointwiseMultiHeadCrossAttention(
            num_heads=num_heads,
            embed_dim=num_channels,
            layer_kernels=self.layer_factory,
            attn_channels=attn_channels,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            dropout_p=dropout_p,
            attention_implementation=attention_implementation,
        )
        for source_embedding in self.source_embeddings.values():
            nn.init.normal_(source_embedding, std=0.02)

    @property
    def hidden_dim(self) -> int:
        return self.num_channels

    def _forward(
        self,
        hidden_latent: Tensor,
        source_names: Sequence[str],
        source_latents: Sequence[Tensor],
    ) -> Tensor:
        projected_latents = tuple(
            self.source_projections[name](latent) for name, latent in zip(source_names, source_latents, strict=True)
        )
        source_latents = self.source_norm(torch.stack(projected_latents, dim=-2))
        source_embeddings = torch.stack([self.source_embeddings[name] for name in source_names])
        keys = source_latents + source_embeddings
        residual = self.hidden_projection(hidden_latent)
        query = self.hidden_norm(residual)
        update = self.attention(query, keys, source_latents)
        return residual + update
