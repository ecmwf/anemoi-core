# (C) Copyright 2026 Anemoi contributors.
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


class BaseLatentFusion(nn.Module, ABC):
    """Fuse dataset encoder latents."""

    def __init__(
        self,
        *,
        input_channels: int,
        num_channels: int,
        dataset_names: Sequence[str],
        layer_kernels: DotDict,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}.")
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}.")
        if not dataset_names:
            raise ValueError("At least one dataset name is required for latent fusion.")
        if len(dataset_names) != len(set(dataset_names)):
            raise ValueError(f"Dataset names must be unique, got {list(dataset_names)}.")

        self.input_channels = input_channels
        self.num_channels = num_channels
        self.dataset_names = tuple(dataset_names)
        self.layer_factory = load_layer_kernels(layer_kernels)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, hidden_latent: Tensor, latents: Mapping[str, Tensor]) -> Tensor:
        """Fuse encoder outputs into one latent tensor."""
        dataset_names = tuple(latents)
        dataset_latents = tuple(latents.values())
        return maybe_checkpoint(
            self._forward,
            self.gradient_checkpointing,
            hidden_latent,
            dataset_names,
            dataset_latents,
        )

    @abstractmethod
    def _forward(
        self,
        hidden_latent: Tensor,
        dataset_names: Sequence[str],
        dataset_latents: Sequence[Tensor],
    ) -> Tensor:
        """Fuse encoder outputs."""


class SumLatentFusion(BaseLatentFusion):
    """Sum encoder outputs elementwise."""

    def __init__(
        self,
        *,
        input_channels: int,
        num_channels: int,
        dataset_names: Sequence[str],
        layer_kernels: DotDict,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_channels=num_channels,
            dataset_names=dataset_names,
            layer_kernels=layer_kernels,
            gradient_checkpointing=gradient_checkpointing,
        )

    def _forward(
        self,
        hidden_latent: Tensor,
        dataset_names: Sequence[str],
        dataset_latents: Sequence[Tensor],
    ) -> Tensor:
        """Return the elementwise sum of dataset latents."""
        latent_iterator = iter(dataset_latents)
        first_latent = next(latent_iterator)
        return sum(latent_iterator, start=first_latent)


class CrossAttentionLatentFusion(BaseLatentFusion):
    """Update the projected hidden state with pointwise cross-attention over encoder outputs."""

    def __init__(
        self,
        *,
        input_channels: int,
        num_channels: int,
        dataset_names: Sequence[str],
        layer_kernels: DotDict,
        num_heads: int,
        attn_channels: int | None = None,
        dropout_p: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attention_implementation: str = "scaled_dot_product_attention",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_channels=num_channels,
            dataset_names=dataset_names,
            layer_kernels=layer_kernels,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.hidden_projection = self.layer_factory.Linear(input_channels, num_channels)
        self.hidden_norm = self.layer_factory.LayerNorm(normalized_shape=num_channels)
        self.source_norm = self.layer_factory.LayerNorm(normalized_shape=num_channels)
        self.dataset_embeddings = nn.ParameterDict(
            {dataset_name: nn.Parameter(torch.empty(num_channels)) for dataset_name in self.dataset_names}
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
        for dataset_embedding in self.dataset_embeddings.values():
            nn.init.normal_(dataset_embedding, std=0.02)

    def _forward(
        self,
        hidden_latent: Tensor,
        dataset_names: Sequence[str],
        dataset_latents: Sequence[Tensor],
    ) -> Tensor:
        """Update the learned hidden state by attending across encoder outputs."""
        source_latents = self.source_norm(torch.stack(dataset_latents, dim=-2))
        dataset_embeddings = torch.stack([self.dataset_embeddings[name] for name in dataset_names])
        keys = source_latents + dataset_embeddings
        residual = self.hidden_projection(hidden_latent)
        query = self.hidden_norm(residual)
        update = self.attention(query, keys, source_latents)
        return residual + update
