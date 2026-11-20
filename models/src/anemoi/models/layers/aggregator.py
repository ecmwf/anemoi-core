# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import random
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


class PointwiseCrossAttentionAggregator(BaseLatentAggregator):
    """Fuse dataset latents with pointwise cross-attention over sources."""

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
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            source_channels=source_channels,
            gradient_checkpointing=gradient_checkpointing,
        )
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
            {source_name: nn.Parameter(0.02 * torch.randn(num_channels)) for source_name in self.source_names},
        )
        self.attention = PointwiseMultiHeadCrossAttention(
            num_heads=num_heads,
            embed_dim=num_channels,
            layer_kernels=self.layer_factory,
            attn_channels=attn_channels,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            dropout_p=dropout_p,
        )

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
        values = self.source_norm(torch.stack(projected_latents, dim=-2))
        keys = values + torch.stack([self.source_embeddings[name] for name in source_names])
        residual = self.hidden_projection(hidden_latent)
        query = self.hidden_norm(residual)
        update = self.attention(query, keys, values)
        return residual + update


class GatedLatentFusion(nn.Module):
    """Gated fusion block that folds an encoder latent into a running latent.

    Both tensors are node-aligned (same nodes, same channels), so the block works
    pointwise: the two layer-normalised latents are concatenated and fed to a
    sigmoid gate ("how much") and a two-layer MLP ("what"). The result is added
    to the running latent as a gated residual (Flamingo-style), so the block can
    learn to be a no-op.

    Weights are shared across nodes; the parameter count depends only on
    ``hidden_dim``.

    Parameters
    ----------
    hidden_dim : int
        Channel dimension of both latents.
    zero_init_value : bool, optional
        Zero-initialise the output projection of the value MLP so the block is
        an exact identity at initialisation while gradients still flow (the
        gate stays open). Closing the sigmoid gate instead would saturate it
        and stall learning. Default False.
    """

    def __init__(self, hidden_dim: int, zero_init_value: bool = False) -> None:
        super().__init__()
        # Separate norms: the running latent and a fresh encoder output have
        # different scales and distributions.
        self.norm_latent = nn.LayerNorm(hidden_dim)
        self.norm_input = nn.LayerNorm(hidden_dim)
        self.to_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.to_value = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if zero_init_value:
            nn.init.zeros_(self.to_value[-1].weight)
            nn.init.zeros_(self.to_value[-1].bias)

    def forward(self, latent: Tensor, encoder_output: Tensor) -> Tensor:
        """Fold ``encoder_output`` into the running ``latent``.

        Parameters
        ----------
        latent : Tensor
            Running latent, shape (nodes, hidden_dim).
        encoder_output : Tensor
            Encoder latent to incorporate, shape (nodes, hidden_dim).

        Returns
        -------
        Tensor
            Updated latent, shape (nodes, hidden_dim).
        """
        combined = torch.cat([self.norm_latent(latent), self.norm_input(encoder_output)], dim=-1)
        gate = self.to_gate(combined)
        value = self.to_value(combined)
        return latent + gate * value


class GatedFusionAggregator(BaseLatentAggregator):
    """Fold auxiliary latents into the principal latent through learned sigmoid gates.

    The principal source's latent is the running latent; every other active
    source is merged into it by its own :class:`GatedLatentFusion` block
    (sequential, one block per auxiliary source). Auxiliary sources may be
    absent at call time, the principal source may not. While training, the
    merge order is shuffled to avoid order dependence; in eval mode the
    configured source order is used.

    All sources must share the same channel dimension, which is also the
    output width.

    With ``zero_init_value=True`` every fusion block starts as an exact
    identity, so a dataset added to a trained model (e.g. through transfer
    learning, where its block is missing from the checkpoint) leaves the
    model's predictions unchanged at the start of fine-tuning.

    ```
    latent_aggregator:
      _target_: anemoi.models.layers.aggregator.GatedFusionAggregator
      principal_source: data
    ```
    """

    def __init__(
        self,
        *,
        input_channels: int,
        source_channels: Mapping[str, int],
        principal_source: str | None = None,
        zero_init_value: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            source_channels=source_channels,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._hidden_dim = next(iter(self.source_channels.values()))
        if any(channels != self._hidden_dim for channels in self.source_channels.values()):
            raise ValueError(
                f"All latent sources must have the same channel dimension for {self.__class__.__name__}, "
                f"got {self.source_channels}.",
            )

        self.principal_source = principal_source if principal_source is not None else self.source_names[0]
        if self.principal_source not in self.source_channels:
            raise ValueError(
                f"{self.__class__.__name__}: principal_source {self.principal_source!r} "
                f"is not one of the latent sources {self.source_names}.",
            )

        self.fusion = nn.ModuleDict(
            {
                name: GatedLatentFusion(hidden_dim=self._hidden_dim, zero_init_value=zero_init_value)
                for name in self.source_names
                if name != self.principal_source
            },
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
        latents = dict(zip(source_names, source_latents, strict=True))
        if self.principal_source not in latents:
            raise ValueError(
                f"{self.__class__.__name__} requires the principal source {self.principal_source!r}; "
                f"got {sorted(latents)}.",
            )

        x_latent = latents[self.principal_source]
        remaining = [name for name in source_names if name != self.principal_source]
        if self.training:
            random.shuffle(remaining)
        for name in remaining:
            x_latent = self.fusion[name](x_latent, latents[name])
        return x_latent
