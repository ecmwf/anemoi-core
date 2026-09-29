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
from collections.abc import Collection
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor
from torch import nn

from anemoi.models.layers.utils import maybe_checkpoint


class BaseLatentAggregator(nn.Module, ABC):
    """Combine named dataset latents for the processor.

    Sources listed in ``dropped_sources`` at call time are still passed to
    ``_forward`` (so the computation graph stays static for DDP) but must not
    contribute to the aggregated latent.
    """

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

    def forward(
        self,
        hidden_latent: Tensor,
        latents: Mapping[str, Tensor],
        dropped_sources: Optional[Collection[str]] = None,
    ) -> Tensor:
        """Aggregate dataset latents in configured source order.

        Parameters
        ----------
        hidden_latent : Tensor
            Hidden-grid node latent, shape (nodes, input_channels).
        latents : Mapping[str, Tensor]
            Encoder latents keyed by source name, each (nodes, channels).
        dropped_sources : Collection[str], optional
            Sources present in ``latents`` whose contribution must be masked
            out (e.g. dataset dropout) while keeping the graph static.
        """
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

        dropped = frozenset(dropped_sources or ()).intersection(source_names)

        return maybe_checkpoint(
            self._forward,
            self.gradient_checkpointing,
            hidden_latent,
            source_names,
            source_latents,
            dropped,
        )

    @abstractmethod
    def _forward(
        self,
        hidden_latent: Tensor,
        source_names: Sequence[str],
        source_latents: Sequence[Tensor],
        dropped_sources: frozenset[str],
    ) -> Tensor:
        """Aggregate dataset latents."""

    @staticmethod
    def _mask_dropped(
        source_names: Sequence[str],
        source_latents: Sequence[Tensor],
        dropped_sources: frozenset[str],
    ) -> tuple[Tensor, ...]:
        """Zero the latents of dropped sources (multiplication keeps them in the graph)."""
        if not dropped_sources:
            return tuple(source_latents)
        return tuple(
            latent * 0.0 if name in dropped_sources else latent
            for name, latent in zip(source_names, source_latents, strict=True)
        )


class SumAggregator(BaseLatentAggregator):
    """Sum latents element-wise. Dropped sources are zeroed before summing."""

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
        dropped_sources: frozenset[str],
    ) -> Tensor:
        source_latents = self._mask_dropped(source_names, source_latents, dropped_sources)
        if len(source_latents) == 1:
            return source_latents[0]
        return torch.stack(source_latents, dim=0).sum(dim=0)


class MeanAggregator(BaseLatentAggregator):
    """Average latents element-wise. Dropped sources are excluded from the mean."""

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
        dropped_sources: frozenset[str],
    ) -> Tensor:
        n_active = len(source_latents) - len(dropped_sources)
        if n_active <= 0:
            raise ValueError(f"{self.__class__.__name__}: every latent source was dropped.")
        source_latents = self._mask_dropped(source_names, source_latents, dropped_sources)
        if len(source_latents) == 1:
            return source_latents[0]
        return torch.stack(source_latents, dim=0).sum(dim=0) / n_active


class ConcatAggregator(BaseLatentAggregator):
    """Concatenate dataset latents in source order. Dropped sources are zeroed."""

    @property
    def hidden_dim(self) -> int:
        return sum(self.source_channels.values())

    def _forward(
        self,
        hidden_latent: Tensor,
        source_names: Sequence[str],
        source_latents: Sequence[Tensor],
        dropped_sources: frozenset[str],
    ) -> Tensor:
        if tuple(source_names) != self.source_names:
            missing_sources = set(self.source_names).difference(source_names)
            raise ValueError(
                f"{self.__class__.__name__} requires every configured latent source; "
                f"missing {sorted(missing_sources)}.",
            )
        return torch.cat(self._mask_dropped(source_names, source_latents, dropped_sources), dim=-1)


class GatedLatentFusion(nn.Module):
    """Gated fusion block to incorporate an encoder latent into the running latent.

    NOT attention — there is no Q/K/V or softmax over positions. Both tensors
    are already node-aligned (same N nodes, same D dims), so we use concatenation
    + MLP instead. This is simpler, cheaper, and sufficient for pointwise fusion.

    Uses a learned sigmoid gate (Flamingo-style) so the block can learn to be a
    no-op — critical for optional encoders that may be absent at inference time.

    Applied independently per node with full weight sharing across all nodes.
    Parameter count depends only on hidden_dim (D), not on number of nodes (N).

    Both inputs and output have shape [N, D].
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Separate norms because latent (running accumulation) and encoder output
        # (fresh from encoder) have different scale/distribution.
        self.norm_latent = nn.LayerNorm(hidden_dim)
        self.norm_input = nn.LayerNorm(hidden_dim)
        # Gate: single linear → sigmoid. Only needs to learn "how much" to incorporate.
        self.to_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        # Value: 2-layer MLP with SiLU. Needs more capacity to learn "what" to add.
        self.to_value = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, latent: Tensor, encoder_output: Tensor, drop: bool = False) -> Tensor:
        """Fold encoder_output into the running latent.

        Parameters
        ----------
        latent : Tensor
            Running latent of shape [N, D].
        encoder_output : Tensor
            Encoder output to incorporate, shape [N, D].
        drop : bool, optional
            If True, force the gate to 0 so this block becomes a no-op while still
            running the full computation (needed to keep the DDP graph static).

        Returns
        -------
        Tensor
            Updated latent of shape [N, D].
        """
        ln = self.norm_latent(latent)
        en = self.norm_input(encoder_output)
        combined = torch.cat([ln, en], dim=-1)  # [N, 2D]
        gate = self.to_gate(combined)  # [N, D] in (0, 1)
        if drop:
            gate = gate * 0.0
        value = self.to_value(combined)  # [N, D]
        # Gated residual: if gate → 0, block is a no-op (safe for missing encoders)
        return latent + gate * value


class GatedFusionAggregator(BaseLatentAggregator):
    """Fold auxiliary latents into the principal latent through learned sigmoid gates.

    The principal source's latent is the running latent; every other source is
    merged into it by its own :class:`GatedLatentFusion` block (Perceiver-style
    sequential flow, one block per auxiliary source). Dropped sources still run
    their block with the gate forced to zero so the graph stays static for DDP.
    The merge order is shuffled while training to avoid order dependence.

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
        principal_source: Optional[str] = None,
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
                name: GatedLatentFusion(hidden_dim=self._hidden_dim)
                for name in self.source_names
                if name != self.principal_source
            }
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def _forward(
        self,
        hidden_latent: Tensor,
        source_names: Sequence[str],
        source_latents: Sequence[Tensor],
        dropped_sources: frozenset[str],
    ) -> Tensor:
        latents = dict(zip(source_names, source_latents, strict=True))
        if self.principal_source not in latents:
            raise ValueError(
                f"{self.__class__.__name__} requires the principal source {self.principal_source!r}; "
                f"got {sorted(latents)}.",
            )
        if self.principal_source in dropped_sources:
            raise ValueError(
                f"{self.__class__.__name__}: the principal source {self.principal_source!r} cannot be dropped.",
            )

        x_latent = latents[self.principal_source]
        remaining = [name for name in source_names if name != self.principal_source]
        if self.training:
            random.shuffle(remaining)
        for name in remaining:
            x_latent = self.fusion[name](x_latent, latents[name], drop=name in dropped_sources)
        return x_latent
