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
from typing import Any

import einops
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from anemoi.models.variable.embedmetadata import EmbedMetadata
from anemoi.models.variable.variablevocabular import VariableVocabulary


class BaseVariableTokenizer(nn.Module, ABC):
    """Pool a variable-length field axis into a fixed-width node representation."""

    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        vocabulary: VariableVocabulary,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        if emb_dim < 1 or out_dim < 1:
            raise ValueError("emb_dim and out_dim must be positive")
        if not isinstance(vocabulary, VariableVocabulary):
            raise TypeError("vocabulary must be a VariableVocabulary")

        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.emb_variables_metadata = EmbedMetadata(vocabulary=vocabulary, emb_dim=emb_dim)
        self.value_encoder = nn.Linear(1, emb_dim)
        self.normalize = nn.LayerNorm(emb_dim)
        self.output_projection = nn.Linear(emb_dim, out_dim)

    def _embed_values(self, x: torch.Tensor, variables: str | list[str]) -> torch.Tensor:
        variables = [variables] if isinstance(variables, str) else variables
        if x.shape[-1] != len(variables):
            raise ValueError(f"Input has {x.shape[-1]} fields but {len(variables)} variable names were provided")
        metadata = self.emb_variables_metadata(variables)
        return self.value_encoder(x.unsqueeze(-1)) + metadata

    @abstractmethod
    def forward(self, x: torch.Tensor, variables: str | list[str]) -> torch.Tensor: ...


class MeanPoolingTransform(BaseVariableTokenizer):
    """Mean-pool the available field tokens at every node."""

    def forward(self, x: torch.Tensor, variables: str | list[str]) -> torch.Tensor:
        tokens = self._embed_values(x, variables).mean(dim=-2)
        tokens = self.output_projection(self.normalize(tokens))
        return einops.rearrange(
            tokens,
            "batch time ensemble grid features -> (batch ensemble grid) (time features)",
        )


class SumPoolingTransform(BaseVariableTokenizer):
    """Sum-pool the available field tokens at every node."""

    def forward(self, x: torch.Tensor, variables: str | list[str]) -> torch.Tensor:
        tokens = self._embed_values(x, variables).sum(dim=-2)
        tokens = self.output_projection(self.normalize(tokens))
        return einops.rearrange(
            tokens,
            "batch time ensemble grid features -> (batch ensemble grid) (time features)",
        )


class MultiHeadAttentionTransform(BaseVariableTokenizer):
    """Attention-pool available fields in bounded grid chunks."""

    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        vocabulary: VariableVocabulary,
        num_heads: int = 8,
        chunk_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(emb_dim=emb_dim, out_dim=out_dim, vocabulary=vocabulary)
        if emb_dim % num_heads:
            raise ValueError("emb_dim must be divisible by num_heads")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        self.chunk_size = chunk_size
        self.attention = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True, **kwargs)
        self.query = nn.Parameter(torch.empty(1, 1, emb_dim))
        nn.init.xavier_uniform_(self.query)

    def _pool_chunk(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, time, ensemble, grid, _, _ = tokens.shape
        flat = einops.rearrange(
            tokens,
            "batch time ensemble grid fields features -> (batch time ensemble grid) fields features",
        )
        query = self.query.expand(flat.shape[0], -1, -1)
        pooled, _ = self.attention(query, flat, flat, need_weights=False)
        pooled = self.output_projection(self.normalize(pooled.squeeze(1)))
        return einops.rearrange(
            pooled,
            "(batch time ensemble grid) features -> batch time ensemble grid features",
            batch=batch,
            time=time,
            ensemble=ensemble,
            grid=grid,
        )

    def forward(self, x: torch.Tensor, variables: str | list[str]) -> torch.Tensor:
        tokens = self._embed_values(x, variables)
        chunks = []
        for chunk in tokens.split(self.chunk_size, dim=-3):
            pooled = (
                checkpoint(self._pool_chunk, chunk, use_reentrant=False) if self.training else self._pool_chunk(chunk)
            )
            chunks.append(pooled)
        pooled = torch.cat(chunks, dim=-2)
        return einops.rearrange(
            pooled,
            "batch time ensemble grid features -> (batch ensemble grid) (time features)",
        )
