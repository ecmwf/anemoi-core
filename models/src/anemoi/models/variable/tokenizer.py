# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import einops
import torch
from torch.utils.checkpoint import checkpoint

from anemoi.models.layers.embeddmetadata import EmbeddVariablesMetadata
from anemoi.models.variable.variablevocabular import VariableVocabulary


class VariableTokenizer(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        num_heads: int,
        vocabulary: VariableVocabulary,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        assert isinstance(emb_dim, int)
        assert isinstance(out_dim, int)
        assert isinstance(num_heads, int)
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.io = "input"
        self.embedd_variables = EmbeddVariablesMetadata(emb_dim=emb_dim, vocabulary=vocabulary)

        self.value_encoder = torch.nn.Linear(1, emb_dim)

        self.mha = torch.nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True, **kwargs)

        self.query = torch.nn.Parameter(torch.empty(1, 1, emb_dim))
        torch.nn.init.xavier_uniform_(self.query)
        self.output_projection = torch.nn.Linear(emb_dim, out_dim)

    def _forward_chunk(
        self,
        x_chunk: torch.Tensor,
        emb_variables: torch.Tensor,
    ) -> torch.Tensor:

        # [*, grid, vars] -> [*, grid, vars, emb]
        x_chunk = self.value_encoder(x_chunk.unsqueeze(-1))
        x_chunk = x_chunk + emb_variables

        # Each grid point becomes an independent attention batch
        x_chunk = einops.rearrange(
            x_chunk,
            "... grid variable embedding_dim " "-> (... grid) variable embedding_dim",
        )

        q = self.query.expand(
            x_chunk.shape[0],
            -1,
            -1,
        )

        attn, _ = self.mha(
            query=q,
            key=x_chunk,
            value=x_chunk,
            need_weights=False,
        )

        return self.output_projection(attn.squeeze(1))

    def forward(
        self,
        x: torch.Tensor,
        variables: str | list[str],
    ) -> torch.Tensor:

        if isinstance(variables, str):
            variables = [variables]

        emb_variables = self.embedd_variables(variables)

        chunk_size = 1024
        outputs = []

        for x_chunk in x.split(chunk_size, dim=-2):
            if self.training:
                out = checkpoint(
                    self._forward_chunk,
                    x_chunk,
                    emb_variables,
                    use_reentrant=False,
                )
            else:
                out = self._forward_chunk(
                    x_chunk,
                    emb_variables,
                )

            outputs.append(out)

        return torch.cat(outputs, dim=0)


class VariableDeTokenizer(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        in_dim: int,
        vocabulary: VariableVocabulary,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        assert isinstance(emb_dim, int)
        assert isinstance(in_dim, int)

        self.io = "output"
        self.embedd_variables = EmbeddVariablesMetadata(emb_dim=emb_dim, vocabulary=vocabulary)

        self.grid_projection = torch.nn.Linear(in_dim, emb_dim)
        self.variable_projection = torch.nn.Linear(emb_dim, emb_dim)
        self.variable_bias = torch.nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor, variables: str | list[str]) -> torch.Tensor:
        x = self.grid_projection(x)

        variables = self.embedd_variables(variables)

        queries = self.variable_projection(variables)

        # hadamard product between queries and x, then sum over the embedding dimension
        x = torch.einsum("ge,ve->gv", x, queries)
        x = x / queries.shape[-1] ** 0.5
        return x + self.variable_bias(variables).squeeze(-1)
