# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import abstractmethod
from typing import Any
from typing import Optional

import einops
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from anemoi.models.layers.input_transform import BaseInputTransform
from anemoi.models.variable.embedmetadata import EmbedMetadata
from anemoi.models.variable.variablevocabular import VariableVocabulary


class BaseVariableTokenizer(BaseInputTransform):
    def __init__(self, emb_dim: int, out_dim: int, variable_metadata: Optional[dict] = None, **kwargs: Any) -> None:

        super().__init__(variable_metadata)

        if not variable_metadata:
            raise ValueError(f"{type(self).__name__} needs the variables' metadata to build its vocabulary.")
        vocabulary = VariableVocabulary.from_foundation(variable_metadata)

        assert isinstance(emb_dim, int), f"expecting emb_dim to be int, got {type(emb_dim)}"
        assert isinstance(out_dim, int), f"expecting out_dim to be int, got {type(out_dim)}"

        assert isinstance(
            vocabulary, VariableVocabulary
        ), f"vocabulary must be an object created for VariableVocabulary class, got {type(vocabulary)}"
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.output_dim = out_dim

        self.emb_variables_metadata = EmbedMetadata(vocabulary=vocabulary, emb_dim=self.emb_dim)
        self.value_encoder = nn.Linear(1, self.emb_dim)
        self.normalize = nn.LayerNorm(self.emb_dim)
        self.output_projection = nn.Linear(self.emb_dim, self.out_dim)

    @abstractmethod
    def forward(
        self, x: torch.Tensor, node_attributes_data: torch.Tensor, variables: str | list[str]
    ) -> torch.Tensor: ...

    def _to_encoder_input(self, x_vars: torch.Tensor, node_attributes_data: torch.Tensor) -> torch.Tensor:
        # Pooling tokenizers keep the time axis; the encoder expects it folded into the features.
        if x_vars.ndim == 5:
            x_vars = einops.rearrange(x_vars, "batch time ensemble grid emb -> (batch ensemble grid) (time emb)")
        return self.append_node_attributes(x_vars, node_attributes_data)


class MultiHeadAttentionTransform(BaseVariableTokenizer):
    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        variable_metadata: Optional[dict] = None,
        num_heads: int = 8,
        chunk_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(emb_dim=emb_dim, out_dim=out_dim, variable_metadata=variable_metadata)

        self.num_heads = num_heads
        self.chunk_size = chunk_size

        assert isinstance(self.num_heads, int), f"num_heads has to be of type int, got {type(self.num_heads)}"
        assert isinstance(self.chunk_size, int), f"chunk_size has to be of type int, got {type(self.chunk_size)}"

        assert (
            emb_dim % num_heads == 0
        ), f"emb_dim and num_heads needs to be divisble, got emb_dim: {emb_dim} and num_heads: {num_heads}"

        self.mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=self.num_heads, batch_first=True, **kwargs)

        self.query = nn.Parameter(torch.empty(1, 1, emb_dim))
        nn.init.xavier_uniform_(self.query)

    def _forward_chunk(self, x_chunk: torch.Tensor, emb_variables: torch.Tensor) -> torch.Tensor:
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
        attn = attn.squeeze(1)
        attn = self.normalize(attn)

        return self.output_projection(attn)

    # def forward(self, x: torch.Tensor, variables: str | list[str]) -> torch.Tensor:
    #     if isinstance(variables, str):
    #         variables = [variables]

    #     emb_variables = self.emb_variables_metadata(variables)
    #     # maybe some einops ops here for getting the correct shape before mha
    #     outputs = []
    #     batch, time, ensemble, grid, num_vars = x.shape

    #     for x_chunk in x.split(self.chunk_size, dim=-2):
    #         if self.training:
    #             out = checkpoint(self._forward_chunk, x_chunk, emb_variables, use_reentrant=False)
    #         else:
    #             out = self._forward_chunk(
    #                 x_chunk,
    #                 emb_variables,
    #             )
    #         outputs.append(out)
    #     outputs = torch.cat(outputs, dim=0)

    #     outputs = einops.rearrange(
    #         outputs,
    #         "(batch time ensemble grid) embedding_dim " "-> (batch ensemble grid) (time embedding_dim)",
    #         batch=batch,
    #         time=time,
    #         ensemble=ensemble,
    #         grid=grid,
    #     )

    #     return outputs
    def forward(
        self,
        x: torch.Tensor,
        node_attributes_data: torch.Tensor,
        variables: str | list[str],
    ) -> torch.Tensor:
        if isinstance(variables, str):
            variables = [variables]

        emb_variables = self.emb_variables_metadata(variables)

        batch, time, ensemble, grid, num_vars = x.shape

        outputs = []

        for x_chunk in x.split(self.chunk_size, dim=-2):
            if self.training:
                out = checkpoint(
                    self._forward_chunk,
                    x_chunk,
                    emb_variables,
                    use_reentrant=False,
                )
            else:
                out = self._forward_chunk(x_chunk, emb_variables)

            # Restore the grid dimension before concatenating chunks.
            out = einops.rearrange(
                out,
                "(batch time ensemble grid) embedding -> batch time ensemble grid embedding",
                batch=batch,
                time=time,
                ensemble=ensemble,
                grid=x_chunk.shape[-2],
            )

            outputs.append(out)

        outputs = torch.cat(outputs, dim=-2)

        outputs = einops.rearrange(
            outputs,
            "batch time ensemble grid embedding -> (batch ensemble grid) (time embedding)",
        )
        return self._to_encoder_input(outputs, node_attributes_data)


class MeanPoolingTransform(BaseVariableTokenizer):
    def __init__(self, emb_dim: int, out_dim: int, variable_metadata: Optional[dict] = None, **kwargs: Any) -> None:
        super().__init__(emb_dim=emb_dim, out_dim=out_dim, variable_metadata=variable_metadata)

    def forward(self, x: torch.Tensor, node_attributes_data: torch.Tensor, variables: str | list[str]) -> torch.Tensor:
        # shape [B, E, G, VAR]
        if isinstance(variables, str):
            variables = [variables]

        emb_variables = self.emb_variables_metadata(variables)

        x = self.value_encoder(x.unsqueeze(-1))
        # shape [B,E,G,VAR,EMB_DIM]

        x = x + emb_variables
        x = x.mean(dim=-2)
        x = self.normalize(x)
        # shape [B, E, G, EMB_DIM]

        return self._to_encoder_input(self.output_projection(x), node_attributes_data)


class SumPoolingTransform(BaseVariableTokenizer):
    def __init__(self, emb_dim: int, out_dim: int, variable_metadata: Optional[dict] = None, **kwargs: Any) -> None:
        super().__init__(emb_dim=emb_dim, out_dim=out_dim, variable_metadata=variable_metadata)

    def forward(self, x: torch.Tensor, node_attributes_data: torch.Tensor, variables: str | list[str]) -> torch.Tensor:
        # shape [B, E, G, VAR]

        if isinstance(variables, str):
            variables = [variables]
        emb_variables = self.emb_variables_metadata(variables)

        x = self.value_encoder(x.unsqueeze(-1))
        # shape [B,E,G,VAR,EMB_DIM]

        x = x + emb_variables
        x = x.sum(dim=-2)

        x = self.normalize(x)
        # shape [B, E, G, EMB_DIM]

        return self._to_encoder_input(self.output_projection(x), node_attributes_data)
