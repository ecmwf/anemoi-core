# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import einops
import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import PairTensor

try:
    from flash_attn import flash_attn_func as attn_func
except ImportError:
    from flash_attn.layers.rotary import RotaryEmbedding
    from torch.nn.functional import scaled_dot_product_attention as attn_func

    _FLASH_ATTENTION_AVAILABLE = False
else:
    _FLASH_ATTENTION_AVAILABLE = True

from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence
from anemoi.models.layers.utils import AutocastLayerNorm

LOGGER = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """Multi Head Self Attention Pytorch Layer."""

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        bias: bool = False,
        is_causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        qk_norm: bool = False,
        rotary_embeddings: bool = False,
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads  # q k v
        self.window_size = (window_size, window_size)  # flash attention
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.qk_norm = qk_norm
        self.rotary_embeddings = rotary_embeddings

        self.lin_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.lin_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.lin_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attention = attn_func

        if not _FLASH_ATTENTION_AVAILABLE:
            LOGGER.warning("Flash attention not available, falling back to pytorch scaled_dot_product_attention")

        self.projection = nn.Linear(embed_dim, embed_dim, bias=True)

        if self.qk_norm:
            self.q_norm = AutocastLayerNorm(self.head_dim, bias=False)
            self.k_norm = AutocastLayerNorm(self.head_dim, bias=False)

        if self.rotary_embeddings:  # find alternative implementation
            assert _FLASH_ATTENTION_AVAILABLE, "Rotary embeddings require flash attention"
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

    def attention_computation(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        shapes: list,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        query, key, value = (
            einops.rearrange(
                t,
                "(batch grid) (heads vars) -> batch heads grid vars",
                batch=batch_size,
                heads=self.num_heads,
            )
            for t in (query, key, value)
        )
        query = shard_heads(query, shapes=shapes, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shapes, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shapes, mgroup=model_comm_group)
        dropout_p = self.dropout_p if self.training else 0.0

        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        if _FLASH_ATTENTION_AVAILABLE:
            query, key, value = (
                einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
            )
            if self.rotary_embeddings:  # can this be done in a better way?
                key = key.unsqueeze(-3)
                value = value.unsqueeze(-3)
                keyvalue = torch.cat((key, value), dim=-3)
                query, keyvalue = self.rotary_emb(
                    query, keyvalue, max_seqlen=max(keyvalue.shape[1], query.shape[1])
                )  # assumption seq const
                key = keyvalue[:, :, 0, ...]
                value = keyvalue[:, :, 1, ...]
            out = self.attention(query, key, value, causal=False, window_size=self.window_size, dropout_p=dropout_p)
            out = einops.rearrange(out, "batch grid heads vars -> batch heads grid vars")
        else:
            out = self.attention(
                query,
                key,
                value,
                is_causal=False,
                dropout_p=dropout_p,
            )  # expects (batch heads grid variable) format
        out = shard_sequence(out, shapes=shapes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")
        return self.projection(out)

    def forward(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:
        query = self.lin_q(x)
        key = self.lin_k(x)
        value = self.lin_v(x)
        return self.attention_computation(query, key, value, shapes, batch_size, model_comm_group)


class MultiHeadCrossAttention(MultiHeadSelfAttention):
    """Multi Head Cross Attention Pytorch Layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, x: PairTensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:
        query = self.lin_q(x[1])
        key = self.lin_k(x[0])
        value = self.lin_v(x[0])
        return self.attention_computation(query, key, value, shapes, batch_size, model_comm_group)