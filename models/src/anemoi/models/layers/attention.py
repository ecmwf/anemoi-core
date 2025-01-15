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
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

try:
    from flash_attn import flash_attn_func as attn_func
except ImportError:
    from torch.nn.functional import scaled_dot_product_attention as attn_func

    _FLASH_ATTENTION_AVAILABLE = False
else:
    _FLASH_ATTENTION_AVAILABLE = True


from anemoi.models.distributed.transformer import halo_exchange
from anemoi.models.distributed.transformer import remove_halos
from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence

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
        shard_strategy: str = "shard_heads",
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
        self.shard_strategy = shard_strategy

        self.lin_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.attention = attn_func

        if not _FLASH_ATTENTION_AVAILABLE:
            LOGGER.warning("Flash attention not available, falling back to pytorch scaled_dot_product_attention")

        if shard_strategy not in ["shard_heads", "shard_sequence"]:
            raise ValueError(f"Invalid shard_strategy: {shard_strategy}")

        if shard_strategy == "shard_sequence":  # remove this after PR #47 is merged (sliding window support)
            assert _FLASH_ATTENTION_AVAILABLE, "Flash attention is required for shard_sequence strategy"

        self.projection = nn.Linear(embed_dim, embed_dim, bias=True)

    def get_qkv_shard_sequence(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:
        assert (
            shapes[-1][0] // 2 >= self.window_size[0]
        ), f"Sharded sequence length ({shapes[-1][0]}) must be at least twice the window size (2*{self.window_size[0]})"

        # unpack grid dimension first to allow for halo exchange
        x = einops.rearrange(
            x,
            "(batch grid) channels -> batch grid channels",
            batch=batch_size,
        )

        # communicate halos (adds halos to x)
        x_plus_halos, halo_size_left, halo_size_right = halo_exchange(
            x, halo_size=self.window_size[0], mgroup=model_comm_group
        )

        query, key, value = self.lin_qkv(x_plus_halos).chunk(3, -1)

        query, key, value = (
            einops.rearrange(
                t,
                "batch grid (heads vars) -> batch heads grid vars",
                heads=self.num_heads,
            )
            for t in (query, key, value)
        )

        return query, key, value, halo_size_left, halo_size_right

    def get_qkv_shard_heads(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:
        query, key, value = self.lin_qkv(x).chunk(3, -1)

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

        return query, key, value

    def forward(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        if self.shard_strategy == "shard_sequence":
            query, key, value, halo_size_left, halo_size_right = self.get_qkv_shard_sequence(
                x, shapes, batch_size, model_comm_group
            )
        if self.shard_strategy == "shard_heads":
            query, key, value = self.get_qkv_shard_heads(x, shapes, batch_size, model_comm_group)

        dropout_p = self.dropout_p if self.training else 0.0

        if _FLASH_ATTENTION_AVAILABLE:
            query, key, value = (
                einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
            )
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

        if self.shard_strategy == "shard_sequence":
            out = remove_halos(out, halo_size_left, halo_size_right)
        if self.shard_strategy == "shard_heads":
            out = shard_sequence(out, shapes=shapes, mgroup=model_comm_group)

        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        out = self.projection(out)

        return out
