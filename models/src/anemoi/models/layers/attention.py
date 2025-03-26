# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import math
from typing import Optional

import einops
import torch
from packaging import version
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import PairTensor

from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence
from anemoi.models.layers.normalization import AutocastLayerNorm
from anemoi.utils.config import DotDict
from typing import Callable
from anemoi.models.layers.attention_flex import BlockMaskManager, BlockMask
LOGGER = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """Multi Head Self Attention Pytorch Layer

    allows for three different attention implementations:
    - scaled dot product attention, see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    - flash attention, see https://github.com/Dao-AILab/flash-attention
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        layer_kernels: DotDict,
        bias: bool = False,
        is_causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        attention_implementation: str = "flash_attention",
        softcap: Optional[float] = None,
        use_alibi_slopes: bool = False,
        use_qk_norm: bool = False,
        use_rotary_embeddings: bool = False,
        block_mask: Tensor | BlockMaskManager | BlockMask | None = None,
        
    ):
        """Initialize MultiHeadSelfAttention.

        For the flash attention implementation, two additional parameters are available: softcap, use_alibi_slopes

        softcap: Softcapping prevents the logits from growing excessively large

        use_alibi_slopes: Adds bias of `(-alibi_slope * |i + seqlen_k - seqlen_q - j|)` to the attention score of
        query i and key j, where alibi_slope is calculated using get_alibi_slopes

        Parameters
        ----------
        num_heads : int
            number of heads
        embed_dim : int
            embedding dimension
        bias : bool, optional
            bias, by default False
        is_causal : bool, optional
            apply causal attention mask, by default False
        window_size : Optional[int], optional
            window_size, by default None
        dropout_p : float, optional
            dropout probability, by default 0.0
        attention_implementation: str, optional
            A predefined string which selects which underlying attention
            implementation, by default "flash_attention"
        softcap : float, optional
            Anything > 0 activates softcapping attention, by default None
        use_alibi_slopes : bool, optional
            Adds bias
        """
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"

        self.attention_implementation = attention_implementation
        self.use_alibi_slopes = use_alibi_slopes

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads  # q k v
        self.window_size = window_size
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.softcap = softcap
        self.use_qk_norm = use_qk_norm
        self.use_rotary_embeddings = use_rotary_embeddings

        self.set_attention_function()

        if self.use_alibi_slopes:
            self.alibi_slopes = get_alibi_slopes(num_heads)
            assert self.alibi_slopes.shape[0] == num_heads, "Error: Number of alibi_slopes must match number of heads"
        else:
            self.alibi_slopes = None

        linear = layer_kernels["Linear"]
        # self.lin_qkv = linear(embed_dim, 3 * embed_dim, bias=bias)
        self.lin_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.lin_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.lin_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.projection = linear(embed_dim, embed_dim, bias=True)
        if self.use_qk_norm:
            self.q_norm = AutocastLayerNorm(self.head_dim, bias=False)
            self.k_norm = AutocastLayerNorm(self.head_dim, bias=False)

    def set_attention_function(self):
        attn_funcs = {
            "flash_attention": FlashAttentionWrapper,
            "scaled_dot_product_attention": SDPAAttentionWrapper,
            "flex_attention": FlexAttentionWrapper,
        }
        assert (
            self.attention_implementation in attn_funcs
        ), f"{self.attention_implementation} not supported. \
              Please change model.processor.attention_implementation to one of: {attn_funcs.keys()}"
        LOGGER.info(f"Using {self.attention_implementation}")

        # initalise the attn func here
        if self.attention_implementation == "flash_attention":
            self.attention = attn_funcs[self.attention_implementation](
                use_rotary_embeddings=self.use_rotary_embeddings, head_dim=self.head_dim
            )
        else:
            self.attention = attn_funcs[self.attention_implementation]()

    def attention_computation(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        shapes: list,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
        score_mod: Optional[Callable] = None,
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

        if self.use_qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        out = self.attention(
            query,
            key,
            value,
            batch_size,
            causal=False,
            window_size=self.window_size,
            dropout_p=dropout_p,
            softcap=self.softcap,
            alibi_slopes=self.alibi_slopes,
        )

        out = shard_sequence(out, shapes=shapes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        out = self.projection(out)

        return out

    def forward(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:

        query = self.lin_q(x)
        key = self.lin_k(x)
        value = self.lin_v(x)

        return self.attention_computation(query, key, value, shapes, batch_size, model_comm_group)


class SDPAAttentionWrapper(nn.Module):
    """Wrapper for Pytorch scaled dot product attention"""

    def __init__(self):
        super().__init__()

        from torch.nn.functional import scaled_dot_product_attention

        self.attention = scaled_dot_product_attention
        self.mask = None
        self.window_size = None

    def update_mask(self, seq_len, window_size: int, device: str):

        self.mask = (
            torch.abs(
                torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
            )
            <= window_size
        )

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal=False,
        window_size=None,
        dropout_p=0.0,
        softcap=None,
        alibi_slopes=None,
    ):
        if softcap is not None:
            NotImplementedError(
                "Softcap not supported by Pytorchs SDPA. please switch to flash attention or disable softcap."
            )
        if alibi_slopes is not None:
            NotImplementedError(
                "Alibi slopes not supported by Pytorchs SDPA. please switch to flash attention or disable alibi slopes."
            )

        sequence_len = query.shape[-2]

        if window_size is not None and (self.mask is None or tuple(self.mask.shape) != (sequence_len, sequence_len)):
            self.update_mask(sequence_len, window_size=window_size, device=query.device)

        with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
            out = self.attention(
                query,
                key,
                value,
                attn_mask=self.mask,
                is_causal=causal,
                dropout_p=dropout_p,
            )

        return out


class FlashAttentionWrapper(nn.Module):
    """Wrapper for Flash attention."""

    def __init__(self, use_rotary_embeddings: bool = False, head_dim: int = None):
        super().__init__()
        try:
            import flash_attn
        except ImportError:
            raise ImportError("Error: Flash-attn not installed. Please install flash-attn to use Flash Attention")

        if version.parse(flash_attn.__version__) < version.parse("2.6.0"):
            raise RuntimeError("Error: Flash-attn version is too low. Update to 2.6.0 or higher.")
        else:
            from flash_attn.layers.rotary import RotaryEmbedding

            self.attention = flash_attn.flash_attn_func

        self.use_rotary_embeddings = use_rotary_embeddings

        if self.use_rotary_embeddings:  # find alternative implementation
            self.rotary_emb = RotaryEmbedding(dim=head_dim)

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal: bool = False,
        window_size: int = None,
        dropout_p: float = 0.0,
        softcap: Optional[float] = None,
        alibi_slopes: torch.Tensor = None,
    ):
        query, key, value = (
            einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
        )

        alibi_slopes = alibi_slopes.repeat(batch_size, 1).to(query.device) if alibi_slopes is not None else None

        if self.use_rotary_embeddings:  # can this be done in a better way?
            key = key.unsqueeze(-3)
            value = value.unsqueeze(-3)
            keyvalue = torch.cat((key, value), dim=-3)
            query, keyvalue = self.rotary_emb(
                query, keyvalue, max_seqlen=max(keyvalue.shape[1], query.shape[1])
            )  # assumption seq const
            key = keyvalue[:, :, 0, ...]
            value = keyvalue[:, :, 1, ...]

        out = self.attention(
            query,
            key,
            value,
            causal=False,
            window_size=(window_size, window_size) if window_size is not None else (-1, -1),
            dropout_p=dropout_p,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
        )
        out = einops.rearrange(out, "batch grid heads vars -> batch heads grid vars")
        return out


class FlexAttentionWrapper(nn.Module):
    """Wrapper for Flex Attention mechanism."""

    def __init__(self, block_mask: Tensor | BlockMaskManager | BlockMask | None = None):
        super().__init__()
        try:
            from torch.nn.attention.flex_attention import flex_attention
        except ImportError:
            raise ImportError("Error: Flex attention not available in your PyTorch version. Please update PyTorch.")
            
        self.block_mask = block_mask
        self.flex_attention = flex_attention
        
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        batch_size: int,
        causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        softcap: Optional[float] = None,
        alibi_slopes: Optional[Tensor] = None,
        score_mod: Optional[Callable] = None,
    ) -> Tensor:
        """Apply flex attention to inputs.
        
        Parameters
        ----------
        query : Tensor
            Query tensor
        key : Tensor
            Key tensor
        value : Tensor
            Value tensor
        batch_size : int
            Batch size
        causal : bool, optional
            Whether to use causal attention, by default False
        window_size : Optional[int], optional
            Window size, by default None (unused in flex attention)
        dropout_p : float, optional
            Dropout probability, by default 0.0 (unused in flex attention)
        softcap : Optional[float], optional
            Softcap value, by default None (unused in flex attention)
        alibi_slopes : Optional[Tensor], optional
            Alibi slopes tensor, by default None (unused in flex attention)
        score_mod : Optional[Callable], optional
            Function to modify attention scores, by default None
            
        Returns
        -------
        Tensor
            Output tensor
        """
        if softcap is not None:
            LOGGER.warning("Softcap not supported in Flex Attention, ignoring.")
        
        if alibi_slopes is not None:
            LOGGER.warning("Alibi slopes not supported in Flex Attention, ignoring.")
            
        if window_size is not None:
            LOGGER.warning("Window size parameter not used in Flex Attention, ignoring.")
        
        if dropout_p > 0.0:
            LOGGER.warning("Dropout not supported in Flex Attention, ignoring.")
            
        # Get the block mask if we have a manager
        block_mask = None
        if isinstance(self.block_mask, BlockMaskManager):
            block_mask = self.block_mask.get_block_mask(query.device)
        elif isinstance(self.block_mask, (BlockMask, Tensor)):
            block_mask = self.block_mask
            
        # Apply flex attention
        return self.flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            score_mod=score_mod
        )

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


def get_alibi_slopes(num_heads: int) -> Tensor:
    """Calculates linearly decreasing slopes for alibi attention.

    Parameters
    ----------
    num_heads : int
        number of attention heads

    Returns
    -------
    Tensor
        aLiBi slopes
    """
    n = 2 ** math.floor(math.log2(num_heads))
    slope_0 = 2 ** (-8 / n)
    alibi_slopes = torch.pow(slope_0, torch.arange(1, 1 + n))
    if n < num_heads:
        slope_hat_0 = 2 ** (-4 / n)
        alibi_slopes_hat = torch.pow(slope_hat_0, torch.arange(1, 1 + 2 * (num_heads - n), 2))
        alibi_slopes = torch.cat([alibi_slopes, alibi_slopes_hat])
    return alibi_slopes
