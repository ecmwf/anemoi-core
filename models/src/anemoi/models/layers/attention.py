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
from torch import where
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import PairTensor

from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """Multi Head Self Attention Pytorch Layer

    allows for three different attention implementations:
    - scaled dot product attention, see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    - flash attention, see https://github.com/Dao-AILab/flash-attention

    The config parameter "model.processor.attention_implementation" is used to control which attention implementation is used.

    "scaled_dot_product_attention" (SDPA)
        SDPA is a pytorch function, so it is easiest to use but the least performant.
        It runs on CPUs and GPUs.

    "flash_attention"
        Flash attention is optimised for efficient usage of the GPUs memory hierarchy. It loads smaller chunks
        into fast local memory, and fuses attention into a single kernel to reduce the passes through memory.
        It runs on Nvidia Ampere (e.g. A100) GPUs or newer and AMD MI200 GPUs or newer. Check the GitHub for
        the full requirements.
        You have to install flash attention yourself. If you are running on an x86 system, there are prebuilt
        wheels available on the GitHub repo. On an aarch64 system, you have to build flash attention from source.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        layer_kernels: DotDict,
        attn_channels: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        is_causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        attention_implementation: str = "flash_attention",
        softcap: Optional[float] = None,
        use_alibi_slopes: bool = False,
        use_rotary_embeddings: bool = False,
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
            Input and output embedding dimension
        attn_channels : int, optional
            Internal attention width used for q/k/v projections. If None,
            defaults to embed_dim.
        qkv_bias : bool, optional
            bias for querys, keys and values, by default False
        qk_norm : bool, optional
            normalize q and k, by default False
        is_causal : bool, optional
            apply causal attention mask, by default False
        window_size : Optional[int], optional
            window_size, by default None
        dropout_p : float, optional
            dropout probability, by default 0.0
        attention_implementation: str
            A predefined string which selects which underlying attention
            implementation, by default "flash_attention"
        softcap : float, optional
            Anything > 0 activates softcapping attention, by default None
        use_alibi_slopes : bool, optional
            Adds bias
        """
        super().__init__()

        self.attn_channels = embed_dim if attn_channels is None else attn_channels
        if self.attn_channels <= 0:
            raise ValueError(f"attn_channels must be > 0, got {self.attn_channels}")
        if self.attn_channels % num_heads != 0:
            raise ValueError(f"attn_channels ({self.attn_channels}) must be divisible by number of heads ({num_heads})")

        self.attention_implementation = attention_implementation
        self.use_alibi_slopes = use_alibi_slopes

        self.num_heads = num_heads
        self.head_dim = self.attn_channels // num_heads  # q k v
        self.window_size = window_size
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.qk_norm = qk_norm
        self.softcap = softcap
        self.use_rotary_embeddings = use_rotary_embeddings

        self.set_attention_function()

        if self.use_alibi_slopes:
            self.alibi_slopes = get_alibi_slopes(num_heads)
            assert self.alibi_slopes.shape[0] == num_heads, "Error: Number of alibi_slopes must match number of heads"
        else:
            self.alibi_slopes = None

        linear = layer_kernels.Linear
        self.lin_q = nn.Linear(embed_dim, self.attn_channels, bias=qkv_bias)
        self.lin_k = nn.Linear(embed_dim, self.attn_channels, bias=qkv_bias)
        self.lin_v = nn.Linear(embed_dim, self.attn_channels, bias=qkv_bias)

        self.projection = linear(self.attn_channels, embed_dim, bias=True)

        if self.qk_norm:
            self.q_norm = layer_kernels["QueryNorm"](self.head_dim)
            self.k_norm = layer_kernels["KeyNorm"](self.head_dim)

    def set_attention_function(self):
        attn_funcs = {
            "flash_attention": FlashAttentionWrapper,
            "scaled_dot_product_attention": SDPAAttentionWrapper,
        }
        assert self.attention_implementation in attn_funcs, f"{self.attention_implementation} not supported. \
              Please change model.processor.attention_implementation to one of: {attn_funcs.keys()}"

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

        out = shard_sequence(out, shapes=shapes, num_heads=self.num_heads, mgroup=model_comm_group)
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
    """Wrapper for Pytorch scaled dot product attention
    To use this attention implementation: model.processor.attention_implementation='scaled_dot_product_attention'
    """

    def __init__(self):
        super().__init__()

        from torch.nn.functional import scaled_dot_product_attention

        self.attention = scaled_dot_product_attention
        LOGGER.info("Using scaled_dot_product_attention.")

        self.attn_mask = None
        from torch.nn.attention.flex_attention import create_mask

        self.create_mask = create_mask

    def create_sliding_window_mask(self, B, H, Q_LEN, KV_LEN, window_size, device="cpu") -> Tensor:
        """Create a mask for sliding window attention compatible with SDPA.

        Parameters
        ----------
        B : int
            Batch size
        H : int
            Number of heads
        Q_LEN : int
            Query sequence length
        KV_LEN : int
            Key/value sequence length
        window_size : tuple
            Tuple of (left_window, right_window). Use -1 for unlimited.
        device : str
            Device for the mask tensor

        Returns
        -------
        Tensor
            2D attention mask
        """
        window_size_l = KV_LEN if window_size[0] == -1 else window_size[0]
        window_size_r = KV_LEN if window_size[1] == -1 else window_size[1]

        def sliding_window_mask(b, h, q_idx, kv_idx):
            l_mask = where(kv_idx <= q_idx, abs(q_idx - kv_idx) <= window_size_l, False)
            r_mask = where(q_idx <= kv_idx, abs(q_idx - kv_idx) <= window_size_r, False)
            return l_mask | r_mask

        # a mask for use with SDPA: tensor type, < 4D
        mask = self.create_mask(sliding_window_mask, B, H, Q_LEN, KV_LEN, device=device)
        mask = mask[0, 0, :, :]
        return mask

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
        if softcap is not None and softcap > 0:
            raise NotImplementedError(
                "Softcap not supported by Pytorchs SDPA. please switch to flash attention or disable softcap."
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "Alibi slopes not supported by Pytorchs SDPA. please switch to flash attention v2 or disable alibi slopes."
            )
        if window_size is not None and self.attn_mask is None:
            # build the attention mask for sliding window attention. We build the mask once and reuse it,
            # since it is the same for every forward pass (assuming the sequence length does not change).

            if isinstance(window_size, int):
                window_size = (window_size, window_size)
            self.attn_mask = self.create_sliding_window_mask(
                1, query.shape[1], query.shape[2], key.shape[2], window_size, device=query.device
            )

        out = self.attention(
            query,
            key,
            value,
            # self.attn_mask is None if global or causal attention is used, since SDPA will automatically apply a causal mask if causal=True. If window_size is used, we use the precomputed attn_mask.
            attn_mask=self.attn_mask,
            is_causal=False,
            dropout_p=dropout_p,
        )

        return out


class FlashAttentionWrapper(nn.Module):
    """Wrapper for Flash attention.

    Either flash attn v2 or flash attn v3 (optimised for hoppers and newer), based on
    what is installed.
    flash attention v3 does not support rotary embeddings or alibi slopes. To use these
    features, you should downgrade to flash attention v2.

    """

    def __init__(self, use_rotary_embeddings: bool = False, head_dim: int = None):
        super().__init__()

        flash_attn_func = self._import_flash_attn()

        self._init_rotary_embeddings(use_rotary_embeddings, head_dim)

        self.attention = flash_attn_func

    def _init_rotary_embeddings(self, use_rotary_embeddings: bool, head_dim: int) -> None:
        """Enables rotary embeddings if flash attention version is between 2.6.0 and 3."""
        self.use_rotary_embeddings = False
        if use_rotary_embeddings:
            if self.use_flash_attn_v4 or self.use_flash_attn_v3:
                raise RuntimeError(
                    "Rotary Embeddings not supported with flash attention v3 and v4. Please switch to flash attention v2 to use rotary embeddings."
                )

            # import flash attn v2 to check the version
            import flash_attn

            if flash_attn.__version__ <= version.parse("2.6"):
                raise RuntimeError("Rotary Embeddings not supported with flash attention v2 < v2.6.0")

            from flash_attn.layers.rotary import RotaryEmbedding

            self.use_rotary_embeddings = True
            self.rotary_emb = RotaryEmbedding(dim=head_dim)

    def _import_flash_attn(self) -> tuple:
        """imports either flash attention v2, v3 or v4, based on what is installed. prioritising v4, then v3, then v2. if none are installed, raises an error.

        returns:
            flash attention function
        """
        # will be set to a valid version if either flash attention v2, v3 or v4 is successfully imported
        flash_attn_func = None

        self.use_flash_attn_v3 = False
        self.use_flash_attn_v4 = False

        e_v4 = None
        e_v3 = None
        e_v2 = None

        try:
            from flash_attn.cute import flash_attn_func

            LOGGER.info("Using flash attention v4")
            self.use_flash_attn_v4 = True
            return flash_attn_func
        except ImportError as e:
            e_v4 = e
            LOGGER.debug(f"Flash attention v4 not available: {e_v4}")

        try:
            from flash_attn_interface import flash_attn_func

            LOGGER.info("Using flash attention v3")
            self.use_flash_attn_v3 = True
            return flash_attn_func
        except ImportError as e:
            e_v3 = e
            LOGGER.debug(f"Flash attention v3 not available: {e_v3}")
        try:
            from flash_attn import flash_attn_func

            LOGGER.info("Using flash attention v2")
            return flash_attn_func
        except ImportError as e:
            e_v2 = e
            LOGGER.debug(f"Flash attention v2 not available: {e_v2}")

        raise ImportError(
            "Flash attention is not installed. Please install flash attention v4, v3 or v2 to use this attention implementation. "
            f"Attempted imports resulted in the following errors: "
            f"v4 import error: {e_v4} "
            f"v3 import error: {e_v3} "
            f"v2 import error: {e_v2} "
        )

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        softcap: Optional[float] = None,
        alibi_slopes: torch.Tensor = None,
    ):
        query, key, value = (
            einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
        )

        if alibi_slopes is not None and self.use_flash_attn_v3:
            raise NotImplementedError(
                "Alibi slopes is currently not supported by flash attention v3. please switch to flash attention v2 or disable alibi slopes."
            )

        alibi_slopes = alibi_slopes.repeat(batch_size, 1).to(query.device) if alibi_slopes is not None else None

        if self.use_rotary_embeddings:
            key = key.unsqueeze(-3)
            value = value.unsqueeze(-3)
            keyvalue = torch.cat((key, value), dim=-3)
            query, keyvalue = self.rotary_emb(
                query, keyvalue, max_seqlen=max(keyvalue.shape[1], query.shape[1])
            )  # assumption seq const
            key = keyvalue[:, :, 0, ...]
            value = keyvalue[:, :, 1, ...]

        if self.use_flash_attn_v4:
            out = self.attention(
                query,
                key,
                value,
                softmax_scale=1.0 / math.sqrt(query.shape[-1]),
                causal=False,
                window_size=(window_size, window_size) if window_size is not None else (-1, -1),
            )[0]
        elif self.use_flash_attn_v3:
            out = self.attention(
                query,
                key,
                value,
                causal=False,
                window_size=(window_size, window_size) if window_size is not None else (-1, -1),
                softcap=softcap,
            )
            if isinstance(out, tuple):
                out = out[
                    0
                ]  # early versions of flash attention v3 returns a tuple with '(out, softmax_lse)'. here we drop to 'out'
        else:
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
