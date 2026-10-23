# (C) Copyright 2024-2026 Anemoi contributors.
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
import os
from typing import Optional
from typing import Union

import einops
import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import PairTensor

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.attention_wrapper import AttentionWrapper
from anemoi.models.layers.neighbourhood_attention import Neighbourhood
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodBand
from anemoi.models.layers.spherical_rotary import SphericalRotaryEmbedding
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

# Change attention implementation during inference runtime
ATTENTION_BACKEND = os.environ.get("ANEMOI_INFERENCE_TRANSFORMER_ATTENTION_BACKEND", "")


class MultiHeadSelfAttention(nn.Module):
    """Multi Head Self Attention Pytorch Layer

    allows for three different attention implementations:
    - scaled dot product attention, see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    - Triton attention, the flash attention algorithm written in Triton, see :mod:`anemoi.models.triton.attention`
    - neighbourhood attention on a global grid, see :mod:`anemoi.models.layers.neighbourhood_attention`

    The config parameter "model.processor.attention_implementation" is used to control which attention implementation is used.

    "scaled_dot_product_attention" (SDPA)
        SDPA is a pytorch function. It runs on CPUs and GPUs. A sliding window becomes a mask of one
        byte per query-key pair.

    With both, a sliding window needs queries and keys of the same length: position i of one is
    position i of the other. Cross attention between sequences of different lengths attends to every key.

    "triton_attention"
        The flash attention algorithm written in Triton, for GPUs. It does self attention and cross
        attention between sequences of any length, and sliding windows but not dropout.

    "neighbourhood"
        Each query attends only to the keys around it on a registered grid family, such as octahedral
        or HEALPix grids, or the refined icosahedron. The grids and the neighbourhood size come from ``neighbourhood``. The
        ANEMOI_INFERENCE_TRANSFORMER_ATTENTION_BACKEND environment variable does not change it, as the
        other implementations attend to different keys.
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
        attention_implementation: str = "scaled_dot_product_attention",
        neighbourhood: Optional[Neighbourhood] = None,
        rotary: Optional[SphericalRotaryEmbedding] = None,
    ):
        """Initialize MultiHeadSelfAttention.

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
            implementation, by default "scaled_dot_product_attention"
        neighbourhood : Neighbourhood, optional
            The grids and neighbourhood of the "neighbourhood" implementation (see
            :func:`anemoi.models.layers.neighbourhood_attention.build_grid_neighbourhood`)
        rotary : SphericalRotaryEmbedding, optional
            Rotary position embeddings that turn queries and keys by the positions of their nodes,
            shared by the attention layers of a model component; works with every implementation
        """
        super().__init__()

        self.attn_channels = embed_dim if attn_channels is None else attn_channels
        if self.attn_channels <= 0:
            raise ValueError(f"attn_channels must be > 0, got {self.attn_channels}")
        if self.attn_channels % num_heads != 0:
            raise ValueError(f"attn_channels ({self.attn_channels}) must be divisible by number of heads ({num_heads})")

        self.attention_implementation = attention_implementation
        self._attention_backend_applied = False

        self.num_heads = num_heads
        self.head_dim = self.attn_channels // num_heads  # q k v
        self.window_size = window_size
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.qk_norm = qk_norm
        self.neighbourhood = neighbourhood
        self.rotary = rotary

        self.set_attention_function()

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
            "scaled_dot_product_attention": SDPAAttentionWrapper,
            "triton_attention": TritonAttentionWrapper,
        }

        # Check if 'ANEMOI_INFERENCE_TRANSFORMER_ATTENTION_BACKEND' env var has been set
        if ATTENTION_BACKEND and self.attention_implementation != "neighbourhood":
            if ATTENTION_BACKEND == self.attention_implementation:
                # Attention backend has already been updated, return early
                return
            LOGGER.info(
                "'ANEMOI_INFERENCE_TRANSFORMER_ATTENTION_BACKEND' environment variable has been set. Overwriting attention backend from '%s' to '%s'",
                self.attention_implementation,
                ATTENTION_BACKEND,
            )
            self.attention_implementation = ATTENTION_BACKEND

        implementations = [*attn_funcs, "neighbourhood"]
        assert (
            self.attention_implementation in implementations
        ), f"backend '{self.attention_implementation}' not supported. \
              Please change model.processor.attention_implementation to one of: {implementations}"

        # initalise the attn func here
        if self.attention_implementation == "neighbourhood":
            if self.neighbourhood is None:
                raise ValueError(
                    "The 'neighbourhood' attention implementation needs a neighbourhood, see build_grid_neighbourhood."
                )
            self.attention = self.neighbourhood.attention_module()
        else:
            self.attention = attn_funcs[self.attention_implementation]()
        self.attention.check_supported(dropout_p=self.dropout_p > 0, window_size=self.window_size is not None)
        self.attention.check_head_dim(self.head_dim)

    def attention_computation(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        grid_shard_sizes: Union[ShardSizes, tuple[ShardSizes, ShardSizes]],
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

        # Shard heads: split along heads (dim -3), gather along sequence/grid (dim -2)
        q_shard_sizes = grid_shard_sizes[1] if isinstance(grid_shard_sizes, tuple) else grid_shard_sizes
        kv_shard_sizes = grid_shard_sizes[0] if isinstance(grid_shard_sizes, tuple) else grid_shard_sizes
        head_shard_sizes = get_shard_sizes(query, -3, model_comm_group)

        query = all_to_all_transpose(query, -3, head_shard_sizes, -2, q_shard_sizes, model_comm_group)
        key, value = (
            all_to_all_transpose(t, -3, head_shard_sizes, -2, kv_shard_sizes, model_comm_group) for t in (key, value)
        )

        dropout_p = self.dropout_p if self.training else 0.0

        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        # Every node is present here (the heads are split across GPUs, not the nodes), in node order.
        if self.rotary is not None:
            query, key = self.rotary(query, key)

        out = self.attention(
            query,
            key,
            value,
            batch_size,
            causal=False,
            window_size=self.window_size,
            dropout_p=dropout_p,
        )

        # Shard sequence: split along sequence/grid (dim -2), gather along heads (dim -3)
        out = all_to_all_transpose(out, -2, q_shard_sizes, -3, head_shard_sizes, model_comm_group)

        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        out = self.projection(out)

        return out

    def forward(
        self,
        x: Tensor,
        grid_shard_sizes: GraphShardInfo,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:

        query = self.lin_q(x)
        key = self.lin_k(x)
        value = self.lin_v(x)

        # Check once at runtime if the Attention backend env var has been set, and update attention backend accordingly
        if (
            ATTENTION_BACKEND
            and not self._attention_backend_applied
            and self.attention_implementation != "neighbourhood"
        ):
            self.set_attention_function()
            self._attention_backend_applied = True

        return self.attention_computation(query, key, value, grid_shard_sizes.nodes, batch_size, model_comm_group)

    def attend_band(self, query: Tensor, key: Tensor, value: Tensor, band: NeighbourhoodBand) -> Tensor:
        """Neighbourhood attention for one band of queries, followed by the output projection.

        Does for one band what :meth:`attention_computation` does for the whole grid, on a single
        GPU: the same normalisation, rotary turn, attention and projection of every point.

        Parameters
        ----------
        query : Tensor
            Projected queries of the band, shape ``(batch, band query points, channels)``.
        key, value : Tensor
            Projected keys and values of the key rows the band reaches, shape
            ``(batch, band key points, channels)``.
        band : NeighbourhoodBand
            The band, with its own neighbourhood attention.

        Returns
        -------
        Tensor
            Attention output of the band, shape ``(batch, band query points, embed_dim)``.
        """
        query, key, value = (
            einops.rearrange(t, "batch grid (heads vars) -> batch heads grid vars", heads=self.num_heads)
            for t in (query, key, value)
        )
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if self.rotary is not None:
            query = self.rotary.turn_queries(query, band.query_points)
            key = self.rotary.turn_keys(key, band.key_points)

        out = band.attention(
            query,
            key,
            value,
            query.shape[0],
            window_size=self.window_size,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = einops.rearrange(out, "batch heads grid vars -> batch grid (heads vars)")
        return self.projection(out)

    def forward_band(self, x: Tensor, band: NeighbourhoodBand) -> Tensor:
        """Self attention for one band of points.

        Parameters
        ----------
        x : Tensor
            Input of the band's key points, which include its query points, shape
            ``(batch, band key points, embed_dim)``.
        band : NeighbourhoodBand
            The band.

        Returns
        -------
        Tensor
            Attention output of the band's query points, shape ``(batch, band query points, embed_dim)``.
        """
        queries = x[:, band.queries_in_keys]
        return self.attend_band(self.lin_q(queries), self.lin_k(x), self.lin_v(x), band)


def check_window_lengths(query: Tensor, key: Tensor, window_size: Optional[int]) -> None:
    """Raises if a sliding window is asked for between sequences of different lengths."""
    if window_size is not None and query.shape[-2] != key.shape[-2]:
        raise ValueError(
            "A sliding window needs queries and keys of the same length, as position i of one is "
            f"position i of the other; got {query.shape[-2]} queries and {key.shape[-2]} keys. Use "
            "window_size: null for global cross attention, or neighbourhood attention for local "
            "cross attention between two grids."
        )


class SDPAAttentionWrapper(AttentionWrapper):
    """Wrapper for Pytorch scaled dot product attention
    To use this attention implementation: model.processor.attention_implementation='scaled_dot_product_attention'
    """

    def __init__(self):
        super().__init__()

        from torch.nn.functional import scaled_dot_product_attention

        self.attention = scaled_dot_product_attention
        LOGGER.info("Using scaled_dot_product_attention.")

        self.attn_mask = None

    @staticmethod
    def create_sliding_window_mask(num_queries: int, num_keys: int, window_size: int, device="cpu") -> Tensor:
        """Mask of the keys within ``window_size`` positions of each query, on either side.

        Parameters
        ----------
        num_queries, num_keys : int
            Number of queries and keys.
        window_size : int
            Number of positions a query reaches on each side.
        device : str
            Device for the mask tensor.

        Returns
        -------
        Tensor
            Boolean mask of shape ``(num_queries, num_keys)``, True where the query attends to the key.
        """
        query_idx = torch.arange(num_queries, device=device)[:, None]
        key_idx = torch.arange(num_keys, device=device)[None, :]
        return (query_idx - key_idx).abs() <= window_size

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal=False,
        window_size=None,
        dropout_p=0.0,
    ):
        check_window_lengths(query, key, window_size)
        if window_size is not None and self.attn_mask is None:
            # The mask is the same for every forward pass as long as the sequence lengths do not
            # change, so it is built once and kept.
            self.attn_mask = self.create_sliding_window_mask(
                query.shape[2], key.shape[2], window_size, device=query.device
            )

        out = self.attention(
            query,
            key,
            value,
            # No mask for global attention; the sliding window mask otherwise.
            attn_mask=self.attn_mask,
            is_causal=False,
            dropout_p=dropout_p,
        )

        return out


class TritonAttentionWrapper(AttentionWrapper):
    """Wrapper for Anemoi Triton attention, the flash attention algorithm written in Triton.

    Queries and keys may differ in number, as in the transformer mappers.
    """

    unsupported = frozenset({"dropout_p"})

    def __init__(self):
        super().__init__()

        # Helper function to check if triton is available
        # Prevents strange errors from importing triton functions on unsupported systems
        from anemoi.models.triton.utils import is_triton_available

        if not is_triton_available():
            raise ImportError(
                "Triton is not supported on your system. Either it is not installed or no GPUs are available"
            )

        from anemoi.models.triton.attention import TritonAttention

        self.attention = TritonAttention

    def check_head_dim(self, head_dim: int) -> None:
        if head_dim not in (16, 32, 64, 128, 256):
            raise ValueError(f"{self.describe()} needs heads of 16, 32, 64, 128 or 256 channels, got {head_dim}.")

    def forward(
        self,
        query,
        key,
        value,
        batch_size: int,
        causal: bool = False,
        window_size: int = None,
        dropout_p: float = 0.0,
    ):
        check_window_lengths(query, key, window_size)
        softmax_scale = 1 / math.sqrt(query.size(-1))

        out = self.attention.apply(query, key, value, causal, window_size, softmax_scale).to(query.dtype)

        return out


class MultiHeadCrossAttention(MultiHeadSelfAttention):
    """Multi Head Cross Attention Pytorch Layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: PairTensor,
        shard_info: BipartiteGraphShardInfo,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        query = self.lin_q(x[1])
        key = self.lin_k(x[0])
        value = self.lin_v(x[0])

        shard_sizes = (shard_info.src_nodes, shard_info.dst_nodes)

        return self.attention_computation(query, key, value, shard_sizes, batch_size, model_comm_group)

    def forward_band(self, x: PairTensor, band: NeighbourhoodBand) -> Tensor:
        """Cross attention for one band of queries.

        Parameters
        ----------
        x : PairTensor
            Input of the band's key points, shape ``(batch, band key points, embed_dim)``, and of its
            query points, shape ``(batch, band query points, embed_dim)``.
        band : NeighbourhoodBand
            The band.

        Returns
        -------
        Tensor
            Attention output of the band's query points, shape ``(batch, band query points, embed_dim)``.
        """
        return self.attend_band(self.lin_q(x[1]), self.lin_k(x[0]), self.lin_v(x[0]), band)


class PointwiseMultiHeadCrossAttention(nn.Module):
    """Attend over source tokens independently at each hidden node."""

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        layer_kernels: DotDict,
        attn_channels: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_channels = embed_dim if attn_channels is None else attn_channels
        if self.attn_channels % num_heads != 0:
            raise ValueError(
                f"attn_channels ({self.attn_channels}) must be divisible by number of heads ({num_heads}).",
            )

        self.num_heads = num_heads
        self.head_dim = self.attn_channels // num_heads
        self.dropout_p = dropout_p
        self.qk_norm = qk_norm

        linear = layer_kernels.Linear
        self.lin_q = linear(embed_dim, self.attn_channels, bias=qkv_bias)
        self.lin_k = linear(embed_dim, self.attn_channels, bias=qkv_bias)
        self.lin_v = linear(embed_dim, self.attn_channels, bias=qkv_bias)
        self.projection = linear(self.attn_channels, embed_dim, bias=True)

        if qk_norm:
            self.q_norm = layer_kernels.QueryNorm(self.head_dim)
            self.k_norm = layer_kernels.KeyNorm(self.head_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply cross-attention over source tokens at each hidden node."""
        query = einops.rearrange(self.lin_q(query), "grid (heads vars) -> grid heads vars", heads=self.num_heads)
        key, value = (
            einops.rearrange(tensor, "grid sources (heads vars) -> grid heads sources vars", heads=self.num_heads)
            for tensor in (self.lin_k(key), self.lin_v(value))
        )

        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        # Score each source against the node's query: (g,h,s,v) @ (g,h,v,1) -> (g,h,s)
        scores = (key @ query.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.head_dim)
        # Turn the scores into weights over the sources that sum to 1 at each node and head.
        weights = nn.functional.dropout(scores.softmax(dim=-1), p=self.dropout_p, training=self.training)
        # Weighted average of the source values: (g,h,1,s) @ (g,h,s,v) -> (g,h,v)
        output = (weights.unsqueeze(-2) @ value).squeeze(-2)
        return self.projection(einops.rearrange(output, "grid heads vars -> grid (heads vars)"))
