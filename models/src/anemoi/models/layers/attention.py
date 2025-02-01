# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional, Union, Tuple

import einops
import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import PairTensor
import os
import numpy as np



from flash_attn import flash_attn_func as attn_func
from flash_attn.layers.rotary import RotaryEmbedding
from torch_geometric.data import HeteroData
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention
from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence
from anemoi.models.layers.utils import AutocastLayerNorm

from sklearn.neighbors import NearestNeighbors

LOGGER = logging.getLogger(__name__)

torch._dynamo.config.cache_size_limit = 8192
torch._dynamo.config.accumulated_cache_size_limit = 8192
torch._dynamo.config.fail_on_cache_limit_hit = False  # Turn on true for debugging
torch._dynamo.config.optimize_ddp = os.getenv("TORCH_DYNAMO_OPTIMIZE_DDP", "ddp_optimizer")

torch._inductor.config.mixed_mm_choice = "triton"  # removed ATEN_

torch._inductor.config.max_autotune = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1"
torch._inductor.config.max_autotune_gemm = True
torch._inductor.config.max_autotune_gemm_backends = os.getenv("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "TRITON,CPP")
torch._inductor.config.max_autotune_gemm_search_space = "DEFAULT"  # "EXHAUSTIVE"
torch._inductor.config.max_autotune_pointwise = os.getenv("TUNEALL") == "1"
torch._inductor.config.max_fusion_size = int(os.getenv("TORCHINDUCTOR_MAX_FUSION_SIZE", "64"))


# torch._inductor.config.autotune_num_choices_displayed = 3

flex_attention = torch.compile(
    flex_attention,
    dynamic=False,
    fullgraph=os.getenv("FULL_GRAPH", "1") == "1",
    #    mode="max-autotune",
    #    options={"shape_padding": True}
)

# create_block_mask = torch.compile(create_block_mask, dynamic=True, fullgraph=True) #https://twitter.com/cHHillee/status/1851418255749169419
create_block_mask = torch.compile(
    create_block_mask,
    dynamic=False,
    fullgraph=os.getenv("FULL_GRAPH", "1") == "1",
    #    mode="max-autotune",
    #    options={"shape_padding": True}
)  # https://twitter.com/cHHillee/status/1851418255749169419ate

LOGGER = logging.getLogger(__name__)

class BlockMaskManager(nn.Module):

    scaling_method = {
        "haversine": "constant_span_relative_to_grid_size",
        "knn_haversine": "scale_span_relative_to_grid_size",
        "window": "scale_span_relative_to_grid_size",
    }

    def __init__(
        self,
        graph: HeteroData,
        query_grid_name: str,
        keyvalue_grid_name: str,
        devices: list[str | torch.device] = None,
        attention_span: Optional[int] = None,
        base_attention_span: Optional[int] = None,
        method: str = "knn_haversine",
        base_grid: str = "query",
        block_size: Union[int, Tuple[int, int]] = torch.nn.attention.flex_attention._DEFAULT_SPARSE_BLOCK_SIZE,
        **kwargs,
    ):

        super().__init__()

        self.graph = graph

        self.map_device_block_mask: dict[torch.device, torch.nn.attention.BlockMask] = {}

        assert method in ["knn_adjacency_matrix", "knn_index_matrix", "haversine", "knn_haversine", "window"]
        self.method = method

        assert attention_span is not None or base_attention_span is not None

        if attention_span is not None:
            self.attention_span = attention_span
        else:

            self.attention_span = calculate_scaled_attention_attention_spans(
                base_attention_span,
                base_grid_name=kwargs.get("base_attention_span_grid", None),
                target_grid_name=query_grid_name if base_grid == "query" else keyvalue_grid_name,
                scaling_method=self.scaling_method[self.method],
                _graph_data=self.graph,
                method=self.method,
            )

        # Attn span is an int for the knn based methods
        # Attn span is a map for the latlon proximity method, where the keys will represent the dimension and the values will represent the max distance in degrees

        self.base_grid = base_grid  # NOTE: this is not used in the latlon proximity method yet

        self.query_grid_name = query_grid_name
        self.keyvalue_grid_name = keyvalue_grid_name

        self.devices = devices
        # FIX: THE ISSUE WITH CURRENT METHODOLOGY IS THAT EACH GPU will have a copy of data needed for all other GPUS as well as its own data.
        # FIX: This by including a hook that ensures the necessary tensors are moved to the correct device when to() is called

        self.block_size = block_size

        self.earth_radius = 6371.0

        self.setup_attn_mask()

        self.setup_mask_mod()

    def setup_attn_mask(self):

        if self.method == "window":
            self.map_attention_span = {}

        elif self.method == "knn_index_matrix":
            self.knn_index_matrix = self.setup_knn_index_matrix()

            self.map_knn_index_matrix = {}

        elif self.method == "knn_adjacency_matrix":
            self.connectivity_matrix = self.setup_connectivity_matrix()

            self.map_connectivity_matrix = {}

        elif self.method in ["haversine", "knn_haversine"]:

            self.map_query_grid_lat = {}
            self.map_query_grid_lon = {}
            self.map_keyvalue_grid_lat = {}
            self.map_keyvalue_grid_lon = {}
            self.map_earth_radius = {}

            if self.method == "haversine":
                self.map_attention_span = {
                    device: torch.as_tensor(self.attention_span).to(device) for device in self.devices
                }

                self.map_attention_span = {}

            if self.method == "knn_haversine":
                # So here we implement knn attention in a way that does not utilize the any form of reduction which causes CUDA ERRORS
                # Step 1: For each node on the source grid and target grid we retrieve the latlon coordinates
                # Step 2: We then calculate the haversine distance between each node in source and the n'th closest node in target
                # Step 3: Then we hold these tensors on the appropriate device
                self.haversine_distance_of_nth_closest_node = self.setup_haversine_distance_of_nth_closest_node()

                self.map_haversine_distance_of_nth_closest_node = {}

        else:
            raise ValueError(f"Method {self.method} not supported")

    def copy_to_device(self, device: torch.device):
        if self.method == "window":
            self.map_attention_span[device] = torch.as_tensor(self.attention_span).to(device)

        elif self.method == "knn_index_matrix":

            self.map_knn_index_matrix[device] = self.knn_index_matrix.to(device)

        elif self.method == "knn_adjacency_matrix":

            self.map_connectivity_matrix[device] = self.connectivity_matrix.to(device)

        elif self.method in ["haversine", "knn_haversine"]:
            query_grid_latlon = self.get_grid_latlon(self.query_grid_name)

            self.map_query_grid_lat[device] = query_grid_latlon[:, 0].to(device).contiguous()
            self.map_query_grid_lon[device] = query_grid_latlon[:, 1].to(device).contiguous()
            self.map_earth_radius[device] = torch.as_tensor(self.earth_radius).to(device)

            if self.query_grid_name == self.keyvalue_grid_name:
                self.map_keyvalue_grid_lat[device] = self.map_query_grid_lat[device]
                self.map_keyvalue_grid_lon[device] = self.map_query_grid_lon[device]
            else:
                keyvalue_grid_latlon = self.get_grid_latlon(self.keyvalue_grid_name)
                self.map_keyvalue_grid_lat[device] = keyvalue_grid_latlon[:, 0].to(device).contiguous()
                self.map_keyvalue_grid_lon[device] = keyvalue_grid_latlon[:, 1].to(device).contiguous()

            if self.method == "haversine":
                self.map_attention_span[device] = torch.as_tensor(self.attention_span).to(device)

            if self.method == "knn_haversine":
                self.map_haversine_distance_of_nth_closest_node[device] = (
                    self.haversine_distance_of_nth_closest_node.to(device).contiguous()
                )

    def get_connected_nodes(self, nodes, reference_nodes, attn_span):
        # Behaviour is that for the target grid, we get the indices of the attn_span closest nodes on the source grid

        nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=24, n_neighbors=attn_span)
        nearest_neighbour.fit(reference_nodes.x.numpy())
        indices = nearest_neighbour.kneighbors(nodes.x.numpy(), return_distance=False)

        return indices  # shape (source_nodes, attn_span)

    def setup_connectivity_matrix(self):

        attn_span = self.attention_span

        query_nodes = self.graph[self.query_grid_name]
        keyvalue_nodes = self.graph[self.keyvalue_grid_name]

        if self.base_grid == "query":
            indices = self.get_connected_nodes(query_nodes, keyvalue_nodes, attn_span)
            adj_matrix = np.zeros(
                (query_nodes.x.shape[0], keyvalue_nodes.x.shape[0]), dtype=np.bool
            )  # shape (source_grid_size, target_grid_size)
            adj_matrix[indices] = True

        elif self.base_grid == "keyvalue":
            indices = self.get_connected_nodes(keyvalue_nodes, query_nodes, attn_span)
            adj_matrix = np.zeros(
                (keyvalue_nodes.x.shape[0], query_nodes.x.shape[0]), dtype=np.bool
            )  # shape (target_grid_size, source_grid_size)
            adj_matrix[indices] = True
            adj_matrix = adj_matrix.T  # shape (source_grid_size, target_grid_size)

        return adj_matrix

    def setup_knn_index_matrix(self) -> dict[int, torch.Tensor]:

        keyvalue_nodes = self.graph[self.keyvalue_grid_name]
        query_nodes = self.graph[self.query_grid_name]
        attn_span = self.attention_span

        if self.base_grid == "query":
            connected_nodes = self.get_connected_nodes(
                query_nodes, keyvalue_nodes, attn_span
            )  # shape (source_grid_size, attn_span)

        elif self.base_grid == "keyvalue":
            connected_nodes = self.get_connected_nodes(
                keyvalue_nodes, query_nodes, attn_span
            )  # shape (target_grid_size, attn_span)

        index_matrix = connected_nodes  # shape (target_grid_size, attention_span)

        return index_matrix  # shape (target_grid_size, attention_span) or (source_grid_size, attention_span)

    def get_grid_latlon(self, grid_name: str):
        return self.graph[grid_name].x  # (grid_size, 2)squeue

    def setup_haversine_distance_of_nth_closest_node(self):

        knn_index_matrix = (
            self.setup_knn_index_matrix()
        )  # shape (tgt_grid_size, attention_span) # contains indexes from the base grid

        # Now for each node in source grid we calculate the haversine distance to each of its knn nearest neighbours in target grid

        keyvalue_grid_latlon = self.get_grid_latlon(self.keyvalue_grid_name)
        query_grid_latlon = self.get_grid_latlon(self.query_grid_name)

        if self.base_grid == "query":

            # Vectorized version - no loops needed
            query_lats = query_grid_latlon[:, 0].unsqueeze(1)  # [query_size, 1]
            query_lons = query_grid_latlon[:, 1].unsqueeze(1)  # [query_size, 1]

            # Index into keyvalue grid using knn_index_matrix
            keyvalue_lats = keyvalue_grid_latlon[knn_index_matrix, 0]  # [query_size, k]
            keyvalue_lons = keyvalue_grid_latlon[knn_index_matrix, 1]  # [query_size, k]

            # Calculate distances for all pairs at once
            distances = haversine_distance(
                query_lats, query_lons, keyvalue_lats, keyvalue_lons, self.earth_radius
            )  # [query_size, k]

            # Get maximum distance for each query point
            haversine_distance_of_nth_closest_node = torch.max(distances, dim=1)[0]  # [query_size]

        elif self.base_grid == "keyvalue":

            # Vectorized version for keyvalue base grid
            keyvalue_lats = keyvalue_grid_latlon[:, 0].unsqueeze(1)  # [keyvalue_size, 1]
            keyvalue_lons = keyvalue_grid_latlon[:, 1].unsqueeze(1)  # [keyvalue_size, 1]

            # Index into query grid using knn_index_matrix
            query_lats = query_grid_latlon[knn_index_matrix, 0]  # [keyvalue_size, k]
            query_lons = query_grid_latlon[knn_index_matrix, 1]  # [keyvalue_size, k]

            # Calculate distances for all pairs at once
            distances = haversine_distance(
                query_lats, query_lons, keyvalue_lats, keyvalue_lons, self.earth_radius
            )  # [keyvalue_size, k]

            # Get maximum distance for each keyvalue point
            haversine_distance_of_nth_closest_node = torch.max(distances, dim=1)[0]  # [keyvalue_size]

        return haversine_distance_of_nth_closest_node

    def get_block_mask(self, device: torch.device):
        """Check if block mask exists for device, if not create it for device"""
        if device in self.map_device_block_mask:
            return self.map_device_block_mask[device]
        else:
            q_grid_size = self.graph[self.query_grid_name].num_nodes
            kv_grid_size = self.graph[self.keyvalue_grid_name].num_nodes

            self.copy_to_device(device)

            block_mask = create_block_mask(
                self.mask_mod,
                B=None,
                H=None,
                Q_LEN=q_grid_size,
                KV_LEN=kv_grid_size,
                device=device,
                BLOCK_SIZE=self.block_size,
            )
            self.map_device_block_mask[device] = block_mask
            return block_mask

    def setup_mask_mod(self):

        if self.method == "window":
            self.mask_mod = self.mask_mod_window
        elif self.method == "knn_adjacency_matrix":
            self.mask_mod = self.mask_mod_knn_adjacency_matrix
        elif self.method == "knn_index_matrix":
            if self.base_grid == "query":
                self.mask_mod = self.mask_mod_knn_index_matrix_base_grid_query
            elif self.base_grid == "keyvalue":
                self.mask_mod = self.mask_mod_knn_index_matrix_base_grid_keyvalue
        elif self.method == "haversine":
            self.mask_mod = self.mask_mod_haversine
        elif self.method == "knn_haversine":
            if self.base_grid == "query":
                self.mask_mod = self.mask_mod_knn_haversine_base_grid_query
            elif self.base_grid == "keyvalue":
                self.mask_mod = self.mask_mod_knn_haversine_base_grid_keyvalue
        else:
            raise ValueError(f"Invalid method: {self.method} with base grid: {self.base_grid}")

    def mask_mod_window(self, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        l_outp = q_idx - kv_idx <= self.map_attention_span[kv_idx.device]
        r_outp = -self.map_attention_span[kv_idx.device] <= q_idx - kv_idx
        return l_outp & r_outp

    def mask_mod_knn_adjacency_matrix(self, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return self.map_connectivity_matrix[kv_idx.device][kv_idx, q_idx]

    def mask_mod_knn_index_matrix_base_grid_query(self, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        _1 = self.map_knn_index_matrix[kv_idx.device][q_idx] == kv_idx
        _2 = torch.sum(_1, dim=0, keepdim=True)
        _3 = _2.to(torch.bool)
        return _3[0]

    def mask_mod_knn_index_matrix_base_grid_keyvalue(
        self, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor
    ) -> Tensor:
        _1 = self.map_knn_index_matrix[kv_idx.device][kv_idx] == q_idx
        _2 = torch.sum(_1, dim=0, keepdim=True)
        _3 = _2.to(torch.bool)
        return _3[0]

    def mask_mod_haversine(self, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        keyvalue_lat = self.map_keyvalue_grid_lat[kv_idx.device]
        keyvalue_lon = self.map_keyvalue_grid_lon[kv_idx.device]
        query_lat = self.map_query_grid_lat[kv_idx.device]
        query_lon = self.map_query_grid_lon[kv_idx.device]

        dist = haversine_distance(
            keyvalue_lat[kv_idx],
            keyvalue_lon[kv_idx],
            query_lat[q_idx],
            query_lon[q_idx],
            self.map_earth_radius[kv_idx.device],
        )
        return dist <= self.map_attention_span[kv_idx.device]

    def mask_mod_knn_haversine_base_grid_query(self, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        keyvalue_lat = self.map_keyvalue_grid_lat[kv_idx.device]
        keyvalue_lon = self.map_keyvalue_grid_lon[kv_idx.device]
        query_lat = self.map_query_grid_lat[kv_idx.device]
        query_lon = self.map_query_grid_lon[kv_idx.device]

        max_haversine_distance = self.map_haversine_distance_of_nth_closest_node[kv_idx.device][q_idx]

        dist = haversine_distance(
            keyvalue_lat[kv_idx],
            keyvalue_lon[kv_idx],
            query_lat[q_idx],
            query_lon[q_idx],
            self.map_earth_radius[kv_idx.device],
        )
        return dist <= max_haversine_distance

    def mask_mod_knn_haversine_base_grid_keyvalue(self, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        keyvalue_lat = self.map_keyvalue_grid_lat[kv_idx.device]
        keyvalue_lon = self.map_keyvalue_grid_lon[kv_idx.device]
        query_lat = self.map_query_grid_lat[kv_idx.device]
        query_lon = self.map_query_grid_lon[kv_idx.device]

        max_haversine_distance = self.map_haversine_distance_of_nth_closest_node[kv_idx.device][kv_idx]

        dist = haversine_distance(
            query_lat[q_idx],
            query_lon[q_idx],
            keyvalue_lat[kv_idx],
            keyvalue_lon[kv_idx],
            self.map_earth_radius[kv_idx.device],
        )
        return dist <= max_haversine_distance

    def signature(self) -> tuple:
        return (self.attention_span, self.keyvalue_grid_name, self.query_grid_name, self.base_grid)


def haversine_distance(lat1, lon1, lat2, lon2, earth_radius):

    # # Compute the differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Apply the haversine formula
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return earth_radius * c

    # return 264.1


def calculate_scaled_attention_attention_spans(
    base_attention_span: int,
    base_grid_name: str,
    target_grid_name: str,
    scaling_method: str = "constant_span_relative_to_grid_size",
    _graph_data: HeteroData = None,
    method: str = "knn_haversine",
) -> int:
    """Calculates the scaled attention window sizes for the encoder and decoder."""

    if scaling_method == "scale_span_relative_to_grid_size":
        base_grid_size = _graph_data[base_grid_name].num_nodes
        target_grid_size = _graph_data[target_grid_name].num_nodes

        attn_span = base_attention_span * (target_grid_size / base_grid_size)

    elif scaling_method == "constant_span_relative_to_grid_size":
        attn_span = base_attention_span

    elif scaling_method == "inverse_scale_span_relative_to_grid_size":

        base_grid_size = _graph_data[base_grid_name].num_nodes
        target_grid_size = _graph_data[target_grid_name].num_nodes

        attn_span = base_attention_span * (base_grid_size / target_grid_size)

    else:
        raise ValueError(f"Invalid scaling method: {scaling_method}")

    if method in ["knn_haversine", "window"]:
        attn_span = int(max(3, attn_span))  # Ensure Triagulation ??? or not

    return attn_span


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
        block_mask: Optional[BlockMaskManager] = None,
        kernel_options: Optional[dict] = None,
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads  # q k v
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.qk_norm = qk_norm
        self.rotary_embeddings = rotary_embeddings

        self.lin_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.lin_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.lin_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attention = attn_func


        self.projection = nn.Linear(embed_dim, embed_dim, bias=True)

        if self.qk_norm:
            self.q_norm = AutocastLayerNorm(self.head_dim, bias=False)
            self.k_norm = AutocastLayerNorm(self.head_dim, bias=False)

        if self.rotary_embeddings:  # find alternative implementation
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        
        
        if block_mask is None:
            self.block_mask = None
            self.flex_attention = False
            self.window_size = (window_size, window_size)  # flash attention
        else:

            self.block_mask = block_mask
            self.flex_attention = True
            self.kernel_options = kernel_options
        


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

        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        if self.rotary_embeddings:  # can this be done in a better way?
            key = key.unsqueeze(-3)
            value = value.unsqueeze(-3)
            keyvalue = torch.cat((key, value), dim=-3)
            query, keyvalue = self.rotary_emb(
                query, keyvalue, max_seqlen=max(keyvalue.shape[1], query.shape[1])
            )  # assumption seq const
            key = keyvalue[:, :, 0, ...]
            value = keyvalue[:, :, 1, ...]
        
        if not self.flex_attention:
            out = self.attention(query, key, value, window_size=self.window_size)
        else:
            # Don't include dropout_p, not used in any top models anymore
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            out = flex_attention(
                query,
                key,
                value,
                block_mask=self.block_mask.get_block_mask(device=query.device),
                kernel_options=self.kernel_options,
            )

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


