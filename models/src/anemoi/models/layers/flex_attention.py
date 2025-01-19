# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from typing import Optional

import einops
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention
from torch_geometric.data import HeteroData

from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence

# Compile the flex_attention function
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

flex_attention = torch.compile(flex_attention, dynamic=False, fullgraph=False)
# flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune")
# flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
# flex_attention = torch.compile(flex_attention, backend="inductor")
# flex_attention = torch.compile(flex_attention, backend="aot_eager")
# flex_attention = torch.compile(flex_attention)

# create_block_mask = torch.compile(create_block_mask, dynamic=True, fullgraph=True) #https://twitter.com/cHHillee/status/1851418255749169419
create_block_mask = torch.compile(
    create_block_mask, dynamic=False
)  # https://twitter.com/cHHillee/status/1851418255749169419


torch.set_float32_matmul_precision("medium")

torch._dynamo.config.cache_size_limit = 3072
torch._inductor.config.mixed_mm_choice = "triton"  # removed ATEN_


torch._inductor.config.max_autotune = True
# torch._inductor.config.max_autotune = False
torch._inductor.config.max_autotune_gemm_backends = "TRITON,CPP"
torch._inductor.config.max_autotune_gemm_search_space = "EXHAUSTIVE"
# torch._inductor.config.max_autotune_gemm_search_space = "DEFAULT"

torch._inductor.config.max_autotune_pointwise = False
torch._inductor.config.max_autotune_gemm = True
torch._inductor.config.autotune_num_choices_displayed = 10 # 5 is default setting, (NOTE: I think 10 may make a noticeable difference in performance in the GraphTransformer w/ TransformerProcessor knn flex attention ) 

LOGGER = logging.getLogger(__name__)


class BlockMaskCreator(nn.Module):

    def __init__(
        self,
        graph: HeteroData,
        attention_span: int | dict[str, float],
        query_grid_name: str,
        keyvalue_grid_name: str,
        devices: list[str | torch.device],
        method: str = "knn_adjacency_matrix",
        base_grid: str = "query",
        **method_kwargs,
    ):

        super().__init__()

        self.graph = graph

        assert method in ["knn_adjacency_matrix", "knn_index_matrix", "haversine", "knn_haversine"]
        self.method = method

        self.attention_span = attention_span
        # Attn span is an int for the knn based methods
        # Attn span is a map for the latlon proximity method, where the keys will represent the dimension and the values will represent the max distance in degrees

        self.base_grid = base_grid  # NOTE: this is not used in the latlon proximity method yet

        self.query_grid_name = query_grid_name
        self.keyvalue_grid_name = keyvalue_grid_name

        self.devices = devices
        # FIX: THE ISSUE WITH CURRENT METHODOLOGY IS THAT EACH GPU will have a copy of data needed for all other GPUS as well as its own data.
        # FIX: This by including a hook that ensures the necessary tensors are moved to the correct device when to() is called

        self.debug = False

        self.setup_attn_mask(**method_kwargs)

    def setup_attn_mask(self, **method_kwargs):
        assert self.devices is not None and len(self.devices) > 0

        if self.debug:
            self.map_attention_span = {
                device: torch.as_tensor(self.attention_span).to(device) for device in self.devices
            }
            return None

        # self.mode is True/False if we are checking distance from each node in keyvalue grid or query grid
        if self.base_grid == "query":
            self.mode = {device: torch.as_tensor(False).to(device) for device in self.devices}
        elif self.base_grid == "keyvalue":
            self.mode = {device: torch.as_tensor(True).to(device) for device in self.devices}

        if self.method == "knn_index_matrix":
            knn_index_matrix = self.setup_knn_index_matrix()

            self.map_knn_index_matrix = {device: knn_index_matrix.to(device) for device in self.devices}

        elif self.method == "knn_adjacency_matrix":
            connectivity_matrix = self.setup_connectivity_matrix()

            self.map_connectivity_matrix = {device: connectivity_matrix.to(device) for device in self.devices}

        elif self.method in ["haversine", "knn_haversine"]:
            src_grid_latlon = self.get_grid_latlon(self.query_grid_name)

            self.map_src_grid_lat = {device: src_grid_latlon[:, 0].to(device).contiguous() for device in self.devices}
            self.map_src_grid_lon = {device: src_grid_latlon[:, 1].to(device).contiguous() for device in self.devices}

            if self.query_grid_name == self.keyvalue_grid_name:
                tgt_grid_latlon = src_grid_latlon
                self.map_tgt_grid_lat = self.map_src_grid_lat
                self.map_tgt_grid_lon = self.map_src_grid_lon
            else:
                tgt_grid_latlon = self.get_grid_latlon(self.keyvalue_grid_name)
                self.map_tgt_grid_lat = {
                    device: tgt_grid_latlon[:, 0].to(device).contiguous() for device in self.devices
                }
                self.map_tgt_grid_lon = {
                    device: tgt_grid_latlon[:, 1].to(device).contiguous() for device in self.devices
                }

            if self.method == "haversine":
                self.map_attention_span = {
                    device: torch.as_tensor(self.attention_span).to(device) for device in self.devices
                }

            if self.method == "knn_haversine":
                # So here we implement knn attention in a way that does not utilize the any form of reduction which causes CUDA ERRORS
                # Step 1: For each node on the source grid and target grid we retrieve the latlon coordinates
                # Step 2: We then calculate the haversine distance between each node in source and the n'th closest node in target
                # Step 3: Then we hold these tensors on the appropriate device
                haversine_distance_of_nth_closest_node = self.setup_haversine_distance_of_nth_closest_node()
                self.map_haversine_distance_of_nth_closest_node = {
                    device: haversine_distance_of_nth_closest_node.to(device) for device in self.devices
                }

    def get_connected_nodes(self, nodes, reference_nodes, attn_span):
        # Behaviour is that for the target grid, we get the indices of the attn_span closest nodes on the source grid

        nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=8, n_neighbors=attn_span)
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
        )  # shape (tgt_grid_size, attention_span) # contains indexes from the src grid

        # Now for each node in source grid we calculate the haversine distance to each of its knn nearest neighbours in target grid

        keyvalue_grid_latlon = self.get_grid_latlon(self.keyvalue_grid_name)
        query_grid_latlon = self.get_grid_latlon(self.query_grid_name)

        if self.base_grid == "query":

            haversine_distance_of_nth_closest_node = torch.zeros(
                (query_grid_latlon.shape[0],),
            )  # shape (src_grid_size,)

            for query_node_idx in range(knn_index_matrix.shape[0]):

                query_lat = query_grid_latlon[query_node_idx, 0]
                query_lon = query_grid_latlon[query_node_idx, 1]

                keyvalue_node_idxs = knn_index_matrix[query_node_idx]

                keyvalue_lats = keyvalue_grid_latlon[keyvalue_node_idxs, 0]
                keyvalue_lons = keyvalue_grid_latlon[keyvalue_node_idxs, 1]

                dist = haversine_distance(query_lat, query_lon, keyvalue_lats, keyvalue_lons)

                max_dist = torch.max(dist)

                haversine_distance_of_nth_closest_node[query_node_idx] = max_dist

        elif self.base_grid == "keyvalue":

            haversine_distance_of_nth_closest_node = torch.zeros(
                (keyvalue_grid_latlon.shape[0],),
            )

            for keyvalue_node_idx in range(knn_index_matrix.shape[0]):

                keyvalue_lat = keyvalue_grid_latlon[keyvalue_node_idx, 0]
                keyvalue_lon = keyvalue_grid_latlon[keyvalue_node_idx, 1]

                query_node_idxs = knn_index_matrix[keyvalue_node_idx]

                query_lats = query_grid_latlon[query_node_idxs, 0]
                query_lons = query_grid_latlon[query_node_idxs, 1]

                dist = haversine_distance(query_lats, query_lons, keyvalue_lat, keyvalue_lon)

                max_dist = torch.max(dist)

                haversine_distance_of_nth_closest_node[keyvalue_node_idx] = max_dist

        return haversine_distance_of_nth_closest_node

    def setup_block_mask(self) -> dict:
        q_grid_size = self.graph[self.query_grid_name].num_nodes
        kv_grid_size = self.graph[self.keyvalue_grid_name].num_nodes
        map_device_block_mask = {}

        for device in self.devices:

            block_mask = create_block_mask(
                self.block_mask_func, B=None, H=None, Q_LEN=q_grid_size, KV_LEN=kv_grid_size, device=device
            )  #

            map_device_block_mask[device] = block_mask

        return map_device_block_mask

    def block_mask_func(self, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:

        if self.debug:

            l_outp = (
                q_idx - kv_idx <= self.map_attention_span[kv_idx.device]
            )  # & q_idx - kv_idx >= -self.map_attention_span[kv_idx.device]

            r_outp = -self.map_attention_span[kv_idx.device] <= q_idx - kv_idx

            outp = l_outp & r_outp

            return outp

        if self.method == "knn_adjacency_matrix":
            return self.map_connectivity_matrix[kv_idx.device][kv_idx, q_idx]

        elif self.method == "knn_index_matrix":

            if self.mode[kv_idx.device]:
                _1 = self.map_knn_index_matrix[kv_idx.device][kv_idx] == q_idx
                _2 = torch.sum(_1, dim=0, keepdim=True)
                _3 = _2.to(torch.bool)

            elif self.mode[kv_idx.device]:
                _1 = self.map_knn_index_matrix[kv_idx.device][q_idx] == kv_idx
                _2 = torch.sum(_1, dim=0, keepdim=True)
                _3 = _2.to(torch.bool)

            return _3[0]

        elif self.method == "haversine":
            src_lat = self.map_src_grid_lat[kv_idx.device]  # shape (src_grid_size, 2)
            src_lon = self.map_src_grid_lon[kv_idx.device]  # shape (src_grid_size, 2)
            tgt_lat = self.map_tgt_grid_lat[kv_idx.device]  # shape (tgt_grid_size, 2)
            tgt_lon = self.map_tgt_grid_lon[kv_idx.device]  # shape (tgt_grid_size, 2)

            dist = haversine_distance(src_lat[q_idx], src_lon[q_idx], tgt_lat[kv_idx], tgt_lon[kv_idx])

            res = dist < self.map_attention_span[kv_idx.device]

            return res

        elif self.method == "knn_haversine":
            src_lat = self.map_src_grid_lat[kv_idx.device]  # shape (src_grid_size, 2)
            src_lon = self.map_src_grid_lon[kv_idx.device]  # shape (src_grid_size, 2)
            tgt_lat = self.map_tgt_grid_lat[kv_idx.device]  # shape (tgt_grid_size, 2)
            tgt_lon = self.map_tgt_grid_lon[kv_idx.device]  # shape (tgt_grid_size, 2)

            # if self.mode[kv_idx.device]:
            #     max_haversine_distance = self.map_haversine_distance_of_nth_closest_node[kv_idx.device][kv_idx]
            # else:
            max_haversine_distance = self.map_haversine_distance_of_nth_closest_node[kv_idx.device][q_idx]

            dist = haversine_distance(src_lat[q_idx], src_lon[q_idx], tgt_lat[kv_idx], tgt_lon[kv_idx])

            return dist < max_haversine_distance


class MultiHeadCrossFlexAttnQK_same_V_diff(nn.Module):
    """Multi Head Flex KNN Local Attention Pytorch Layer.
    In this attention, the q and k are the same and the v is different.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim_qk: int,
        embed_dim_v: int,
        bias: bool = False,
        is_causal: bool = False,
        block_mask: Tensor | None = None,
        **kwargs,
    ):
        """Initialize the MultiHeadCrossAttentionFlexKnnLocalAttn_QK_same_V_diff layer.

        NOTE: In its current form, this can only be used on the grids that are defined during initialization of this class.

        Args:
            num_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            bias: Whether to use bias in the linear layers.
            is_causal: Whether to apply causal masking.
            attention_span: Window size for flash attention.
            dropout_p: Dropout probability.
            flash_attention: Whether to use flash attention.
            knn_index_matrix: KNN index matrix of shape (query_grid_size, knn_num_nearest_neighbours) representing the KNN indices in key/value grid for the query grid.
        """
        super().__init__()

        assert (
            embed_dim_qk % num_heads == 0
        ), f"Embedding dimension ({embed_dim_qk}) must be divisible by number of heads ({num_heads})"

        self.num_heads = num_heads
        self.embed_dim_qk = embed_dim_qk
        self.embed_dim_v = embed_dim_v
        self.head_dim = embed_dim_qk // num_heads  # q k v
        # self.dropout_p = dropout_p

        self.block_mask = block_mask

        self.lin_qk = nn.Linear(embed_dim_qk, 2 * embed_dim_qk, bias=bias)
        self.lin_v = nn.Linear(embed_dim_v, embed_dim_v, bias=bias)
        self.projection = nn.Linear(embed_dim_v, embed_dim_v, bias=True)

    def forward(
        self, qk: Tensor, v: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:

        query, key = self.lin_qk(qk).chunk(2, -1)
        value = self.lin_v(v)

        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        query, key, value = (
            einops.rearrange(
                t,
                "(batch grid) (heads dim) -> batch heads grid dim",  # NOTE: Slight issue since here we batch contains ( batch, ensemble, timestep)
                batch=batch_size,
                heads=self.num_heads,
            )
            for t in (query, key, value)
        )

        query = shard_heads(query, shapes=shapes, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shapes, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shapes, mgroup=model_comm_group)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        out = flex_attention(
            query,
            key,
            value,
            block_mask=self.block_mask[query.device],
        )

        out = shard_sequence(out, shapes=shapes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        out = self.projection(out)

        return out


class MultiHeadCrossFlexAttetionQKV_diff(nn.Module):
    """Multi Head Flex KNN Local Attention Pytorch Layer.
    In this attention, the q and k are the same and the v is different.

    This would be used in the transformer mapper .
    query =  tgt_grid_latlon
    key = src_grid_latlon + src_grid_weather
    value = src_grid_weather
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim_qk: int,
        embed_dim_v: int,
        bias: bool = False,
        is_causal: bool = False,
        block_mask: Tensor | None = None,
        **kwargs,
    ):
        """Initialize the MultiHeadCrossAttentionFlexKnnLocalAttn_QK_same_V_diff layer.

        Args:
            num_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            bias: Whether to use bias in the linear layers.
            is_causal: Whether to apply causal masking.
            attention_span: Window size for flash attention.
            dropout_p: Dropout probability.
            flash_attention: Whether to use flash attention.
            knn_index_matrix: KNN index matrix of shape (query_grid_size, knn_num_nearest_neighbours) representing the KNN indices in key/value grid for the query grid.
        """
        super().__init__()

        assert (
            embed_dim_qk % num_heads == 0
        ), f"Embedding dimension ({embed_dim_qk}) must be divisible by number of heads ({num_heads})"

        self.num_heads = num_heads
        self.embed_dim_qk = embed_dim_qk
        self.embed_dim_v = embed_dim_v
        self.head_dim = embed_dim_qk // num_heads  # q k v

        self.lin_q = nn.Linear(embed_dim_qk, embed_dim_qk, bias=bias)
        self.lin_k = nn.Linear(embed_dim_qk, embed_dim_qk, bias=bias)
        self.lin_v = nn.Linear(embed_dim_v, embed_dim_v, bias=bias)

        self.block_mask = block_mask

        # NOTE: why do we have this projection here? check to see if this is normal in attn structure
        self.projection = nn.Linear(embed_dim_v, embed_dim_v, bias=False)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        shapes: list,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:

        query = self.lin_q(q)
        key = self.lin_k(k)
        value = self.lin_v(v)

        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        query, key, value = (
            einops.rearrange(
                t,
                "(batch grid) (heads dim) -> batch heads grid dim",  # NOTE: Slight issue since here we batch contains ( batch, ensemble, timestep)
                batch=batch_size,
                heads=self.num_heads,
            )
            for t in (query, key, value)
        )

        query = shard_heads(query, shapes=shapes, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shapes, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shapes, mgroup=model_comm_group)

        ## New version
        # TODO: figure out what the input dimensions order should be
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        out = flex_attention(query, key, value, block_mask=self.block_mask[query.device])

        out = shard_sequence(out, shapes=shapes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        out = self.projection(out)

        return out


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km

    # Compute the differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Apply the haversine formula
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return R * c


def calculate_scaled_attention_attention_spans(
    base_processor_attention_span: int,
    base_grid_name: str,
    target_grid_name: str,
    scaling_method: str = "constant_span_relative_to_grid_size",
    _graph_data: HeteroData = None,
) -> int:
    """Calculates the scaled attention window sizes for the encoder and decoder."""

    if scaling_method == "scale_span_relative_to_grid_size":
        base_grid_size = _graph_data[base_grid_name].num_nodes
        target_grid_size = _graph_data[target_grid_name].num_nodes

        grid_size = base_processor_attention_span * (target_grid_size / base_grid_size)
        grid_size_efficient = int(max(3, (8 * np.round(grid_size / 8))))

    elif scaling_method == "constant_span_relative_to_grid_size":
        grid_size_efficient = int(max(3, (8 * np.round(base_processor_attention_span / 8))))

    elif scaling_method == "inverse_scale_span_relative_to_grid_size":

        base_grid_size = _graph_data[base_grid_name].num_nodes
        target_grid_size = _graph_data[target_grid_name].num_nodes

        grid_size = base_processor_attention_span * (base_grid_size / target_grid_size)
        grid_size_efficient = int(max(3, (8 * np.round(grid_size / 8))))

    return grid_size_efficient
