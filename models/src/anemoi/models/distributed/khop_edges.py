# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from dataclasses import dataclass
from typing import Optional
from typing import Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor
from torch_geometric.utils import bipartite_subgraph
from torch_geometric.utils import degree
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import mask_to_index

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.graph import sync_tensor
from anemoi.models.distributed.primitives import _gather
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes


def get_k_hop_edges(
    nodes: Tensor,
    edge_attr: Tensor,
    edge_index: Adj,
    num_hops: int = 1,
    num_nodes: Optional[int] = None,
) -> tuple[Adj, Tensor]:
    """Return 1 hop subgraph.

    Parameters
    ----------
    nodes : Tensor
        destination nodes
    edge_attr : Tensor
        edge attributes
    edge_index : Adj
        edge index
    num_hops: int, Optional, by default 1
        number of required hops

    Returns
    -------
    tuple[Adj, Tensor]
        K-hop subgraph of edge index and edge attributes
    """
    _, edge_index_k, _, edge_mask_k = k_hop_subgraph(
        node_idx=nodes,
        num_hops=num_hops,
        edge_index=edge_index,
        directed=True,
        num_nodes=num_nodes,
    )

    return edge_attr[mask_to_index(edge_mask_k)], edge_index_k


def sort_edges_1hop_sharding(
    num_nodes: Union[int, tuple[int, int]],
    edge_attr: Tensor,
    edge_index: Adj,
    mgroup: Optional[ProcessGroup] = None,
    relabel_dst_nodes: bool = False,
) -> tuple[Adj, Tensor, ShardSizes]:
    """Rearanges edges into 1 hop neighbourhoods for sharding across GPUs.

    Parameters
    ----------
    num_nodes : Union[int, tuple[int, int]]
        Number of (target) nodes in Graph
    edge_attr : Tensor
        edge attributes
    edge_index : Adj
        edge index
    mgroup : ProcessGroup
        model communication group
    relabel_dst_nodes : bool, optional
        whether to relabel destination nodes to be contiguous, by default False

    Returns
    -------
    tuple[Adj, Tensor, list]
        edges sorted according to k hop neigh., edge attributes of sorted edges,
        shapes of edges for partitioning between GPUs
    """
    if mgroup:
        num_chunks = dist.get_world_size(group=mgroup)

        edge_attr_list, edge_index_list = sort_edges_1hop_chunks(
            num_nodes, edge_attr, edge_index, num_chunks, relabel_dst_nodes=relabel_dst_nodes
        )

        edge_shard_sizes = [e.shape[0] for e in edge_attr_list]

        return torch.cat(edge_attr_list, dim=0), torch.cat(edge_index_list, dim=1), edge_shard_sizes

    return edge_attr, edge_index, []


def shard_edges_1hop(
    edge_attr: Tensor,
    edge_index: Adj,
    src_size: int,
    dst_size: int,
    model_comm_group: Optional[ProcessGroup],
) -> tuple[Tensor, Adj, ShardSizes]:
    """Sort and shard edges for 1-hop sharding.

    Parameters
    ----------
    edge_attr : Tensor
        Edge attributes
    edge_index : Adj
        Edge index
    src_size : int
        Number of source nodes
    dst_size : int
        Number of destination nodes
    model_comm_group : ProcessGroup, optional
        Model communication group

    Returns
    -------
    tuple[Tensor, Adj, list[int]]
        Sharded edge_attr, sharded edge_index, and list of edge_shard_sizes
    """
    from anemoi.models.distributed.graph import shard_tensor

    num_nodes = (src_size, dst_size)
    edge_attr, edge_index, edge_shard_sizes = sort_edges_1hop_sharding(
        num_nodes, edge_attr, edge_index, model_comm_group
    )
    edge_index = shard_tensor(edge_index, 1, edge_shard_sizes, model_comm_group)
    edge_attr = shard_tensor(edge_attr, 0, edge_shard_sizes, model_comm_group)

    return edge_attr, edge_index, edge_shard_sizes


def sort_edges_1hop_chunks(
    num_nodes: Union[int, tuple[int, int]],
    edge_attr: Tensor,
    edge_index: Adj,
    num_chunks: int,
    relabel_dst_nodes: bool = False,
) -> tuple[list[Tensor], list[Adj]]:
    """Rearanges edges into 1 hop neighbourhood chunks.

    Parameters
    ----------
    num_nodes : Union[int, tuple[int, int]]
        Number of (target) nodes in Graph, tuple for bipartite graph
    edge_attr : Tensor
        edge attributes
    edge_index : Adj
        edge index
    num_chunks : int
        number of chunks used if mgroup is None
    relabel_dst_nodes : bool, optional
        whether to relabel nodes in the subgraph, by default False

    Returns
    -------
    tuple[list[Tensor], list[Adj]]
        list of sorted edge attribute chunks, list of sorted edge_index chunks
    """
    if isinstance(num_nodes, int):
        node_chunks = torch.arange(num_nodes, device=edge_index.device).tensor_split(num_chunks)
    else:
        nodes_src = torch.arange(num_nodes[0], device=edge_index.device)
        node_chunks = torch.arange(num_nodes[1], device=edge_index.device).tensor_split(num_chunks)

    edge_index_list = []
    edge_attr_list = []
    for node_chunk in node_chunks:
        if isinstance(num_nodes, int):
            edge_attr_chunk, edge_index_chunk = get_k_hop_edges(node_chunk, edge_attr, edge_index, num_nodes=num_nodes)
        else:
            edge_index_chunk, edge_attr_chunk = bipartite_subgraph(
                (nodes_src, node_chunk),
                edge_index,
                edge_attr,
                size=(num_nodes[0], num_nodes[1]),
            )

        if relabel_dst_nodes:  # relabel dst nodes to be contiguous
            edge_index_chunk[1] -= node_chunk[0]  # shift dst nodes to start from 0

        edge_index_list.append(edge_index_chunk)
        edge_attr_list.append(edge_attr_chunk)

    return edge_attr_list, edge_index_list


def drop_unconnected_src_nodes(
    x_src: Tensor, edge_index: Adj, num_nodes: tuple[int, int]
) -> tuple[Tensor, Adj, Tensor]:
    """Drop unconnected nodes from x_src and relabel edges.

    Parameters
    ----------
    x_src : Tensor
        source node features
    edge_index : Adj
        edge index
    num_nodes : tuple[int, int]
        number of nodes in graph (src, dst)

    Returns
    -------
    tuple[Tensor, Adj, Tensor]
        reduced node features, relabeled edge index (contiguous, starting from 0),
        indices of connected source nodes
    """
    connected_src_nodes = torch.unique(edge_index[0])
    dst_nodes = torch.arange(num_nodes[1], device=x_src.device)

    edge_index_new, _ = bipartite_subgraph(
        (connected_src_nodes, dst_nodes),
        edge_index,
        size=num_nodes,
        relabel_nodes=True,
    )

    return x_src[connected_src_nodes], edge_index_new, connected_src_nodes


@dataclass(frozen=True)
class GraphPartition:
    """Precomputed partitioning metadata for a bipartite graph with dst-sorted edges.

    Enables O(1) slicing for both distributed sharding and local chunking
    by exploiting the fact that edges are already sorted by destination node.

    Parameters
    ----------
    num_nodes : tuple[int, int]
        Number of (src, dst) nodes in the full graph.
    num_edges : int
        Total number of edges.
    num_parts : int
        Number of partitions (= world size for sharding, or num_parts for local chunking).
    dst_splits : list[int]
        Per-partition destination node counts.
    edge_splits : list[int]
        Per-partition edge counts (derived from dst-sorted edge structure).
    """

    num_nodes: tuple[int, int]
    num_edges: int
    num_parts: int
    dst_splits: list[int]
    edge_splits: list[int]

    def materialise(
        self,
        partition_id: int,
        x: PairTensor,
        edge_attr: Tensor,
        edge_index: Adj,
        cond: Optional[PairTensor] = None,
    ) -> tuple[PairTensor, Tensor, Adj, Tensor, Optional[PairTensor]]:
        """Materialise a single partition by slicing nodes, edges and conditioning.

        Pure local operation — no communication. Suitable for chunking within
        a single device.

        Parameters
        ----------
        partition_id : int
            The partition to materialise.
        x : PairTensor
            Node features (src, dst).
        edge_attr : Tensor
            Edge attributes.
        edge_index : Adj
            Edge indices (assumed dst-sorted).
        cond : tuple[Tensor, Tensor], optional
            Conditioning tensors (cond_src, cond_dst).

        Returns
        -------
        tuple[PairTensor, Tensor, Adj, Optional[PairTensor]]
            (x_src_subset, x_dst_subset), edge_attr_subset, edge_index_relabeled,
            cond subset (or None).
        """
        x_src, x_dst = x

        # slice edges and dst nodes for this partition
        edge_range = self._get_edge_range(partition_id)
        edge_attr_subset = edge_attr[edge_range]
        edge_index_subset = edge_index[:, edge_range].clone()  # clone to avoid in-place corruption

        dst_range = self._get_dst_range(partition_id)
        x_dst_subset = x_dst[dst_range]

        # relabel dst indices to local [0, partition_dst_size)
        self._relabel_dst_nodes(edge_index_subset, partition_id)

        # drop src nodes with no edges in this partition
        x_src_subset, edge_index_subset, src_ids = _drop_unconnected_src_nodes(x_src, edge_index_subset)

        # subset conditioning if provided
        cond_subset = None
        if cond is not None:
            cond_src, cond_dst = cond
            cond_subset = (cond_src[src_ids], cond_dst[dst_range])

        return (x_src_subset, x_dst_subset), edge_attr_subset, edge_index_subset, cond_subset

    def _get_edge_range(self, partition_id: int) -> slice:
        start, end = get_partition_range(self.edge_splits, partition_id)
        return slice(start, end)

    def _get_dst_range(self, partition_id: int) -> slice:
        start, end = get_partition_range(self.dst_splits, partition_id)
        return slice(start, end)

    def _relabel_dst_nodes(self, edge_index: Adj, partition_id: int) -> None:
        """Relabel dst indices from global to partition-local (in-place).

        Caller must ensure edge_index is a clone if the original must be preserved.
        """
        dst_offset = get_partition_range(self.dst_splits, partition_id)[0]
        edge_index[1] -= dst_offset


def _drop_unconnected_src_nodes(x_src: Tensor, edge_index: Adj) -> tuple[Tensor, Adj, Tensor]:
    """Drop src nodes with no edges and relabel src indices to be contiguous.

    Modifies edge_index[0] in-place. Caller must clone edge_index beforehand
    if the original must be preserved.

    Parameters
    ----------
    x_src : Tensor
        Source node features.
    edge_index : Adj
        Edge index (will be modified in-place on row 0).

    Returns
    -------
    tuple[Tensor, Adj, Tensor]
        Subset of x_src, relabeled edge_index, indices of connected source nodes.
    """
    connected_src_nodes = torch.unique(edge_index[0])
    x_src_subset = x_src[connected_src_nodes]

    relabel_map = torch.empty(x_src.shape[0], dtype=torch.long, device=x_src.device)
    relabel_map[connected_src_nodes] = torch.arange(connected_src_nodes.size(0), device=x_src.device)
    edge_index[0] = relabel_map[edge_index[0]]

    return x_src_subset, edge_index, connected_src_nodes


def shard_graph_to_local(
    partition: GraphPartition,
    x: PairTensor,
    edge_attr: Tensor,
    edge_index: Adj,
    shard_info: BipartiteGraphShardInfo,
    model_comm_group: Optional[ProcessGroup] = None,
    cond: Optional[PairTensor] = None,
) -> tuple[PairTensor, Tensor, Adj, BipartiteGraphShardInfo, Optional[PairTensor]]:
    """Shard graph tensors to the local rank using precomputed partition metadata.

    Handles all communication (sync src, shard dst/edges) and returns
    the local subgraph with updated shard metadata.

    Parameters
    ----------
    partition : GraphPartition
        Precomputed partition metadata.
    x : PairTensor
        Node features (src, dst).
    edge_attr : Tensor
        Edge attributes.
    edge_index : Adj
        Edge indices (assumed dst-sorted).
    shard_info : BipartiteGraphShardInfo
        Current shard metadata.
    model_comm_group : ProcessGroup, optional
        Model communication group.
    cond : tuple[Tensor, Tensor], optional
        Conditioning tensors (cond_src, cond_dst).

    Returns
    -------
    tuple[PairTensor, Tensor, Adj, BipartiteGraphShardInfo, Optional[PairTensor]]
        Sharded (x_src_local, x_dst), edge_attr, edge_index, updated shard_info,
        cond subset (or None).
    """
    if model_comm_group is None or model_comm_group.size() <= 1:
        return x, edge_attr, edge_index, shard_info, cond

    assert (
        model_comm_group.size() == partition.num_parts
    ), f"Expected comm group size {partition.num_parts} but got {model_comm_group.size()}"

    x_src, x_dst = x
    my_rank = torch.distributed.get_rank(group=model_comm_group)

    # shard or validate dst nodes
    if shard_info.dst_is_sharded():
        assert (
            shard_info.dst_nodes == partition.dst_splits
        ), f"Expected dst shard shapes {partition.dst_splits} but got {shard_info.dst_nodes}"
    else:
        x_dst = shard_tensor(x_dst, 0, partition.dst_splits, model_comm_group)

    # shard or validate edges
    if shard_info.edges_are_sharded():
        assert (
            shard_info.edges == partition.edge_splits
        ), f"Expected edge shard shapes {partition.edge_splits} but got {shard_info.edges}"
    else:
        edge_attr = shard_tensor(edge_attr, 0, partition.edge_splits, model_comm_group)
        edge_index = shard_tensor(edge_index, 1, partition.edge_splits, model_comm_group)

    # relabel dst indices to local
    edge_index = edge_index.clone()
    partition._relabel_dst_nodes(edge_index, partition_id=my_rank)

    # gather x_src — always reduce in backward for correct gradients on halo nodes
    x_src_full = sync_tensor(
        x_src,
        0,
        shard_info.src_nodes,
        model_comm_group,
        gather_in_fwd=shard_info.src_is_sharded(),
    )

    x_src_local, edge_index, src_ids = _drop_unconnected_src_nodes(x_src_full, edge_index)

    # same for conditioning [if cond is not None]
    cond_local = None
    if cond is not None:
        cond_src, cond_dst = cond
        cond_src_full = sync_tensor(cond_src, 0, shard_info.src_nodes, model_comm_group)
        cond_local = (cond_src_full[src_ids], cond_dst)

    updated_shard_info = BipartiteGraphShardInfo(
        src_nodes=shard_info.src_nodes,
        dst_nodes=partition.dst_splits,
        edges=partition.edge_splits,
    )

    return (x_src_local, x_dst), edge_attr, edge_index, updated_shard_info, cond_local


def build_graph_partition(edge_index: Adj, num_parts: int, num_nodes: tuple[int, int]) -> GraphPartition:
    """Build graph partitioning information from a dst-sorted edge_index.

    Parameters
    ----------
    edge_index : Adj
        The edge index tensor (must be sorted by destination node).
    num_parts : int
        The number of chunks to partition the graph into.
    num_nodes : tuple[int, int]
        The number of (src, dst) nodes in the graph.

    Returns
    -------
    GraphPartition
        The graph partitioning information.
    """
    n_dst = num_nodes[1]
    dst_splits = get_balanced_partition_sizes(n_dst, num_parts)
    degree_per_dst = degree(edge_index[1], num_nodes=n_dst, dtype=torch.long)
    # use torch.split with dst_splits to match the balanced partitioning exactly
    edge_splits = [chunk.sum().item() for chunk in torch.split(degree_per_dst, dst_splits)]

    return GraphPartition(
        num_nodes=num_nodes,
        num_edges=edge_index.size(1),
        num_parts=num_parts,
        dst_splits=dst_splits,
        edge_splits=edge_splits,
    )


def build_graph_partition_from_shard_info(
    edge_index: Adj,
    x: PairTensor,
    shard_info: BipartiteGraphShardInfo,
    model_comm_group: Optional[ProcessGroup] = None,
) -> GraphPartition:
    """Build a GraphPartition for distributed sharding from current shard metadata.

    Derives num_nodes from shard_info and tensor shapes, and sets num_parts
    to the communication group size.

    Parameters
    ----------
    edge_index : Adj
        The edge index tensor (must be sorted by destination node).
    x : PairTensor
        Node features (src, dst), used to infer sizes when not sharded.
    shard_info : BipartiteGraphShardInfo
        Current shard metadata.
    model_comm_group : ProcessGroup, optional
        Model communication group.

    Returns
    -------
    GraphPartition
        The graph partitioning information.
    """
    x_src, x_dst = x
    n_src = sum(shard_info.src_nodes) if shard_info.src_is_sharded() else x_src.size(0)
    n_dst = sum(shard_info.dst_nodes) if shard_info.dst_is_sharded() else x_dst.size(0)
    comm_size = model_comm_group.size() if model_comm_group is not None else 1

    edge_index_full = edge_index
    if shard_info.edges_are_sharded():  # need full edge_index to build partitioning
        edge_index_full = _gather(edge_index, 1, shard_info.edges, model_comm_group)

    return build_graph_partition(edge_index_full, num_parts=comm_size, num_nodes=(n_src, n_dst))
