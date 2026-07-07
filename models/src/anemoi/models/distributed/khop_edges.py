# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import os
from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor
from torch_geometric.utils import degree
from torch_geometric.utils import index_sort

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.graph import sync_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.utils import model_is_distributed

ANEMOI_DEBUG_SHARDING = os.environ.get("ANEMOI_DEBUG_SHARDING", "") != ""


def sort_edge_index_by_dst(edge_index: Adj, max_value: int = None) -> Tuple[Adj, Tensor]:
    """Sort edge indices by destination node."""
    _, perm = index_sort(edge_index[1], max_value=max_value, stable=True)
    return edge_index[:, perm], perm


def is_edge_index_dst_sorted(edge_index: Adj) -> bool:
    """Check whether edge_index is sorted by destination node (edge_index[1])."""
    dst = edge_index[1]
    if dst.numel() <= 1:
        return True
    return bool(torch.all(dst[1:] >= dst[:-1]).item())


@dataclass(frozen=True)
class GraphPartition:
    """Precomputed partitioning metadata for a graph with dst-sorted edges.

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

    def _relabel_dst_nodes(self, edge_index: Adj, partition_id: int, in_place: bool = True) -> Adj:
        """Relabel dst indices from global to partition-local.

        Modifies edge_index in-place by default, but can return a relabeled copy if in_place=False.
        """
        edge_index_ = edge_index if in_place else edge_index.clone()
        dst_offset = get_partition_range(self.dst_splits, partition_id)[0]
        edge_index_[1] -= dst_offset

        return edge_index_


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

    if ANEMOI_DEBUG_SHARDING:
        assert is_edge_index_dst_sorted(edge_index), (
            "build_graph_partition requires edge_index sorted by destination node, " "but received unsorted edges."
        )

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

    if shard_info.edges_are_sharded():  # build partition from existing edge shard info:
        n_edges = sum(shard_info.edges)
        dst_splits = (
            shard_info.dst_nodes if shard_info.dst_is_sharded() else get_balanced_partition_sizes(n_dst, comm_size)
        )
        edge_splits = shard_info.edges
        return GraphPartition(
            num_nodes=(n_src, n_dst),
            num_edges=n_edges,
            num_parts=comm_size,
            dst_splits=dst_splits,
            edge_splits=edge_splits,
        )

    # otherwise: edge_index is not sharded, so we can build the partition directly from it
    return build_graph_partition(edge_index, num_parts=comm_size, num_nodes=(n_src, n_dst))


def build_graph_partition_from_node_partition(
    edge_index: Adj,
    node_partition: Tensor,
    num_nodes: tuple[int, int],
) -> tuple["GraphPartition", Adj, Tensor]:
    """Build a GraphPartition from a node-to-partition assignment tensor.

    Relabels nodes so that each partition occupies a contiguous block,
    re-sorts the edge index by the relabeled destination node, and computes
    dst/edge splits from the actual per-partition node counts.  Intended for
    use with external partitioners such as METIS.

    Parameters
    ----------
    edge_index : Adj
        Edge index with *original* global node ids, shape ``[2, num_edges]``.
        Does not need to be dst-sorted beforehand.
    node_partition : Tensor
        Integer tensor of shape ``[num_nodes]`` mapping each node to a partition
        id in ``[0, num_parts)``.
    num_nodes : tuple[int, int]
        Number of (src, dst) nodes in the full graph.

    Returns
    -------
    partition : GraphPartition
        Partition metadata with custom dst/edge splits derived from the
        external partitioner.
    edge_index_relabeled : Adj
        Edge index relabeled to new contiguous node ids, sorted by destination.
    perm : Tensor
        Permutation of length ``num_nodes[1]`` such that ``perm[new_id]`` is
        the original node id.  The inverse ``inv_perm[old_id] = new_id`` is
        used internally to relabel edge_index.
    """
    num_parts = int(node_partition.max().item()) + 1

    # Sort nodes by partition id so each partition occupies a contiguous range.
    perm = torch.argsort(node_partition, stable=True)   # perm[new_id] = old_id
    inv_perm = torch.empty(perm.size(0), dtype=torch.long, device=perm.device)
    inv_perm[perm] = torch.arange(perm.size(0), device=perm.device)

    # Relabel all node ids in edge_index and sort by (relabeled) dst.
    edge_index_relabeled = inv_perm[edge_index]
    edge_index_relabeled, _ = sort_edge_index_by_dst(edge_index_relabeled)

    # Compute splits from the actual partition node counts (may be unequal).
    partition_sizes = torch.bincount(node_partition, minlength=num_parts)
    dst_splits = [int(x) for x in partition_sizes.tolist()]

    n_dst = num_nodes[1]
    degree_per_dst = degree(edge_index_relabeled[1], num_nodes=n_dst, dtype=torch.long)
    edge_splits = [int(chunk.sum().item()) for chunk in torch.split(degree_per_dst, dst_splits)]

    partition = GraphPartition(
        num_nodes=num_nodes,
        num_edges=edge_index_relabeled.size(1),
        num_parts=num_parts,
        dst_splits=dst_splits,
        edge_splits=edge_splits,
    )

    return partition, edge_index_relabeled, perm


def build_geo_node_partition_from_coordinates(node_coords: Tensor, num_partitions: int) -> Tensor:
    """Build a geographic node partition from latitude/longitude coordinates.

    The partitioning is intentionally simple and deterministic:
    - split nodes into northern vs southern hemisphere using latitude
    - subdivide each hemisphere into equal-width longitude bands from west to east

    This is intended as a lightweight, geometry-aware alternative to balanced
    contiguous partitioning for offline analysis and halo visualisation.

    Parameters
    ----------
    node_coords : Tensor
        Node coordinates with shape ``[num_nodes, 2]`` where column 0 is latitude
        and column 1 is longitude, both in radians.
    num_partitions : int
        Total number of partitions. Must be even so the hemispheres can be split
        into an equal number of longitude bands.

    Returns
    -------
    Tensor
        Integer tensor of shape ``[num_nodes]`` with partition ids.
    """
    if num_partitions < 2:
        raise ValueError(f"num_partitions must be >= 2, got {num_partitions}")
    if num_partitions % 2 != 0:
        raise ValueError(
            f"Geo partitioning requires an even num_partitions so north/south can be split equally, got {num_partitions}"
        )

    node_coords = node_coords.detach()
    lat = node_coords[:, 0]
    lon = node_coords[:, 1]

    lon_partitions_per_hemisphere = num_partitions // 2

    # 0 = north, 1 = south
    hemisphere_id = (lat < 0).to(torch.long)

    # Equal-width longitude bins from west (-pi) to east (+pi).
    lon_scaled = ((lon + torch.pi) / (2 * torch.pi)).clamp(0.0, 1.0 - torch.finfo(torch.float32).eps)
    lon_bin = torch.floor(lon_scaled * lon_partitions_per_hemisphere).to(torch.long)
    lon_bin = lon_bin.clamp(0, lon_partitions_per_hemisphere - 1)

    return hemisphere_id * lon_partitions_per_hemisphere + lon_bin


def build_metis_node_partition(
    edge_index: Adj,
    num_partitions: int,
    num_nodes: int,
    symmetrize: bool = False,
) -> Tensor:
    """Build a node partition using METIS graph partitioning.

    Converts the edge_index to CSR format and calls ``pyg_lib.partition.metis``
    to assign each node to a partition.

    Parameters
    ----------
    edge_index : Adj
        Edge index with shape ``[2, num_edges]``.  METIS requires an undirected
        graph, so the caller is responsible for passing edges in both directions
        (the typical anemoi-graphs convention).  Set ``symmetrize=True`` if the
        input is a directed graph that needs to be made undirected first.
    num_partitions : int
        Number of partitions to produce.
    num_nodes : int
        Total number of nodes in the graph.
    symmetrize : bool, optional
        If ``True``, symmetrize the adjacency and remove self-loops before
        building the CSR representation.  This is only needed when
        ``edge_index`` is a directed graph (edges present in only one
        direction).  Default: ``False``.

    Returns
    -------
    Tensor
        Integer tensor of shape ``[num_nodes]`` with partition ids in
        ``[0, num_partitions)``.
    """
    try:
        import pyg_lib  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pyg_lib is required for METIS partitioning. "
            "Install it with: pip install pyg-lib"
        ) from exc

    ei = edge_index
    if symmetrize:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
        ei = ei[:, ei[0] != ei[1]]  # remove self-loops

    # Sort by src for CSR construction; deduplicate if symmetrize added duplicates.
    order = torch.argsort(ei[0].long() * num_nodes + ei[1].long())
    ei = ei[:, order]
    if symmetrize and ei.size(1) > 1:
        keep = torch.ones(ei.size(1), dtype=torch.bool, device=edge_index.device)
        keep[1:] = (ei[:, 1:] != ei[:, :-1]).any(dim=0)
        ei = ei[:, keep]

    src = ei[0]
    col = ei[1]
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=edge_index.device)
    rowptr[1:] = torch.cumsum(torch.bincount(src, minlength=num_nodes), dim=0)

    return pyg_lib.partition.metis(
        rowptr=rowptr.cpu(),
        col=col.cpu(),
        num_partitions=num_partitions,
        recursive=False,
    ).to(edge_index.device)


def ensure_edges_are_dst_sorted(
    edge_attr: Tensor,
    edge_index: Adj,
    *,
    num_dst: int,
    edges_are_sharded: bool,
    model_comm_group: ProcessGroup | None = None,
    edges_are_dst_sorted: bool = True,
) -> tuple[Tensor, Adj]:
    """Ensure edge tensors are dst-sorted before GraphTransformer attention."""
    if edges_are_dst_sorted:
        return edge_attr, edge_index

    if edges_are_sharded and model_is_distributed(model_comm_group):
        msg = (
            "Edge-sharded GraphTransformer inputs must be dst-sorted before use. "
            "Sorting an already distributed edge shard would require gathering edge_attr and edge_index together."
        )
        raise ValueError(msg)

    edge_index, perm = sort_edge_index_by_dst(edge_index, max_value=num_dst)
    return edge_attr[perm], edge_index


def shard_edges_1hop(
    edge_attr: Tensor,
    edge_index: Adj,
    src_size: int,
    dst_size: int,
    model_comm_group: Optional[ProcessGroup],
    edges_are_dst_sorted: bool = True,
) -> tuple[Tensor, Adj, ShardSizes]:
    """Sort and shard edges for 1-hop sharding.

    Parameters
    ----------
    edge_attr : Tensor
        Edge attributes.
    edge_index : Adj
        Edge index.
    src_size : int
        Number of source nodes.
    dst_size : int
        Number of destination nodes.
    model_comm_group : ProcessGroup, optional
        Model communication group.
    edges_are_dst_sorted : bool, optional
        Whether `edge_index` and `edge_attr` are already ordered by destination node.
        Edges from graph providers already are. Pass False for custom full-graph
        edges that are not ordered this way. If edges are already sharded, each rank
        is expected to already have the right edges for its local destination nodes.

    Returns
    -------
    tuple[Tensor, Adj, ShardSizes]
        Sharded edge_attr, sharded edge_index, and edge_shard_sizes.
    """
    num_nodes = (src_size, dst_size)
    edge_shard_sizes = None

    if model_is_distributed(model_comm_group):
        if edges_are_dst_sorted:  # fast path: compute splits from degree
            num_parts = model_comm_group.size()
            edge_shard_sizes = build_graph_partition(edge_index, num_parts, num_nodes).edge_splits
        else:  # slow path: sort edges into 1-hop chunks via subgraph extraction
            edge_attr, edge_index, edge_shard_sizes = _sort_edges_1hop_sharding(
                num_nodes, edge_attr, edge_index, model_comm_group
            )

    edge_index = shard_tensor(edge_index, 1, edge_shard_sizes, model_comm_group)
    edge_attr = shard_tensor(edge_attr, 0, edge_shard_sizes, model_comm_group)

    return edge_attr, edge_index, edge_shard_sizes


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
    if not model_is_distributed(model_comm_group):
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


def sort_edges_1hop_chunks(
    num_nodes: Union[int, tuple[int, int]],
    edge_attr: Tensor,
    edge_index: Adj,
    num_chunks: int,
    edges_are_dst_sorted: bool = True,
) -> tuple[list[Tensor], list[Adj]]:
    """Split edges into 1-hop neighbourhood chunks.

    Supports two paths:
    - Fast path (edges_are_dst_sorted=True): O(1) slicing using precomputed partition splits.
    - Slow path (edges_are_dst_sorted=False): explicit subgraph extraction per chunk.

    Parameters
    ----------
    num_nodes : Union[int, tuple[int, int]]
        Number of target nodes in the graph, or tuple (src, dst) for a bipartite graph.
    edge_attr : Tensor
        Edge attributes.
    edge_index : Adj
        Edge index.
    num_chunks : int
        Number of chunks to split into.
    edges_are_dst_sorted : bool, optional
        Whether `edge_index` and `edge_attr` are already ordered by destination node.
        Edges from graph providers already are. Pass False for custom full-graph
        edges that are not ordered this way.

    Returns
    -------
    tuple[list[Tensor], list[Adj]]
        List of edge attribute chunks and edge index chunks.
    """
    if edges_are_dst_sorted:
        return _sort_edges_1hop_chunks_fast(num_nodes, edge_attr, edge_index, num_chunks)
    return _sort_edges_1hop_chunks_subgraph(num_nodes, edge_attr, edge_index, num_chunks)


def _sort_edges_1hop_chunks_fast(
    num_nodes: Union[int, tuple[int, int]],
    edge_attr: Tensor,
    edge_index: Adj,
    num_chunks: int,
) -> tuple[list[Tensor], list[Adj]]:
    """Fast O(1) chunking of dst-sorted edges using degree-based splits.

    Since edges are sorted by destination node, we can partition them by
    simply computing the cumulative degree per destination chunk.
    """
    num_nodes_tuple = (num_nodes, num_nodes) if isinstance(num_nodes, int) else num_nodes
    partition = build_graph_partition(edge_index, num_chunks, num_nodes_tuple)

    edge_attr_list = []
    edge_index_list = []
    for i in range(num_chunks):
        edge_range = partition._get_edge_range(i)
        edge_attr_list.append(edge_attr[edge_range])
        edge_index_list.append(edge_index[:, edge_range])

    return edge_attr_list, edge_index_list


def _drop_unconnected_src_nodes(x_src: Tensor, edge_index: Adj, in_place: bool = True) -> tuple[Tensor, Adj, Tensor]:
    """Drop src nodes with no edges and relabel src indices to be contiguous.

    Parameters
    ----------
    x_src : Tensor
        Source node features.
    edge_index : Adj
        Edge index (row 0 will be modified in-place if in_place=True).
    in_place : bool, optional
        Whether to modify edge_index in-place (default: True).
        If False, a clone is made before relabeling.

    Returns
    -------
    tuple[Tensor, Adj, Tensor]
        Subset of x_src, relabeled edge_index, indices of connected source nodes.
    """
    edge_index = edge_index if in_place else edge_index.clone()
    connected_src_nodes = torch.unique(edge_index[0])
    x_src_subset = x_src[connected_src_nodes]

    relabel_map = torch.empty(x_src.shape[0], dtype=torch.long, device=x_src.device)
    relabel_map[connected_src_nodes] = torch.arange(connected_src_nodes.size(0), device=x_src.device)
    edge_index[0] = relabel_map[edge_index[0]]

    return x_src_subset, edge_index, connected_src_nodes


########## Slow path: explicit subgraph extraction (used when edges are NOT pre-sorted). ##########


def _sort_edges_1hop_sharding(
    num_nodes: Union[int, tuple[int, int]],
    edge_attr: Tensor,
    edge_index: Adj,
    mgroup: Optional[ProcessGroup] = None,
) -> tuple[Tensor, Adj, ShardSizes]:
    """Rearrange edges into 1-hop neighbourhoods for sharding across GPUs.

    Uses explicit subgraph extraction. Prefer the fast path
    (dst-sorted edges + build_graph_partition) when possible.

    Parameters
    ----------
    num_nodes : Union[int, tuple[int, int]]
        Number of target nodes in the graph.
    edge_attr : Tensor
        Edge attributes.
    edge_index : Adj
        Edge index.
    mgroup : ProcessGroup
        Model communication group.

    Returns
    -------
    tuple[Tensor, Adj, ShardSizes]
        Edge attributes and edge indices sorted according to 1-hop neighbourhoods,
        plus edge shard sizes when a model communication group is provided.
    """
    if model_is_distributed(mgroup):
        num_chunks = dist.get_world_size(group=mgroup)

        edge_attr_list, edge_index_list = _sort_edges_1hop_chunks_subgraph(
            num_nodes,
            edge_attr,
            edge_index,
            num_chunks,
        )

        edge_shard_sizes = [e.shape[0] for e in edge_attr_list]

        return torch.cat(edge_attr_list, dim=0), torch.cat(edge_index_list, dim=1), edge_shard_sizes

    return edge_attr, edge_index, None


def _sort_edges_1hop_chunks_subgraph(
    num_nodes: Union[int, tuple[int, int]],
    edge_attr: Tensor,
    edge_index: Adj,
    num_chunks: int,
) -> tuple[list[Tensor], list[Adj]]:
    """Chunking via explicit subgraph extraction.

    For each destination chunk, extracts the subgraph containing all edges
    pointing to that chunk's nodes. Works with unsorted edges but is O(E)
    per chunk.

    Parameters
    ----------
    num_nodes : Union[int, tuple[int, int]]
        Number of target nodes in the graph, or tuple for a bipartite graph.
    edge_attr : Tensor
        Edge attributes.
    edge_index : Adj
        Edge index.
    num_chunks : int
        Number of chunks.

    Returns
    -------
    tuple[list[Tensor], list[Adj]]
        List of sorted edge attribute chunks and sorted edge index chunks.
    """
    from torch_geometric.utils import bipartite_subgraph

    if isinstance(num_nodes, int):
        node_chunks = torch.arange(num_nodes, device=edge_index.device).tensor_split(num_chunks)
    else:
        nodes_src = torch.arange(num_nodes[0], device=edge_index.device)
        node_chunks = torch.arange(num_nodes[1], device=edge_index.device).tensor_split(num_chunks)

    edge_index_list = []
    edge_attr_list = []
    for node_chunk in node_chunks:
        if isinstance(num_nodes, int):
            edge_attr_chunk, edge_index_chunk = _get_k_hop_edges(node_chunk, edge_attr, edge_index, num_nodes=num_nodes)
        else:
            edge_index_chunk, edge_attr_chunk = bipartite_subgraph(
                (nodes_src, node_chunk),
                edge_index,
                edge_attr,
                size=(num_nodes[0], num_nodes[1]),
            )

        edge_index_list.append(edge_index_chunk)
        edge_attr_list.append(edge_attr_chunk)

    return edge_attr_list, edge_index_list


def _get_k_hop_edges(
    nodes: Tensor,
    edge_attr: Tensor,
    edge_index: Adj,
    num_hops: int = 1,
    num_nodes: Optional[int] = None,
) -> tuple[Adj, Tensor]:
    """Return k-hop subgraph edges.

    Parameters
    ----------
    nodes : Tensor
        Destination nodes.
    edge_attr : Tensor
        Edge attributes.
    edge_index : Adj
        Edge index.
    num_hops : int, optional
        Number of required hops, by default 1.
    num_nodes : int, optional
        Total number of nodes.

    Returns
    -------
    tuple[Adj, Tensor]
        K-hop subgraph of edge attributes and edge index.
    """
    from torch_geometric.utils import k_hop_subgraph
    from torch_geometric.utils import mask_to_index

    _, edge_index_k, _, edge_mask_k = k_hop_subgraph(
        node_idx=nodes,
        num_hops=num_hops,
        edge_index=edge_index,
        directed=True,
        num_nodes=num_nodes,
    )

    return edge_attr[mask_to_index(edge_mask_k)], edge_index_k


def drop_unconnected_src_nodes(
    x_src: Tensor, edge_index: Adj, num_nodes: tuple[int, int]
) -> tuple[Tensor, Adj, Tensor]:
    """Drop unconnected nodes from x_src and relabel edges.

    Parameters
    ----------
    x_src : Tensor
        Source node features.
    edge_index : Adj
        Edge index.
    num_nodes : tuple[int, int]
        Number of nodes in graph (src, dst).

    Returns
    -------
    tuple[Tensor, Adj, Tensor]
        Reduced node features, relabeled edge index, and indices of connected source nodes.
    """
    from torch_geometric.utils import bipartite_subgraph

    connected_src_nodes = torch.unique(edge_index[0])
    dst_nodes = torch.arange(num_nodes[1], device=x_src.device)

    edge_index_new, _ = bipartite_subgraph(
        (connected_src_nodes, dst_nodes),
        edge_index,
        size=num_nodes,
        relabel_nodes=True,
    )

    return x_src[connected_src_nodes], edge_index_new, connected_src_nodes
