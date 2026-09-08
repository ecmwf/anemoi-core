# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.khop_edges import GraphPartition
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes


@dataclass(frozen=True)
class HaloInfo:
    """Per-rank halo exchange metadata for distributed graph processing.

    Precomputed once from graph topology, reused across layers and time steps.
    All index tensors use local node numbering:

    - Local (inner) nodes: ``[0, num_local_nodes)``
    - Halo nodes: ``[num_local_nodes, num_local_nodes + num_halo_nodes)``
      ordered by source rank, then by global node ID within each rank.

    Parameters
    ----------
    num_local_nodes : int
        Number of inner (owned) nodes on this rank.
    num_halo_nodes : int
        Total number of halo nodes received from all other ranks.
    send_indices : tuple[Tensor, ...]
        Per-rank local indices of inner nodes to send.  Length = world size.
        ``send_indices[r]`` contains the local indices to gather for rank *r*.
    recv_counts : tuple[int, ...]
        Per-rank number of halo nodes to receive.  Length = world size.
    recv_global_ids : tuple[Tensor, ...] | None
        Per-rank global node IDs of halo nodes to receive.  Length = world size.
        Sorted within each rank.  Only populated when ``debug=True`` in
        :func:`build_halo_info`; ``None`` otherwise.
    edge_index_local : Tensor
        Edge index relabeled to local + halo node IDs.
        Shape ``(2, num_local_edges)``. Row 0 uses ``[0, total_src_nodes)``
        and row 1 uses ``[0, num_local_dst_nodes)``.
    num_local_dst_nodes : int | None
        Number of owned destination nodes. Defaults to ``num_local_nodes``
        for homogeneous graphs and compatibility with existing constructors.
    """

    num_local_nodes: int
    num_halo_nodes: int
    send_indices: tuple[Tensor, ...]
    recv_counts: tuple[int, ...]
    recv_global_ids: Optional[tuple[Tensor, ...]]
    edge_index_local: Tensor
    num_local_dst_nodes: Optional[int] = None

    @property
    def num_local_src_nodes(self) -> int:
        """Number of owned source nodes on this rank."""
        return self.num_local_nodes

    @property
    def total_src_nodes(self) -> int:
        """Total number of owned and halo source nodes."""
        return self.num_local_src_nodes + self.num_halo_nodes

    @property
    def total_nodes(self) -> int:
        """Total number of source nodes, retained for compatibility."""
        return self.total_src_nodes

    @property
    def local_dst_nodes(self) -> int:
        """Number of owned destination nodes on this rank."""
        return self.num_local_nodes if self.num_local_dst_nodes is None else self.num_local_dst_nodes

    @property
    def send_counts(self) -> tuple[int, ...]:
        """Per-rank number of nodes to send."""
        return tuple(t.size(0) for t in self.send_indices)


def cache_specs(
    shard_info: GraphShardInfo | BipartiteGraphShardInfo,
    model_comm_group: ProcessGroup,
) -> tuple[int | tuple[int, ...], ...]:
    """Return specs that determine whether cached halo metadata can be reused."""
    if isinstance(shard_info, BipartiteGraphShardInfo):
        shard_specs = (
            tuple(shard_info.src_nodes or ()),
            tuple(shard_info.dst_nodes or ()),
            tuple(shard_info.edges or ()),
        )
    else:
        shard_specs = (tuple(shard_info.nodes or ()), tuple(shard_info.edges or ()))

    return (
        model_comm_group.size(),
        torch.distributed.get_rank(group=model_comm_group),
        *shard_specs,
    )


def _node_id_to_partition_id(node_ids: Tensor, partition_sizes: list[int]) -> Tensor:
    """Map global node IDs to their owning partition.

    Parameters
    ----------
    node_ids : Tensor
        Global node IDs.
    partition_sizes : list[int]
        Per-partition node counts (e.g. ``GraphPartition.dst_splits``).

    Returns
    -------
    Tensor
        Partition ID for each input node.
    """
    cumulative = torch.cumsum(torch.tensor(partition_sizes, device=node_ids.device, dtype=torch.long), dim=0)
    return torch.searchsorted(cumulative, node_ids, right=True)


def _finalize_halo_info(
    *,
    partition: GraphPartition,
    local_edge_index: Tensor,
    send_nodes_by_rank: list[Tensor],
    recv_nodes_by_rank: list[Tensor],
    src_start: int,
    src_stop: int,
    dst_start: int,
    dst_stop: int,
    model_comm_group: ProcessGroup,
    debug: bool,
) -> HaloInfo:
    """Build local indexing and exchange metadata from per-rank global node IDs."""
    send_nodes_by_rank = [nodes.unique(sorted=True) for nodes in send_nodes_by_rank]
    recv_nodes_by_rank = [nodes.unique(sorted=True) for nodes in recv_nodes_by_rank]
    send_indices = tuple(nodes - src_start for nodes in send_nodes_by_rank)
    recv_counts = tuple(nodes.size(0) for nodes in recv_nodes_by_rank)

    num_local_src_nodes = src_stop - src_start
    all_halo_nodes = torch.cat(recv_nodes_by_rank)
    num_halo_nodes = all_halo_nodes.size(0)

    edge_index_local = local_edge_index.clone()
    edge_index_local[1] -= dst_start

    src_global = local_edge_index[0]
    is_local_src = (src_global >= src_start) & (src_global < src_stop)
    edge_index_local[0, is_local_src] = src_global[is_local_src] - src_start

    if num_halo_nodes > 0:
        halo_relabel = torch.empty(partition.num_nodes[0], dtype=torch.long, device=edge_index_local.device)
        halo_relabel[all_halo_nodes] = (
            torch.arange(num_halo_nodes, device=edge_index_local.device) + num_local_src_nodes
        )
        edge_index_local[0, ~is_local_src] = halo_relabel[src_global[~is_local_src]]

    halo_info = HaloInfo(
        num_local_nodes=num_local_src_nodes,
        num_halo_nodes=num_halo_nodes,
        send_indices=send_indices,
        recv_counts=recv_counts,
        recv_global_ids=tuple(recv_nodes_by_rank) if debug else None,
        edge_index_local=edge_index_local,
        num_local_dst_nodes=dst_stop - dst_start,
    )
    if debug:
        verify_halo_info(halo_info, partition, model_comm_group)
    return halo_info


def build_halo_info(
    partition: GraphPartition,
    edge_index: Tensor,
    model_comm_group: ProcessGroup,
    edge_shard_sizes: ShardSizes = None,
    debug: bool = False,
) -> HaloInfo:
    """Build per-rank halo exchange metadata from graph partitioning.

    Identifies which inner nodes need to be sent to peer ranks and which
    halo nodes need to be received, then relabels the local edge_index to
    use contiguous local + halo node IDs.

    Parameters
    ----------
    partition : GraphPartition
        Global partitioning metadata.  ``partition.num_parts`` must equal
        the communication group size.
    edge_index : Tensor
        Edge index with **global** (un-relabeled) node IDs, sorted by
        destination node.  May be either the full graph or already sharded
        to the local rank (see *edge_shard_sizes*).
    model_comm_group : ProcessGroup
        Model communication group.
    edge_shard_sizes : ShardSizes, optional
        If not ``None``, *edge_index* is already sharded for this rank
        (contains only local edges).  If ``None``, *edge_index* is the
        full (global) edge set and will be sliced using the partition.
    debug : bool, optional
        If ``True``, store ``recv_global_ids`` in the returned
        :class:`HaloInfo` for use with :func:`verify_halo_info`.
        Default ``False``.

    Returns
    -------
    HaloInfo
        Per-rank halo exchange metadata.
    """
    my_rank = torch.distributed.get_rank(group=model_comm_group)
    num_parts = model_comm_group.size()

    assert (
        partition.num_parts == num_parts
    ), f"Partition num_parts ({partition.num_parts}) != comm group size ({num_parts})"

    # local edges for this rank
    if edge_shard_sizes is not None:
        local_edge_index = edge_index
    else:
        local_edge_index = shard_tensor(edge_index, 1, partition.edge_splits, model_comm_group)

    # partition range for this rank
    dst_start, dst_end = get_partition_range(partition.dst_splits, my_rank)
    # identify halo src nodes (outside this rank's partition range)
    # i.e. (halo_src, dst) edges where dst is local but src is not
    src_global = local_edge_index[0]
    # boolean mask for edges where src is a halo node
    is_halo_src = (src_global < dst_start) | (src_global >= dst_end)

    # map halo src nodes to their owning partition
    halo_src_global = src_global[is_halo_src]
    halo_partition_ids = _node_id_to_partition_id(halo_src_global, partition.dst_splits)

    # Processor graphs are symmetric; asymmetric graph connections are represented as mappers.
    # Due to undirected graph symmetry, each (halo_src, dst) edge corresponds to a (dst, halo_src)
    # edge in an other rank's partition, so we will receive halo_src as a halo node from that rank,
    # and send dst as an inner node to that rank.
    halo_dst_global = local_edge_index[1, is_halo_src]

    # build per-rank send / recv info
    send_nodes_list: list[Tensor] = []
    recv_nodes_list: list[Tensor] = []

    for rank in range(num_parts):
        rank_mask = halo_partition_ids == rank

        # recv: unique halo node global IDs from this rank
        recv_nodes_list.append(halo_src_global[rank_mask])
        send_nodes_list.append(halo_dst_global[rank_mask])

    return _finalize_halo_info(
        partition=partition,
        local_edge_index=local_edge_index,
        send_nodes_by_rank=send_nodes_list,
        recv_nodes_by_rank=recv_nodes_list,
        src_start=dst_start,
        src_stop=dst_end,
        dst_start=dst_start,
        dst_stop=dst_end,
        model_comm_group=model_comm_group,
        debug=debug,
    )


def build_halo_info_bipartite(
    partition: GraphPartition,
    edge_index: Tensor,
    model_comm_group: ProcessGroup,
    edge_shard_sizes: ShardSizes = None,
    debug: bool = False,
) -> HaloInfo:
    """Build per-rank halo exchange metadata from graph partitioning.

    Identifies which inner nodes need to be sent to peer ranks and which
    halo nodes need to be received, then relabels the local edge_index to
    use contiguous local + halo node IDs.

    Parameters
    ----------
    partition : GraphPartition
        Global partitioning metadata.  ``partition.num_parts`` must equal
        the communication group size.
    edge_index : Tensor
        Edge index with **global** (un-relabeled) node IDs, sorted by
        destination node.  May be either the full graph or already sharded
        to the local rank (see *edge_shard_sizes*).
    model_comm_group : ProcessGroup
        Model communication group.
    edge_shard_sizes : ShardSizes, optional
        If not ``None``, *edge_index* is already sharded for this rank
        (contains only local edges).  If ``None``, *edge_index* is the
        full (global) edge set and will be sliced using the partition.
    debug : bool, optional
        If ``True``, store ``recv_global_ids`` in the returned
        :class:`HaloInfo` for use with :func:`verify_halo_info`.
        Default ``False``.

    Returns
    -------
    HaloInfo
        Per-rank halo exchange metadata.
    """
    my_rank = torch.distributed.get_rank(group=model_comm_group)
    num_parts = model_comm_group.size()

    assert (
        partition.num_parts == num_parts
    ), f"Partition num_parts ({partition.num_parts}) != comm group size ({num_parts})"

    assert partition.src_splits is not None, "Bipartite partition must have src_splits for halo info"

    if edge_shard_sizes is not None:
        local_edge_index = edge_index
        global_edge_index = gather_tensor(edge_index, 1, edge_shard_sizes, model_comm_group)
    else:
        local_edge_index = shard_tensor(edge_index, 1, partition.edge_splits, model_comm_group)
        global_edge_index = edge_index

    # partition range for this rank
    dst_start, dst_stop = get_partition_range(partition.dst_splits, my_rank)
    src_start, src_stop = get_partition_range(partition.src_splits, my_rank)

    # src_edges_out: mask for edges whose source node is within this rank's source range (delta^{+}(src_start, src_stop))
    src_edges_out = (global_edge_index[0, :] >= src_start) & (global_edge_index[0, :] < src_stop)
    # dst_edges_in: mask for edges whose destination node is within this rank's destination range (delta^{-}(dst_start, dst_stop))
    dst_edges_in = (global_edge_index[1, :] >= dst_start) & (global_edge_index[1, :] < dst_stop)

    # halo masks for edges whose source nodes need to be sent/received (destination rank given by dst node partition)
    # rank i receives v_j for local edges (v_j, w_i), and sends v_i for remote edges (v_i, w_j)
    is_halo_recv_edge = dst_edges_in & ~src_edges_out
    is_halo_send_edge = src_edges_out & ~dst_edges_in

    # halo nodes corresponding to send/recv edges (potentially duplicated)
    halo_recv_nodes = global_edge_index[0, is_halo_recv_edge]
    halo_recv_partition_ids = _node_id_to_partition_id(halo_recv_nodes, partition.src_splits)
    halo_send_nodes = global_edge_index[0, is_halo_send_edge]
    # halo send partition_ids keyed by partition of corresponding dst node
    halo_send_partition_ids = _node_id_to_partition_id(global_edge_index[1, is_halo_send_edge], partition.dst_splits)

    send_nodes_list: list[Tensor] = []
    recv_nodes_list: list[Tensor] = []

    for rank in range(num_parts):
        rank_mask_recv = halo_recv_partition_ids == rank
        rank_mask_send = halo_send_partition_ids == rank

        recv_nodes_list.append(halo_recv_nodes[rank_mask_recv])
        send_nodes_list.append(halo_send_nodes[rank_mask_send])

    return _finalize_halo_info(
        partition=partition,
        local_edge_index=local_edge_index,
        send_nodes_by_rank=send_nodes_list,
        recv_nodes_by_rank=recv_nodes_list,
        src_start=src_start,
        src_stop=src_stop,
        dst_start=dst_start,
        dst_stop=dst_stop,
        model_comm_group=model_comm_group,
        debug=debug,
    )


def verify_halo_info(
    halo_info: HaloInfo,
    partition: GraphPartition,
    model_comm_group: ProcessGroup,
) -> None:
    """Verify send/recv symmetry of halo metadata across all ranks (for debugging)

    Checks that ``send(i, j) == recv(j, i)`` in terms of global node IDs,
    i.e. the set of inner nodes that rank *i* sends to rank *j* equals the
    set of halo nodes that rank *j* expects to receive from rank *i*.

    This is a **collective** operation — all ranks in the group must call
    it.  Intended for debugging only (involves all-to-all communication).

    Parameters
    ----------
    halo_info : HaloInfo
        Per-rank halo metadata (as returned by :func:`build_halo_info`).
    partition : GraphPartition
        Global partitioning metadata (needed to map local send indices
        back to global node IDs).
    model_comm_group : ProcessGroup
        Model communication group.

    Raises
    ------
    AssertionError
        If any send/recv pair is inconsistent.
    """
    assert (
        halo_info.recv_global_ids is not None
    ), "verify_halo_info requires recv_global_ids; rebuild HaloInfo with debug=True"
    my_rank = torch.distributed.get_rank(group=model_comm_group)
    num_parts = model_comm_group.size()
    device = halo_info.edge_index_local.device

    # local send indices → global node IDs
    if partition.src_splits is not None:
        offset = get_partition_range(partition.src_splits, my_rank)[0]
    else:
        offset = get_partition_range(partition.dst_splits, my_rank)[0]

    send_global = [idx + offset for idx in halo_info.send_indices]

    # all-to-all: each rank sends what it will send (global IDs) to each
    # peer; each rank receives what peers intend to send to it.
    recv_global = torch.empty(sum(halo_info.recv_counts), dtype=torch.long, device=device)
    torch.distributed.all_to_all_single(
        recv_global,
        torch.cat(send_global),
        output_split_sizes=list(halo_info.recv_counts),
        input_split_sizes=list(halo_info.send_counts),
        group=model_comm_group,
    )
    recv_from_peers = recv_global.split(halo_info.recv_counts)

    # recv_from_peers[r] now contains the global IDs that rank r will send
    # to us.  By symmetry this must equal recv_global_ids[r] — the halo
    # nodes we expect from rank r.  Both are sorted, so compare directly.
    for rank in range(num_parts):
        received = recv_from_peers[rank]
        expected = halo_info.recv_global_ids[rank]
        assert received.shape == expected.shape and torch.equal(received, expected), (
            f"Rank {my_rank}: halo symmetry violation with rank {rank} — "
            f"send({rank},{my_rank}) has {received.numel()} nodes, "
            f"recv({my_rank},{rank}) has {expected.numel()} nodes"
        )
