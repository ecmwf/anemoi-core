# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.balanced_partition import get_partition_range
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.khop_edges import GraphPartition
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes


@dataclass(frozen=True)
class HaloInfo:
    """Per-rank halo exchange metadata for distributed graph processing.

    Precomputed once from graph topology, reused across layers and time steps.
    Source nodes use local numbering:

    - Local (inner) source nodes: ``[0, num_local_src_nodes)``
    - Halo nodes: ``[num_local_src_nodes, total_src_nodes)``
      ordered by source rank, then by global node ID within each rank.

    Destination nodes are always owned by this rank and use ``[0, num_local_dst_nodes)``.

    Parameters
    ----------
    num_local_src_nodes : int
        Number of source nodes owned by this rank.
    num_local_dst_nodes : int
        Number of destination nodes owned by this rank.
    num_halo_nodes : int
        Total number of halo nodes received from all other ranks.
    send_indices : tuple[Tensor, ...]
        Per-rank local indices of inner source nodes to send.  Length = world size.
        ``send_indices[r]`` contains the local indices to gather for rank *r*.
    recv_counts : tuple[int, ...]
        Per-rank number of halo nodes to receive.  Length = world size.
    edge_index_local : Tensor
        Edge index relabeled to local + halo node IDs.
        Shape ``(2, num_local_edges)``. Row 0 uses ``[0, total_src_nodes)``
        and row 1 uses ``[0, num_local_dst_nodes)``.
    """

    num_local_src_nodes: int
    num_local_dst_nodes: int
    num_halo_nodes: int
    send_indices: tuple[Tensor, ...]
    recv_counts: tuple[int, ...]
    edge_index_local: Tensor

    @property
    def total_src_nodes(self) -> int:
        """Total number of owned and halo source nodes."""
        return self.num_local_src_nodes + self.num_halo_nodes

    @property
    def send_counts(self) -> tuple[int, ...]:
        """Per-rank number of nodes to send."""
        return tuple(t.size(0) for t in self.send_indices)


def cache_specs(
    shard_info: GraphShardInfo,
    model_comm_group: ProcessGroup,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    """Return specs that determine whether cached halo metadata can be reused."""
    return (
        model_comm_group.size(),
        torch.distributed.get_rank(group=model_comm_group),
        tuple(shard_info.nodes or ()),
        tuple(shard_info.edges or ()),
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


def _request_send_nodes(recv_nodes_by_rank: tuple[Tensor, ...], model_comm_group: ProcessGroup) -> tuple[Tensor, ...]:
    """Tell every rank which of its nodes this rank needs, and learn which nodes the other ranks need from us.

    This is a **collective** operation — all ranks in the group must call it.

    Parameters
    ----------
    recv_nodes_by_rank : tuple[Tensor, ...]
        Per-rank global IDs of the halo nodes this rank needs from that rank.
    model_comm_group : ProcessGroup
        Model communication group.

    Returns
    -------
    tuple[Tensor, ...]
        Per-rank global IDs of this rank's nodes that rank needs, in the order
        that rank asked for them.
    """
    device = recv_nodes_by_rank[0].device
    recv_counts = torch.tensor([nodes.size(0) for nodes in recv_nodes_by_rank], dtype=torch.long, device=device)
    send_counts = torch.empty_like(recv_counts)
    dist.all_to_all_single(send_counts, recv_counts, group=model_comm_group)

    send_counts = send_counts.tolist()
    send_nodes = torch.empty(sum(send_counts), dtype=torch.long, device=device)
    dist.all_to_all_single(
        send_nodes,
        torch.cat(recv_nodes_by_rank),
        output_split_sizes=send_counts,
        input_split_sizes=recv_counts.tolist(),
        group=model_comm_group,
    )
    return send_nodes.split(send_counts)


def build_halo_info(
    partition: GraphPartition,
    edge_index: Tensor,
    model_comm_group: ProcessGroup,
    edge_shard_sizes: ShardSizes = None,
    debug: bool = False,
) -> HaloInfo:
    """Build halo metadata for homogeneous or bipartite graphs.

    Finds the source nodes of this rank's edges that live on other ranks,
    asks their owners for them, and learns in return which of its own source
    nodes the other ranks need. Then relabels the local edge_index to use
    contiguous local + halo node IDs.

    This is a **collective** operation — all ranks in the group must call it.

    Parameters
    ----------
    partition : GraphPartition
        Global partitioning metadata.  ``partition.num_parts`` must equal
        the communication group size, and ``partition.src_splits`` must give
        the source node ownership.
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
        If ``True``, check that every local edge ends on a destination node
        owned by this rank, and that the other ranks only ask for source
        nodes this rank owns.
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
    assert partition.src_splits is not None, "Halo exchange needs sharded source nodes (partition.src_splits is None)"

    if edge_shard_sizes is not None:
        local_edge_index = edge_index
    else:
        local_edge_index = shard_tensor(edge_index, 1, partition.edge_splits, model_comm_group)

    src_start, src_stop = get_partition_range(partition.src_splits, my_rank)
    dst_start, dst_stop = get_partition_range(partition.dst_splits, my_rank)
    num_local_src_nodes = src_stop - src_start

    if debug:
        local_dst = local_edge_index[1]
        assert (
            (local_dst >= dst_start) & (local_dst < dst_stop)
        ).all(), f"Rank {my_rank}: local edges must end on this rank's destination nodes [{dst_start}, {dst_stop})"

    # Halo nodes are the sources of local edges that another rank owns, grouped by owner.
    local_src = local_edge_index[0]
    is_remote_src = (local_src < src_start) | (local_src >= src_stop)
    remote_src = local_src[is_remote_src]
    owners = _node_id_to_partition_id(remote_src, partition.src_splits)
    recv_nodes_by_rank = tuple(remote_src[owners == rank].unique(sorted=True) for rank in range(num_parts))

    send_nodes_by_rank = _request_send_nodes(recv_nodes_by_rank, model_comm_group)
    if debug:
        for rank, nodes in enumerate(send_nodes_by_rank):
            assert (
                (nodes >= src_start) & (nodes < src_stop)
            ).all(), f"Rank {my_rank}: rank {rank} asked for source nodes this rank does not own"

    halo_nodes = torch.cat(recv_nodes_by_rank)
    num_halo_nodes = halo_nodes.size(0)

    # Relabel local edge index to account for local and halo nodes
    edge_index_local = local_edge_index.clone()
    edge_index_local[1] -= dst_start
    edge_index_local[0, ~is_remote_src] = local_src[~is_remote_src] - src_start
    if num_halo_nodes > 0:
        halo_relabel = torch.empty(partition.num_nodes[0], dtype=torch.long, device=edge_index.device)
        halo_relabel[halo_nodes] = torch.arange(num_halo_nodes, device=edge_index.device) + num_local_src_nodes
        edge_index_local[0, is_remote_src] = halo_relabel[remote_src]

    return HaloInfo(
        num_local_src_nodes=num_local_src_nodes,
        num_local_dst_nodes=dst_stop - dst_start,
        num_halo_nodes=num_halo_nodes,
        send_indices=tuple(nodes - src_start for nodes in send_nodes_by_rank),
        recv_counts=tuple(nodes.size(0) for nodes in recv_nodes_by_rank),
        edge_index_local=edge_index_local,
    )
