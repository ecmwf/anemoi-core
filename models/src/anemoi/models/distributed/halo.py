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
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.khop_edges import GraphPartition
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
        Shape ``(2, num_local_edges)``.  Row 0 (src) uses ``[0, total_nodes)``,
        row 1 (dst) uses ``[0, num_local_nodes)``.
    """

    num_local_nodes: int
    num_halo_nodes: int
    send_indices: tuple[Tensor, ...]
    recv_counts: tuple[int, ...]
    recv_global_ids: Optional[tuple[Tensor, ...]]
    edge_index_local: Tensor

    @property
    def total_nodes(self) -> int:
        """Total number of nodes (local + halo) for attention size."""
        return self.num_local_nodes + self.num_halo_nodes

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
    num_local_nodes = dst_end - dst_start

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
    send_indices_list: list[Tensor] = []
    recv_nodes_list: list[Tensor] = []

    for rank in range(num_parts):
        rank_mask = halo_partition_ids == rank

        # recv: unique halo node global IDs from this rank
        recv_nodes = halo_src_global[rank_mask].unique(sorted=True)
        # send: unique inner node local indices to send to this rank
        send_nodes_global = halo_dst_global[rank_mask].unique(sorted=True)
        send_nodes_local = send_nodes_global - dst_start

        send_indices_list.append(send_nodes_local)
        recv_nodes_list.append(recv_nodes)

    recv_counts = tuple(t.size(0) for t in recv_nodes_list)

    all_halo_nodes = torch.cat(recv_nodes_list)  # disjoint per rank
    num_halo_nodes = all_halo_nodes.size(0)

    # relabel edge_index to local [0, num_local_nodes) + halo nodes [num_local_nodes, num_local_nodes + num_halo_nodes)
    edge_index_local = local_edge_index.clone()

    # dst: global → local [0, num_local_nodes)
    edge_index_local[1] -= dst_start

    # src – inner nodes: global → local [0, num_local_nodes)
    inner_src_mask = ~is_halo_src
    edge_index_local[0, inner_src_mask] = src_global[inner_src_mask] - dst_start

    # src – halo nodes: global → [num_local_nodes, total_nodes)
    if num_halo_nodes > 0:
        n_total_nodes = partition.num_nodes[0]
        halo_relabel = torch.empty(n_total_nodes, dtype=torch.long, device=edge_index.device)
        halo_relabel[all_halo_nodes] = torch.arange(num_halo_nodes, device=edge_index.device) + num_local_nodes
        edge_index_local[0, is_halo_src] = halo_relabel[src_global[is_halo_src]]

    return HaloInfo(
        num_local_nodes=num_local_nodes,
        num_halo_nodes=num_halo_nodes,
        send_indices=tuple(send_indices_list),
        recv_counts=recv_counts,
        recv_global_ids=tuple(recv_nodes_list) if debug else None,
        edge_index_local=edge_index_local,
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
    dst_start = get_partition_range(partition.dst_splits, my_rank)[0]
    device = halo_info.edge_index_local.device

    # local send indices → global node IDs
    send_global = [idx + dst_start for idx in halo_info.send_indices]

    # all-to-all: each rank sends what it will send (global IDs) to each
    # peer; each rank receives what peers intend to send to it.
    recv_from_peers = [torch.empty(halo_info.recv_counts[r], dtype=torch.long, device=device) for r in range(num_parts)]
    torch.distributed.all_to_all(recv_from_peers, list(send_global), group=model_comm_group)

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
