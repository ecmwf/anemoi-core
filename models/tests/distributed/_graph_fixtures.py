# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Controlled graph inputs shared by mapper sharding tests."""

import torch


def build_mapper_graph(
    world_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Build a bipartite mapper graph with uneven ownership and remote dependencies.

    Edges are destination-sorted, with the remote source first for each
    destination. The repeated remote edge entering destination zero has its
    own attribute row, so parallel edges remain distinguishable.
    """
    src_splits = [3] + [2] * (world_size - 1)
    dst_splits = [4] + [3] * (world_size - 1)
    src_starts = [sum(src_splits[:rank]) for rank in range(world_size)]
    edges = []
    for rank in range(world_size):
        remote_src_rank = (rank + 1) % world_size
        for offset in range(dst_splits[rank]):
            dst = sum(dst_splits[:rank]) + offset
            # Cycle within each source shard when destination shards are larger.
            local_src = src_starts[rank] + offset % src_splits[rank]
            remote_src = src_starts[remote_src_rank] + offset % src_splits[remote_src_rank]
            edges.extend([(remote_src, dst), (local_src, dst)])
            if dst == 0:
                edges.append((remote_src, dst))
    # PyG stores edges as [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).T.contiguous()
    edge_attr = torch.arange(len(edges) * 3, dtype=torch.float32, device=device).reshape(-1, 3) / 17 - 0.4

    assert sum(src_splits) != sum(dst_splits), "Mapper fixture must be bipartite with unequal node counts"
    assert (edge_index[1].diff() >= 0).all(), "Mapper edges must be destination-sorted"
    assert torch.equal(edge_index[:, 0], edge_index[:, 2]), "First destination must have a parallel remote edge"
    assert not torch.equal(edge_attr[0], edge_attr[2]), "Parallel edges must have distinct attributes"
    for rank in range(world_size):
        start, stop = sum(dst_splits[:rank]), sum(dst_splits[: rank + 1])
        sources = edge_index[0, (edge_index[1] >= start) & (edge_index[1] < stop)]
        owned = (sources >= src_starts[rank]) & (sources < src_starts[rank] + src_splits[rank])
        assert (~owned).any(), f"Missing remote source dependency for rank {rank}"
        assert sources[owned].unique().numel() > 1, f"Insufficient owned source variation for rank {rank}"
    return edge_attr, edge_index, src_splits, dst_splits


def mapper_loss_data(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic targets and nonuniform weights in global row order."""
    target = torch.linspace(-0.3, 0.7, output.numel(), device=output.device).reshape_as(output)
    weights = (torch.arange(output.numel(), device=output.device).reshape_as(output) % 7 + 1).float() / 7
    return target, weights
