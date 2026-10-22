# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared utility functions for graphs in distributed tests.

These factories return complete synthetic graphs, not rank-local shards.
Connectivity is chosen to exercise remote reads when the returned ownership
sizes are used. Loss-data generation supplies fixed targets and weights for
the numerical reference comparisons.
"""

import torch
from torch_geometric.data import HeteroData


def build_mapper_graph(
    world_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Build a full bipartite graph designed for the returned shard sizes.

    With source sizes [3, 2, ...] and destination sizes [4, 3, ...], each
    destination reads one source in its own partition and one in the next.
    Edges are destination-sorted. Destination zero has a parallel remote edge
    with distinct attributes. No communication or rank-local slicing occurs.

    Returns complete edge attributes and indices allocated on ``device``,
    followed by source and destination sizes. The caller supplies the layout
    metadata and slices inputs when executing the distributed comparison.
    """
    # Unequal source/destination ownership exercises different global offsets.
    src_splits = [3] + [2] * (world_size - 1)
    dst_splits = [4] + [3] * (world_size - 1)
    src_starts = [sum(src_splits[:rank]) for rank in range(world_size)]
    dst_starts = [sum(dst_splits[:rank]) for rank in range(world_size)]
    edges = []
    for rank in range(world_size):
        remote_src_rank = (rank + 1) % world_size
        for offset in range(dst_splits[rank]):
            dst = dst_starts[rank] + offset
            # Cycle within each source shard when destination shards are larger.
            remote_src = src_starts[remote_src_rank] + offset % src_splits[remote_src_rank]
            local_src = src_starts[rank] + offset % src_splits[rank]
            edges.extend([(remote_src, dst), (local_src, dst)])
            if dst == 0:
                edges.append((remote_src, dst))
    # PyG stores edges as [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).T.contiguous()
    edge_attr = torch.arange(len(edges) * 3, dtype=torch.float32, device=device).reshape(-1, 3) / 17 - 0.4

    return edge_attr, edge_index, src_splits, dst_splits


def build_model_graph(world_size: int) -> tuple[HeteroData, list[int]]:
    """Build a full input/hidden graph for encoder-processor-decoder comparisons.

    The input/output node set is named "data". Intended data partitions have
    sizes [4, 3, ...]; hidden partitions have sizes [3, 2, ...]. Each destination
    has an owned and a next-partition source in every relation. Coordinates and
    edge attributes are seeded synthetic values, independent of connectivity.
    Returns the global CPU graph and intended data shard sizes. Hidden sizes
    are calculated by the model from its hidden-node tensor.
    """
    data_splits = [4] + [3] * (world_size - 1)
    hidden_splits = [3] + [2] * (world_size - 1)
    graph = HeteroData()
    generator = torch.Generator().manual_seed(31)
    # Initialize each node with a random coordinate pair.
    for name, splits in (("data", data_splits), ("hidden", hidden_splits)):
        graph[name].x = torch.randn(sum(splits), 2, generator=generator)
        graph[name].num_nodes = sum(splits)

    relations = (
        ("data", "hidden", data_splits, hidden_splits),
        ("hidden", "hidden", hidden_splits, hidden_splits),
        ("hidden", "data", hidden_splits, data_splits),
    )
    for src_name, dst_name, src_splits, dst_splits in relations:
        src_starts = [sum(src_splits[:rank]) for rank in range(world_size)]
        dst_starts = [sum(dst_splits[:rank]) for rank in range(world_size)]
        edges = []
        for rank in range(world_size):
            remote_src_rank = (rank + 1) % world_size
            for offset in range(dst_splits[rank]):
                dst = dst_starts[rank] + offset
                edges.extend(
                    [
                        (src_starts[rank] + offset % src_splits[rank], dst),
                        (src_starts[remote_src_rank] + offset % src_splits[remote_src_rank], dst),
                    ]
                )
        # PyG stores edges as [2, num_edges]
        relation = (src_name, "to", dst_name)
        graph[relation].edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
        graph[relation].edge_attr = torch.randn(len(edges), 3, generator=generator)
    return graph, data_splits


def mapper_loss_data(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic targets and nonuniform weights in global row order."""
    target = torch.linspace(-0.3, 0.7, output.numel(), device=output.device).reshape_as(output)
    weights = (torch.arange(output.numel(), device=output.device).reshape_as(output) % 7 + 1).float() / 7
    return target, weights
