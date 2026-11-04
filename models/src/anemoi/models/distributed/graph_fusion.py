# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Merge several source datasets into one bipartite graph, for joint encoding.

A joint encoder runs a single mapper pass over the union of its source datasets' nodes, so each
hidden node attends to every source dataset at once. That needs one shared source index space:
the per-dataset rows are interleaved into it, and each dataset's encoder edges are relabelled into
it and concatenated. The destination (hidden) side is untouched, since every source of an encoder
targets the same hidden nodes.

The merged space is a table of **blocks**. Block by block, and inside a block dataset by dataset
(in merge order), the rows are laid out back to back. A block is

* a **rank** when the model is sharded: ``gather_tensor`` concatenates rank shards in rank order,
  so a source's global ids are rank-major. Sharding requires ``batch_size == 1``.
* a **batch element** otherwise: graph providers lay out source ids batch-major (static providers
  offset each copy by ``i * src_size``, dynamic providers build one graph per sample block), and
  flattened tabular sources concatenate their samples the same way.

Inside a block, each dataset's rows stay contiguous and in their original order, so relabelling a
source id is a shift by how far its block moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

import torch
from torch import Tensor
from torch_geometric.typing import Adj

from anemoi.models.distributed.khop_edges import sort_edge_index_by_dst
from anemoi.models.distributed.shapes import ShardSizes

__all__ = ["SourceLayout", "FusedSourceIndex", "fuse_encoder_edges"]


@dataclass(frozen=True)
class SourceLayout:
    """How one source dataset's rows are laid out on this rank.

    Attributes
    ----------
    name : str
        Dataset name.
    local_rows : int
        Number of source rows this rank holds.
    shard_sizes : ShardSizes
        Per-rank row counts when the model is sharded. Ignored when it is not.
    batch_sizes : tuple[int, ...] or None
        Row count per batch element (sample x member) of a tabular dataset. ``None`` for a gridded
        dataset, whose batch elements all hold the same number of rows.
    """

    name: str
    local_rows: int
    shard_sizes: ShardSizes = None
    batch_sizes: tuple[int, ...] | None = None

    def blocks(self, *, batch_size: int, world_size: int) -> tuple[int, ...]:
        """This dataset's row count in each block of the merged space."""
        if world_size > 1:
            if self.shard_sizes is None or len(self.shard_sizes) != world_size:
                msg = (
                    f"Dataset '{self.name}' has shard sizes {self.shard_sizes} in a model sharded over "
                    f"{world_size} ranks; shard replicated sources before fusing them."
                )
                raise ValueError(msg)
            return tuple(self.shard_sizes)

        if self.batch_sizes is not None:
            if len(self.batch_sizes) != batch_size:
                msg = f"Dataset '{self.name}' has {len(self.batch_sizes)} batch sizes for batch_size={batch_size}."
                raise ValueError(msg)
            return tuple(self.batch_sizes)

        if self.local_rows % batch_size:
            msg = (
                f"Dataset '{self.name}' has {self.local_rows} rows, which do not split evenly over "
                f"batch_size={batch_size}."
            )
            raise ValueError(msg)
        return (self.local_rows // batch_size,) * batch_size


def _starts(sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Start offset of each block, for blocks of these sizes laid out back to back."""
    return tuple(accumulate(sizes, initial=0))[:-1]


@dataclass(frozen=True)
class FusedSourceIndex:
    """Where each dataset's rows land in the merged source index space.

    Attributes
    ----------
    names : tuple[str, ...]
        Dataset names in merge order.
    block_sizes : dict[str, tuple[int, ...]]
        Per dataset, its row count in each block (see the module docstring).
    rank : int or None
        This rank when blocks are ranks, ``None`` when blocks are batch elements (all local).
    """

    names: tuple[str, ...]
    block_sizes: dict[str, tuple[int, ...]]
    rank: int | None

    @classmethod
    def build(
        cls, layouts: list[SourceLayout], *, batch_size: int, rank: int = 0, world_size: int = 1
    ) -> FusedSourceIndex:
        """Lay out the merged space over ``layouts``, which are in merge order."""
        if not layouts:
            raise ValueError("Fusing needs at least one source dataset.")
        if world_size > 1 and batch_size != 1:
            msg = f"Joint fusion of sharded sources needs batch_size == 1, got {batch_size}."
            raise ValueError(msg)

        index = cls(
            names=tuple(layout.name for layout in layouts),
            block_sizes={
                layout.name: layout.blocks(batch_size=batch_size, world_size=world_size) for layout in layouts
            },
            rank=rank if world_size > 1 else None,
        )
        for layout in layouts:
            if sum(index._local_blocks(layout.name)) != layout.local_rows:
                msg = (
                    f"Dataset '{layout.name}' holds {layout.local_rows} rows on this rank, but its "
                    f"shard/batch sizes describe {sum(index._local_blocks(layout.name))}."
                )
                raise ValueError(msg)
        return index

    @property
    def merged_block_sizes(self) -> tuple[int, ...]:
        """Row count of each merged block: one table row summed over datasets."""
        return tuple(map(sum, zip(*(self.block_sizes[name] for name in self.names))))

    @property
    def merged_shard_sizes(self) -> ShardSizes:
        """Merged per-rank row counts, or ``None`` when blocks are batch elements."""
        return None if self.rank is None else list(self.merged_block_sizes)

    @property
    def merged_batch_sizes(self) -> tuple[int, ...]:
        """Merged row count per block held on this rank."""
        merged = self.merged_block_sizes
        return merged if self.rank is None else (merged[self.rank],)

    def _local_blocks(self, name: str) -> tuple[int, ...]:
        sizes = self.block_sizes[name]
        return sizes if self.rank is None else (sizes[self.rank],)

    def fuse_rows(self, rows: dict[str, Tensor]) -> Tensor:
        """Interleave per-dataset local rows into this rank's merged rows.

        Works for any per-source-node tensor laid out like the features (e.g. the source side of a
        mapper conditioning). Built from ``split`` and ``cat``, so gradients reach every dataset.
        """
        if len(self.names) == 1:
            return rows[self.names[0]]
        split = {name: rows[name].split(self._local_blocks(name)) for name in self.names}
        n_local_blocks = len(next(iter(split.values())))
        return torch.cat([split[name][block] for block in range(n_local_blocks) for name in self.names], dim=0)

    def relabel_src(self, edge_index: Adj, name: str) -> Adj:
        """Map one dataset's global source ids (row 0) into the merged space.

        An id moves by however far its block moved. ``right=True`` bucketizing against the block
        ends steps over empty blocks, whose end repeats the previous one.
        """
        own_sizes = self.block_sizes[name]
        datasets_ahead = self.names[: self.names.index(name)]
        merged_starts = [
            start + sum(self.block_sizes[other][block] for other in datasets_ahead)
            for block, start in enumerate(_starts(self.merged_block_sizes))
        ]

        long = {"device": edge_index.device, "dtype": torch.long}
        ends = torch.tensor(tuple(accumulate(own_sizes)), **long)
        shift = torch.tensor([m - o for m, o in zip(merged_starts, _starts(own_sizes))], **long)

        src = edge_index[0]
        return torch.stack([src + shift[torch.bucketize(src, ends, right=True)], edge_index[1]])


def fuse_encoder_edges(
    index: FusedSourceIndex,
    edges: dict[str, tuple[Tensor, Adj, ShardSizes]],
    *,
    num_dst: int,
) -> tuple[Tensor, Adj, ShardSizes]:
    """Merge per-dataset ``(edge_attr, edge_index, edge_shard_sizes)`` into one dst-sorted edge set.

    Each rank's edges of every dataset already target that rank's hidden shard, which all datasets
    share, so concatenating them keeps the edge sharding valid and the per-rank edge counts add up.
    The merged edges are re-sorted by destination locally - a rank-local permutation that changes
    no shard size - since the mappers' graph partitioning expects dst-sorted edges.
    """
    shard_sizes = [edges[name][2] for name in index.names]
    if any(sizes is None for sizes in shard_sizes) != all(sizes is None for sizes in shard_sizes):
        raise ValueError("Joint fusion needs the encoder edges of all source datasets to be sharded alike.")
    merged_shard_sizes = None if shard_sizes[0] is None else [sum(per_rank) for per_rank in zip(*shard_sizes)]

    edge_attr = torch.cat([edges[name][0] for name in index.names], dim=0)
    edge_index = torch.cat([index.relabel_src(edges[name][1], name) for name in index.names], dim=1)
    edge_index, perm = sort_edge_index_by_dst(edge_index, max_value=num_dst)
    return edge_attr[perm], edge_index, merged_shard_sizes
