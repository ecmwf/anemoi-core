# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Merge several source datasets into one bipartite graph for joint encoding.

A joint encoder runs a single mapper pass over the union of its source datasets' nodes, so each
destination (hidden) node attends to every source dataset at once. That means relabelling each
dataset's source node indices into a shared index space, and concatenating the per-dataset edge
sets without breaking the sharding metadata the mappers rely on.

The merged index space is one table of **blocks**: block by block, and within a block the datasets
in the order the yaml config listed them under ``source_datasets``. Everything else - offsets, shard
sizes, per-sample sizes - is a cumulative sum over that table, which is why
:class:`FusedSourceIndex` stores only the table and derives the rest.

A block is a **rank** when the sources are sharded and a **batch element** otherwise; never both,
because ``_assert_valid_sharding`` requires ``batch_size == 1`` whenever the model is sharded over
more than one rank. The two cases are the two block flavours because:

* sync_tensor/gather_tensor concatenate rank blocks in rank order, so the outermost axis of any
  gathered source tensor is the rank;

* graph providers lay out their source index space batch-major - StaticGraphProvider._expand_edges
  concatenates ``edge_index + i * edge_inc`` over batch elements, and TabularSourceView.flatten
  concatenates per-sample tensors - so within a rank the batch element varies more slowly than the
  node.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

import torch
from torch import Tensor
from torch_geometric.typing import Adj

from anemoi.models.distributed.khop_edges import sort_edge_index_by_dst
from anemoi.models.distributed.shapes import ShardSizes

__all__ = [
    "FusableSource",
    "FusedSourceIndex",
    "build_fused_source_index",
    "fuse_encoder_edges",
    "fuse_source_features",
]


@dataclass(frozen=True)
class FusableSource:
    """One source dataset of a joint encoder, as the merge operation needs to see it.

    Attributes
    ----------
    name : str
        Dataset name. Its position in the list handed to `build_fused_source_index` is its
        position in the merged index space.
    features : Tensor
        This rank's assembled (and already projected) source rows, ``(num_local_nodes, num_channels)``.
    shard_sizes : ShardSizes
        Per-rank source node counts, or None when the sources are replicated.
    batch_sizes : tuple[int, ...] or None
        Per-sample source node counts for a tabular dataset. ``None`` for a gridded dataset, whose
        samples all hold the same number of nodes.
    edge_attr : Tensor or None
        Encoder edge features, ``(num_local_edges, edge_features)``.
    edge_index : Adj or None
        Encoder edges, in the shared hidden index space.
    edge_shard_sizes : ShardSizes
        Per-rank edge counts, or ``None`` when the edges are replicated (no sharding).
    """

    name: str
    features: Tensor
    shard_sizes: ShardSizes = None
    batch_sizes: tuple[int, ...] | None = None
    edge_attr: Tensor | None = None
    edge_index: Adj | None = None
    edge_shard_sizes: ShardSizes = None

    @property
    def local_size(self) -> int:
        """Number of source rows this rank holds."""
        return self.features.shape[0]


def _blocks_start_at(sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Calculates the start index of each block, when blocks of these sizes are laid out back to back."""
    return tuple(accumulate(sizes, initial=0))[:-1]


@dataclass(frozen=True)
class FusedSourceIndex:
    """Records where each dataset's source nodes land in the merged index space.

    Attributes
    ----------
    order : tuple[str, ...]
        Dataset names in merge order (the configured ``source_datasets`` order, restricted to the
        datasets actually participating in this batch).
    block_sizes : dict[str, tuple[int, ...]]
        The table: per dataset, its node count in each block. See the module docstring for what a
        block is; every other attribute here is a cumulative sum over it.
    local_block_ids : tuple[int, ...]
        Which blocks this rank holds rows for: ``(rank,)`` when sharded, all of them otherwise.
    is_sharded : bool
        Whether blocks are ranks (True) or batch elements (False).
    """

    order: tuple[str, ...]
    block_sizes: dict[str, tuple[int, ...]]
    local_block_ids: tuple[int, ...]
    is_sharded: bool

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the merged space."""
        return len(self.block_sizes[self.order[0]])

    @property
    def merged_block_sizes(self) -> tuple[int, ...]:
        """Node count of each merged block, i.e. one table row summed over datasets."""
        return tuple(sum(self.block_sizes[name][k] for name in self.order) for k in range(self.n_blocks))

    @property
    def dataset_offsets(self) -> dict[str, tuple[int, ...]]:
        """Per dataset, where each of its blocks begins in that dataset's own index space.

        That is the space its graph provider's ``edge_index[0]`` refers to.
        """
        return {name: _blocks_start_at(sizes) for name, sizes in self.block_sizes.items()}

    @property
    def merged_offsets(self) -> dict[str, tuple[int, ...]]:
        """Per dataset, where each of its blocks begins in the *merged* index space.

        A cell of the table starts after every earlier block, plus the datasets ahead of it inside
        its own block.
        """
        offsets: dict[str, list[int]] = {name: [] for name in self.order}
        for block, block_start in enumerate(_blocks_start_at(self.merged_block_sizes)):
            cursor = block_start
            for name in self.order:
                offsets[name].append(cursor)
                cursor += self.block_sizes[name][block]
        return {name: tuple(starts) for name, starts in offsets.items()}

    @property
    def merged_shard_sizes(self) -> ShardSizes:
        """Merged per-rank node counts, or None when the sources are replicated."""
        return list(self.merged_block_sizes) if self.is_sharded else None

    @property
    def merged_batch_sizes(self) -> tuple[int, ...]:
        """Merged node count per batch element on this rank. Length 1 when sharded."""
        return tuple(self.merged_block_sizes[k] for k in self.local_block_ids)

    @property
    def merged_local_size(self) -> int:
        """Number of merged source rows this rank holds."""
        return sum(self.merged_batch_sizes)

    def local_block_sizes(self, dataset_name: str) -> tuple[int, ...]:
        """One dataset's node counts, restricted to the blocks this rank holds."""
        sizes = self.block_sizes[dataset_name]
        return tuple(sizes[k] for k in self.local_block_ids)

    def to_merged_src_ids(self, src_ids: Tensor, dataset_name: str) -> Tensor:
        """Map one dataset's global source ids into the merged global source index space.

        Every id is shifted by however far its own block moved. Blocks are contiguous, so an id's
        block is found by bucketizing against the block ends; ``right=True`` steps over empty
        blocks (whose end repeats the previous one) instead of matching them.
        """
        own = self.dataset_offsets[dataset_name]
        merged = self.merged_offsets[dataset_name]

        as_long = {"device": src_ids.device, "dtype": torch.long}
        # Inclusive block ends in the dataset's own space, and how far each block moved.
        ends = torch.tensor(tuple(accumulate(self.block_sizes[dataset_name])), **as_long)
        shifts = torch.tensor([m - o for m, o in zip(merged, own)], **as_long)

        return src_ids + shifts[torch.bucketize(src_ids, ends, right=True)]


def build_fused_source_index(
    sources: list[FusableSource],
    *,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
) -> FusedSourceIndex:
    """Lay out a merged source index space over several source datasets.

    Parameters
    ----------
    sources
        The source datasets, in merge order.
    batch_size
        Number of batch elements.
    rank, world_size
        Position and size of the model communication group. ``world_size == 1`` means replicated.

    Returns
    -------
    FusedSourceIndex
        The merged layout, which :func:`fuse_source_features` and :func:`fuse_encoder_edges` both
        index against.
    """
    assert sources, "build_fused_source_index needs at least one dataset."

    is_sharded = _sources_are_sharded(sources, batch_size=batch_size, world_size=world_size)
    index = FusedSourceIndex(
        order=tuple(source.name for source in sources),
        block_sizes=_rank_blocks(sources, world_size) if is_sharded else _batch_element_blocks(sources, batch_size),
        local_block_ids=(rank,) if is_sharded else tuple(range(batch_size)),
        is_sharded=is_sharded,
    )

    expected_local = sum(source.local_size for source in sources)
    if index.merged_local_size != expected_local:
        msg = (
            f"Merged source size {index.merged_local_size} does not match the sum of the per-dataset "
            f"local sizes {expected_local}. Shard/batch metadata is inconsistent with the assembled tensors."
        )
        raise ValueError(msg)

    return index


def fuse_source_features(sources: list[FusableSource], index: FusedSourceIndex) -> Tensor:
    """Concatenate the per-dataset source rows into this rank's merged source tensor.

    Built from ``split`` + ``cat``, so gradients flow back to each dataset's own tensor untouched.
    With a single local block - always the case when sharded - this is exactly a concatenation of
    the per-dataset tensors in merge order.
    """
    blocks = {source.name: source.features.split(index.local_block_sizes(source.name)) for source in sources}

    pieces = []
    for local_block in range(len(index.local_block_ids)):
        for name in index.order:
            piece = blocks[name][local_block]
            if piece.shape[0]:
                pieces.append(piece)

    if not pieces:
        raise ValueError("fuse_source_features got no rows to merge.")

    merged = _cat(pieces, dim=0)
    if merged.shape[0] != index.merged_local_size:
        raise ValueError(f"Merged source features have {merged.shape[0]} rows, expected {index.merged_local_size}.")

    return merged


def fuse_encoder_edges(
    sources: list[FusableSource],
    index: FusedSourceIndex,
    *,
    num_dst: int,
) -> tuple[Tensor, Adj, ShardSizes]:
    """Merge the per-dataset encoder edges into one dst-sorted edge set.

    Parameters
    ----------
    sources
        The source datasets, in merge order, carrying the edges to merge.
    index
        The merged source layout from :func:`build_fused_source_index`.
    num_dst
        Total number of destination nodes, used as a sort hint.

    Returns
    -------
    Merged ``edge_attr``, merged ``edge_index``, and merged per-rank edge counts (None when the
    edges are not sharded).

    Notes
    -----
    Each rank's local edge block still holds exactly the edges whose destination lies in that rank's
    destination shard: that is true of every dataset individually, and the hidden destination shard
    is shared by all of them, so it survives concatenation. The merged edges are re-sorted by
    destination **locally** - a rank-local permutation, so it changes no shard size and needs no
    communication - because ``ensure_edges_are_dst_sorted`` refuses to sort an already-sharded edge
    set, and the chunking in ``build_graph_partition`` requires dst-sorted input.
    """
    assert sources, "fuse_encoder_edges needs at least one dataset."

    for source in sources:
        if source.edge_index is None or source.edge_attr is None:
            msg = (
                f"Joint fusion requires edge-based encoder graphs, but dataset '{source.name}' "
                "supplied no edges. Use dataset_fusing_strategy: 'sequential' for edge-less mappers."
            )
            raise ValueError(msg)

    edge_shard_sizes = {source.name: source.edge_shard_sizes for source in sources}
    merged_shard_sizes = (
        _sum_per_rank(edge_shard_sizes) if _all_or_none_sharded(edge_shard_sizes, what="edge") else None
    )

    edge_index = _cat([_with_merged_src_ids(source, index) for source in sources], dim=1)
    edge_attr = _cat([source.edge_attr for source in sources], dim=0)

    # Re-sort the merged edges by destination
    edge_index, perm = sort_edge_index_by_dst(edge_index, max_value=num_dst)
    return edge_attr[perm], edge_index, merged_shard_sizes


def _cat(tensors: list[Tensor], *, dim: int) -> Tensor:
    """``torch.cat``, skipping the copy when there is only one tensor."""
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=dim)


def _with_merged_src_ids(source: FusableSource, index: FusedSourceIndex) -> Adj:
    """One dataset's edges, with row 0 relabelled into the merged source index space."""
    edges = source.edge_index.clone()
    edges[0] = index.to_merged_src_ids(edges[0], source.name)
    return edges


def _all_or_none_sharded(shard_sizes: dict[str, ShardSizes], *, what: str) -> bool:
    """Whether these descriptors are sharded at all; the datasets must agree either way."""
    sharded = [name for name, sizes in shard_sizes.items() if sizes is not None]
    if sharded and len(sharded) != len(shard_sizes):
        replicated = [name for name in shard_sizes if name not in sharded]
        msg = (
            f"Joint fusion requires all of an encoder's source datasets to agree on {what} sharding, "
            f"but {sharded} are sharded and {replicated} are replicated."
        )
        raise ValueError(msg)
    return bool(sharded)


def _sum_per_rank(shard_sizes: dict[str, ShardSizes]) -> list[int]:
    """Per-rank totals across datasets, which must all describe the same communication group."""
    widths = {len(sizes) for sizes in shard_sizes.values()}
    if len(widths) != 1:
        msg = f"Per-dataset shard sizes span different communication group sizes: {widths}."
        raise ValueError(msg)
    return [sum(sizes[rank] for sizes in shard_sizes.values()) for rank in range(widths.pop())]


def _sources_are_sharded(sources: list[FusableSource], *, batch_size: int, world_size: int) -> bool:
    """Whether the merge lays its blocks out over ranks rather than batch elements."""
    sharded = _all_or_none_sharded({source.name: source.shard_sizes for source in sources}, what="node")
    if not sharded:
        return False

    names = [source.name for source in sources]
    if world_size <= 1:
        msg = (
            f"Datasets {names} carry shard sizes but the communication group spans {world_size} "
            "rank(s). Sharded sources require a model communication group; treating them as "
            "replicated would silently mis-index the merged source space."
        )
        raise ValueError(msg)
    if batch_size > 1:
        msg = (
            f"Joint fusion cannot merge sharded sources with batch_size={batch_size}; sharded "
            "training requires batch_size == 1."
        )
        raise ValueError(msg)
    return True


def _rank_blocks(sources: list[FusableSource], world_size: int) -> dict[str, tuple[int, ...]]:
    """One block per rank, taken from each dataset's per-rank shard sizes."""
    blocks = {}
    for source in sources:
        sizes = tuple(source.shard_sizes)
        if len(sizes) != world_size:
            msg = f"Dataset '{source.name}' has {len(sizes)} shard sizes for a communication group of {world_size}."
            raise ValueError(msg)
        blocks[source.name] = sizes
    return blocks


def _batch_element_blocks(sources: list[FusableSource], batch_size: int) -> dict[str, tuple[int, ...]]:
    """One block per batch element: the declared per-sample counts, or an even split if gridded."""
    blocks = {}
    for source in sources:
        if source.batch_sizes is not None:
            sizes = tuple(source.batch_sizes)
            if len(sizes) != batch_size:
                msg = f"Dataset '{source.name}' has {len(sizes)} batch sizes for batch_size={batch_size}."
                raise ValueError(msg)
        else:
            # Gridded dataset: equal node counts per batch element.
            if source.local_size % batch_size:
                msg = (
                    f"Dataset '{source.name}' has {source.local_size} source rows, which is not "
                    f"divisible by batch_size={batch_size}; cannot infer its per-sample node count."
                )
                raise ValueError(msg)
            sizes = (source.local_size // batch_size,) * batch_size
        blocks[source.name] = sizes
    return blocks
