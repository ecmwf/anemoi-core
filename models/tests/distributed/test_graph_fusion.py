# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Single-process tests for the joint-encoder source/edge merge.

Every expected value here is derived by hand from the documented merged layout (block by block,
datasets in merge order within a block, where a block is a rank when sharded and a batch element
otherwise), not recorded from the implementation.
"""

import pytest
import torch

from anemoi.models.distributed.graph_fusion import FusedSourceIndex
from anemoi.models.distributed.graph_fusion import SourceLayout
from anemoi.models.distributed.graph_fusion import fuse_encoder_edges


def _rows(tag: int, n: int) -> torch.Tensor:
    """Rows tagged so their dataset and position survive any permutation."""
    return torch.arange(n, dtype=torch.float32).reshape(n, 1) + tag


def _src_only(src_ids: list[int], dst: int = 0) -> torch.Tensor:
    """Edges that exercise a whole source id range against one destination."""
    ids = torch.tensor(src_ids)
    return torch.stack([ids, torch.full_like(ids, dst)])


def _edges(edge_index: torch.Tensor, shard_sizes=None, attr=None):
    attr = torch.zeros(edge_index.shape[1], 1) if attr is None else attr
    return attr, edge_index, shard_sizes


# --------------------------------------------------------------- single block


def test_single_block_is_plain_concatenation_in_merge_order() -> None:
    index = FusedSourceIndex.build([SourceLayout("a", 3), SourceLayout("b", 2)], batch_size=1)

    assert index.merged_shard_sizes is None
    assert index.merged_batch_sizes == (5,)
    merged = index.fuse_rows({"a": _rows(100, 3), "b": _rows(200, 2)})
    assert merged.flatten().tolist() == [100, 101, 102, 200, 201]


def test_single_block_shifts_later_datasets_source_ids() -> None:
    index = FusedSourceIndex.build([SourceLayout("a", 3), SourceLayout("b", 2)], batch_size=1)
    edges = {"a": _edges(torch.tensor([[0, 2], [1, 0]])), "b": _edges(torch.tensor([[1, 0], [0, 1]]))}

    _, merged, shard_sizes = fuse_encoder_edges(index, edges, num_dst=3)

    assert shard_sizes is None
    # 'a' keeps its ids; 'b' is shifted by 3 (so its src 0 -> 3, src 1 -> 4).
    src, dst = merged.tolist()
    assert sorted(zip(dst, src)) == [(0, 2), (0, 4), (1, 0), (1, 3)]
    assert dst == sorted(dst), "merged edges must come back dst-sorted"


# ------------------------------------------------------ several batch elements


def _batched_index() -> FusedSourceIndex:
    """'a' gridded (4 rows -> 2+2 inferred), 'b' tabular and ragged (1+3)."""
    return FusedSourceIndex.build([SourceLayout("a", 4), SourceLayout("b", 4, batch_sizes=(1, 3))], batch_size=2)


def test_batch_blocks_interleave_datasets_per_batch_element() -> None:
    index = _batched_index()

    assert index.block_sizes == {"a": (2, 2), "b": (1, 3)}
    assert index.merged_batch_sizes == (3, 5)
    merged = index.fuse_rows({"a": _rows(100, 4), "b": _rows(200, 4)})
    assert merged.flatten().tolist() == [100, 101, 200, 102, 103, 201, 202, 203]


def test_batch_blocks_relabel_source_ids_per_block() -> None:
    index = _batched_index()
    edges = {"a": _edges(_src_only([0, 1, 2, 3], dst=0)), "b": _edges(_src_only([0, 1, 2, 3], dst=1))}

    _, merged, _ = fuse_encoder_edges(index, edges, num_dst=2)

    # 'a' ids 0,1 -> 0,1 (block 0) and 2,3 -> 3,4 (block 1); 'b' 0 -> 2 and 1,2,3 -> 5,6,7.
    src, dst = merged.tolist()
    assert sorted(s for s, d in zip(src, dst) if d == 0) == [0, 1, 3, 4]
    assert sorted(s for s, d in zip(src, dst) if d == 1) == [2, 5, 6, 7]


def test_relabelled_edges_point_at_the_same_rows() -> None:
    """The defining property: merged[relabel(id)] is the row the dataset's edge pointed at."""
    index = _batched_index()
    rows = {"a": _rows(100, 4), "b": _rows(200, 4)}
    merged_rows = index.fuse_rows(rows)
    for name in ("a", "b"):
        edge_index = _src_only([0, 1, 2, 3])
        relabelled = index.relabel_src(edge_index, name)
        assert torch.equal(merged_rows[relabelled[0]], rows[name][edge_index[0]])
        assert torch.equal(relabelled[1], edge_index[1])


def test_gridded_rows_not_divisible_by_batch_size_are_rejected() -> None:
    with pytest.raises(ValueError, match="split evenly"):
        FusedSourceIndex.build([SourceLayout("a", 5)], batch_size=2)


def test_batch_sizes_of_wrong_length_are_rejected() -> None:
    with pytest.raises(ValueError, match="batch sizes for batch_size"):
        FusedSourceIndex.build([SourceLayout("a", 4, batch_sizes=(1, 1, 2))], batch_size=2)


def test_single_rank_shard_sizes_are_ignored() -> None:
    """The observation reader sets shard sizes even on one rank; blocks stay batch elements."""
    index = FusedSourceIndex.build(
        [SourceLayout("a", 4, shard_sizes=[4], batch_sizes=(1, 3)), SourceLayout("b", 2)], batch_size=2
    )
    assert index.merged_shard_sizes is None
    assert index.merged_batch_sizes == (2, 4)


# ------------------------------------------------------------------- sharded


def _sharded_index() -> FusedSourceIndex:
    """Shards [3, 2] and [1, 4] over two ranks, as seen from rank 1."""
    return FusedSourceIndex.build(
        [SourceLayout("a", 2, shard_sizes=[3, 2]), SourceLayout("b", 4, shard_sizes=[1, 4])],
        batch_size=1,
        rank=1,
        world_size=2,
    )


def test_sharded_blocks_are_ranks_and_merge_per_rank_sizes() -> None:
    index = _sharded_index()

    assert index.block_sizes == {"a": (3, 2), "b": (1, 4)}
    assert index.merged_shard_sizes == [4, 6]
    assert index.merged_batch_sizes == (6,)
    # Only this rank's rows are held locally, so the merge is a plain concatenation.
    merged = index.fuse_rows({"a": _rows(100, 2), "b": _rows(200, 4)})
    assert merged.flatten().tolist() == [100, 101, 200, 201, 202, 203]


def test_sharded_relabelling_covers_source_ids_owned_by_other_ranks() -> None:
    index = _sharded_index()
    edges = {
        "a": _edges(_src_only([0, 1, 2, 3, 4]), shard_sizes=[2, 3]),
        "b": _edges(_src_only([0, 1, 2, 3, 4]), shard_sizes=[1, 4]),
    }

    _, merged, edge_shards = fuse_encoder_edges(index, edges, num_dst=1)

    # 'a': 0,1,2 -> unchanged (rank-0 block), 3,4 -> 4,5. 'b': 0 -> 3, 1..4 -> 6..9.
    assert sorted(merged[0].tolist()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert edge_shards == [3, 7]


def test_sharded_sources_reject_batch_size_above_one() -> None:
    with pytest.raises(ValueError, match="batch_size == 1"):
        FusedSourceIndex.build([SourceLayout("a", 2, shard_sizes=[2, 2])], batch_size=2, world_size=2)


def test_shard_sizes_must_match_the_communication_group_size() -> None:
    with pytest.raises(ValueError, match="sharded over 2 ranks"):
        FusedSourceIndex.build([SourceLayout("a", 2, shard_sizes=[2, 1, 1])], batch_size=1, world_size=2)


def test_replicated_source_in_a_sharded_model_is_rejected() -> None:
    """The model shards replicated sources before fusing; the index refuses to guess."""
    with pytest.raises(ValueError, match="shard replicated sources"):
        FusedSourceIndex.build(
            [SourceLayout("a", 2, shard_sizes=[2, 2]), SourceLayout("b", 3)], batch_size=1, world_size=2
        )


def test_inconsistent_local_sizes_are_rejected() -> None:
    # shard_sizes say this rank holds 2 rows, the tensor has 3.
    with pytest.raises(ValueError, match="holds 3 rows"):
        FusedSourceIndex.build([SourceLayout("a", 3, shard_sizes=[2, 2])], batch_size=1, world_size=2)


def test_mixed_edge_sharding_is_rejected() -> None:
    index = FusedSourceIndex.build([SourceLayout("a", 2), SourceLayout("b", 2)], batch_size=1)
    edge_index = torch.tensor([[0], [0]])
    with pytest.raises(ValueError, match="sharded alike"):
        fuse_encoder_edges(index, {"a": _edges(edge_index, [1, 0]), "b": _edges(edge_index)}, num_dst=1)


# ------------------------------------------------------------- empty datasets


def test_empty_block_is_skipped_when_relabelling() -> None:
    """A dataset absent from one batch element must not claim ids in that block."""
    index = FusedSourceIndex.build(
        [SourceLayout("a", 2, batch_sizes=(0, 2)), SourceLayout("b", 2, batch_sizes=(1, 1))], batch_size=2
    )
    edges = {"a": _edges(_src_only([0, 1], dst=0)), "b": _edges(_src_only([0, 1], dst=1))}

    _, merged, _ = fuse_encoder_edges(index, edges, num_dst=2)

    # 'a' has no nodes in block 0, so its ids 0,1 belong to block 1 -> merged 1,2, while 'b' takes
    # merged 0 (its block-0 node) and 3 (its block-1 node).
    src, dst = merged.tolist()
    assert sorted(zip(dst, src)) == [(0, 1), (0, 2), (1, 0), (1, 3)]
    merged_rows = index.fuse_rows({"a": _rows(100, 2), "b": _rows(200, 2)})
    assert merged_rows.flatten().tolist() == [200, 100, 101, 201]


def test_entirely_empty_sources_merge_to_zero_rows_and_edges() -> None:
    """A batch without any observations still encodes (zero rows, zero edges)."""
    index = FusedSourceIndex.build(
        [SourceLayout("a", 0, batch_sizes=(0, 0)), SourceLayout("b", 0, batch_sizes=(0, 0))], batch_size=2
    )
    empty = torch.zeros(2, 0, dtype=torch.long)
    attr, merged, _ = fuse_encoder_edges(
        index, {"a": _edges(empty, attr=torch.zeros(0, 3)), "b": _edges(empty, attr=torch.zeros(0, 3))}, num_dst=4
    )
    assert merged.shape == (2, 0) and attr.shape == (0, 3)
    assert index.fuse_rows({"a": torch.zeros(0, 5), "b": torch.zeros(0, 5)}).shape == (0, 5)


# ------------------------------------------------------------------ edge cases


def test_merged_edge_attributes_follow_the_dst_sort_permutation() -> None:
    # 'a' points at dst 2, 'b' at dst 0, so the merge must reorder both rows together.
    index = FusedSourceIndex.build([SourceLayout("a", 2), SourceLayout("b", 2)], batch_size=1)
    edges = {
        "a": _edges(torch.tensor([[0], [2]]), attr=torch.full((1, 1), 7.0)),
        "b": _edges(torch.tensor([[0], [0]]), attr=torch.full((1, 1), 9.0)),
    }

    merged_attr, merged_index, _ = fuse_encoder_edges(index, edges, num_dst=3)

    assert merged_index[1].tolist() == [0, 2]
    assert merged_attr.flatten().tolist() == [9.0, 7.0]


def test_fused_rows_keep_gradients_flowing_to_each_source() -> None:
    a = _rows(100, 4).requires_grad_(True)
    b = _rows(200, 4).requires_grad_(True)

    (_batched_index().fuse_rows({"a": a, "b": b}) * torch.arange(8.0).unsqueeze(1)).sum().backward()

    # Merged order is a0 a1 b0 a2 a3 b1 b2 b3, weighted 0..7.
    assert a.grad.flatten().tolist() == [0, 1, 3, 4]
    assert b.grad.flatten().tolist() == [2, 5, 6, 7]
