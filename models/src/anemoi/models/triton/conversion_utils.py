# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    msg = "Utilities for fast graph conversions were called but Triton is not installed. "
    msg += "To make use of these, please install Triton. Otherwise, use utilities relying on PyTorch and PyTorch Geometric."
    raise ValueError(msg)

import logging
LOGGER = logging.getLogger(__name__)


# PyTorch Geometric mostly operates on graph structures in COO format represented by an 'edge_index' tensor
# of shape [2, num_edges], where the first row contains source node indices and the second row contains
# destination node indices. For fused message-passing kernels, it can be beneficial to convert this COO format
# to a compressed sparse row/column (CSR/CSC) format, which allows for more efficient access patterns.
# The notion of COO/CSR/CSC mainly relates back to the representation of message passing as sparse
# matrix mulitplication of node embeddings X and an adjancency matrix A. Depending on conventions, message-passing is
# either defined as A^T @ X or A @ X which changes what CSC and CSC actually refers to. Following PyTorch Geometric,
# one can simply interpret these as
#   - CSC: destination-centric
#   - CSR: source-centric
#
# To allow efficient conversions between these formats and avoiding permuations of edge features, auxiliary structures 
# can be helpful. Relevant buffers are
#   - edge_index: [2, num_edges] tensor of COO format (src, dst)
#   - csc_offets: [num_dst + 1] tensor of column pointers for CSC format
#   - csc_indices: [num_edges] tensor of source node indices in CSC order
#   - csr_offsets: [num_src + 1] tensor of row pointers for CSR format
#   - csr_indices: [num_edges] tensor of destination node indices in CSR order
#   - map_csr_to_coo: [num_edges] tensor mapping CSR order → COO order
#   - map_csc_to_coo: [num_edges] tensor mapping CSC order → COO order
#   - map_coo_to_csr: [num_edges] tensor mapping COO

@triton.jit
def node_count_kernel(node_counts, indices, num_indices: int, TILE_E: tl.constexpr):
    eidx = tl.program_id(0).to(tl.int64) * TILE_E + tl.arange(0, TILE_E)
    mask = eidx < num_indices
    idx = tl.load(indices + eidx, mask=mask)
    tl.atomic_add(node_counts + idx, 1, mask=mask, sem="relaxed")


@triton.jit
def compress_coo_kernel(
    csr_indices, map_csr_to_coo, 
    csc_indices, map_csc_to_coo,
    edge_index, csr_offsets, csc_offsets, 
    csr_aux_node_counts, csc_aux_node_counts,
    num_indices, SORTED_BY_DST: tl.constexpr, TILE_E: tl.constexpr,
): 
    eidx = tl.program_id(0).to(tl.int64) * TILE_E + tl.arange(0, TILE_E)
    mask = eidx < num_indices

    src_idx = tl.load(edge_index + eidx, mask=mask)
    dst_idx = tl.load(edge_index + eidx + num_indices, mask=mask)

    csr_off = tl.load(csr_offsets + src_idx, mask=mask)
    old_csr_count = tl.atomic_add(csr_aux_node_counts + src_idx, 1, mask=mask)
    tl.store(csr_indices + csr_off + old_csr_count, dst_idx, mask=mask)
    tl.store(map_csr_to_coo + csr_off + old_csr_count, eidx, mask=mask)

    if not SORTED_BY_DST:
        old_csc_count = tl.atomic_add(csc_aux_node_counts + dst_idx, 1, mask=mask)
        csc_off = tl.load(csc_offsets + dst_idx, mask=mask)
        tl.store(csc_indices + csc_off + old_csc_count, src_idx, mask=mask)
        tl.store(map_csc_to_coo + csc_off + old_csc_count, eidx, mask=mask)


def _cosort_segments(offsets, indices, values):
    # Native PyTorch fallback path without a segmented sort implementation available.
    # Define sort_key := segment_id * (max_index + 1) + index for a global sort
    # that respects segment boundaries. Assumption is to not overflow in int64,
    # which should be fine for the graph sizes considered in these contexts here.
    counts = offsets[1:] - offsets[:-1]
    seg_ids = torch.repeat_interleave(
        torch.arange(len(counts), device=indices.device, dtype=torch.int64), counts
    )
    sort_idx = seg_ids.to(torch.int64) * (int(indices.max()) + 1) + indices.to(torch.int64)
    perm = torch.argsort(sort_idx)
    return indices[perm].contiguous(), values[perm].contiguous()


@torch.library.custom_op("anemoi::edge_index_to_csc_impl", mutates_args=())
def fast_edge_index_to_csc_impl(
    edge_index: torch.Tensor, 
    num_src_nodes: int, 
    num_dst_nodes: int,
    is_sorted_by_dst: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_index = edge_index.contiguous()
    num_indices = edge_index.size(1)
    TILE_E = 128
    num_blocks = triton.cdiv(num_indices, TILE_E)

    csr_offsets = torch.zeros(num_src_nodes + 1, dtype=torch.int64, device=edge_index.device)
    csc_offsets = torch.zeros(num_dst_nodes + 1, dtype=torch.int64, device=edge_index.device)

    # create offset buffers by simply counting occurence of each source or destination ID
    # and calling cumsum on these counts afterwards
    grid = (num_blocks, 1, 1)
    node_count_kernel[grid](csr_offsets[1:], edge_index[0], num_indices, TILE_E)
    node_count_kernel[grid](csc_offsets[1:], edge_index[1], num_indices, TILE_E)
    csr_offsets = csr_offsets.cumsum(dim=0)
    csc_offsets = csc_offsets.cumsum(dim=0)

    if is_sorted_by_dst:
        # if already sorted_by_dst, csc_indices already is edge_index[0, :] and map_csc_to_coo is trivial
        csc_indices = edge_index[0]
        map_csc_to_coo = torch.arange(num_indices, dtype=torch.int64, device=edge_index.device)
    else:
        csc_indices = torch.empty(num_indices, dtype=torch.int64, device=edge_index.device)
        map_csc_to_coo = torch.empty(num_indices, dtype=torch.int64, device=edge_index.device)

    csr_indices = torch.empty(num_indices, dtype=torch.int64, device=edge_index.device)
    map_csr_to_coo = torch.empty(num_indices, dtype=torch.int64, device=edge_index.device)

    # auxiliary count buffers to keep track of how many edges we already worked on while filling
    # the indices buffers based on atomic updates of IDs
    aux_src_node_counts = torch.zeros(num_src_nodes, dtype=torch.int64, device=edge_index.device)
    aux_dst_node_counts = torch.zeros(num_dst_nodes, dtype=torch.int64, device=edge_index.device)

    # kernel creating csc_indices, csr_indices and their ID maps back to original COO indices at the same time
    compress_coo_kernel[grid](
        csr_indices, map_csr_to_coo,
        csc_indices, map_csc_to_coo,
        edge_index, csr_offsets, csc_offsets,
        aux_src_node_counts, aux_dst_node_counts,
        num_indices, is_sorted_by_dst, TILE_E,
    )

    if torch.are_deterministic_algorithms_enabled():
        # CSC/CSR-structures allow for deterministic computations afterwards
        # however, the fast path will not produce csr/csc_indices in a deterministic fashion.
        # For deterministic behavior, sort indices within each segment and apply the same
        # permutation to the map arrays to ensure correctness.
        csr_indices, map_csr_to_coo = _cosort_segments(csr_offsets, csr_indices, map_csr_to_coo)
        csc_indices, map_csc_to_coo = _cosort_segments(csc_offsets, csc_indices, map_csc_to_coo)

    return csr_offsets, csr_indices, map_csr_to_coo, csc_offsets, csc_indices, map_csc_to_coo


@fast_edge_index_to_csc_impl.register_fake
def _(
    edge_index: torch.Tensor, 
    num_src_nodes: int, 
    num_dst_nodes: int,
    is_sorted_by_dst: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_edges = edge_index.shape[1]
    return (
        torch.empty(num_src_nodes + 1, dtype=torch.int64, device=edge_index.device),
        torch.empty(num_edges, dtype=torch.int64, device=edge_index.device),
        torch.empty(num_edges, dtype=torch.int64, device=edge_index.device),
        torch.empty(num_dst_nodes + 1, dtype=torch.int64, device=edge_index.device),
        torch.empty(num_edges, dtype=torch.int64, device=edge_index.device),
        torch.empty(num_edges, dtype=torch.int64, device=edge_index.device),
    )


def fast_edge_index_to_csc(
    edge_index: torch.Tensor, 
    num_src_nodes: int, 
    num_dst_nodes: int,
    is_sorted_by_dst: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert COO edge_index to CSC format using a custom Triton kernel incl. structures for the 'reverse' paths.

    Args:
        edge_index (LongTensor): [2, num_edges] edge indices (src, dst).
        num_src_nodes (int): Number of source nodes.
        num_dst_nodes (int): Number of destination nodes.
        is_sorted_by_dst (bool): whether edge_index is already sorted by destination IDs

    Returns:
        csr_offsets: [num_src_nodes + 1] tensor of row pointers for CSR format
        csr_indices: [num_edges] tensor of destination node indices in CSR order
        map_csr_to_coo: [num_edges] tensor mapping CSR order → COO order
        csc_offsets: [num_dst_nodes + 1] tensor of column pointers for CSC format
        csc_indices: [num_edges] tensor of source node indices in CSC order
        map_csc_to_coo: [num_edges] tensor mapping CSC order → COO order
    """

    return fast_edge_index_to_csc_impl(edge_index, num_src_nodes, num_dst_nodes, is_sorted_by_dst)
