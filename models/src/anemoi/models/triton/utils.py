# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from typing import Optional
from typing import Tuple

import torch
from torch_geometric.typing import Adj
from torch_geometric.utils import index_sort
from torch_geometric.utils.sparse import index2ptr

from anemoi.models.distributed.khop_edges import is_edge_index_dst_sorted
from anemoi.models.distributed.khop_edges import sort_edge_index_by_dst

ANEMOI_DEBUG_SHARDING = os.environ.get("ANEMOI_DEBUG_SHARDING", "") != ""


def edge_index_to_csc(
    edge_index: Adj,
    num_nodes: Optional[Tuple[int, int]] = None,
    reverse: bool = True,
    edges_are_dst_sorted: bool = False,
):
    """Convert edge indices to CSC format, optionally also building reverse (CSR-like) metadata.

    Args:
        edge_index (LongTensor): [2, num_edges] edge indices (src, dst).
        num_nodes (Tuple[int, int], optional): (num_src, num_dst).
        reverse (bool): If True, also build CSR-like info for per-source iteration.
        edges_are_dst_sorted (bool): If True, assume the edge indices are already sorted by dst nodes.

    Returns:
        (row, colptr), perm[, (rowptr, edge_id_per_src, edge_dst_csr, csr_pos)]:
            row: source node for each edge (CSC order)
            colptr: column pointers for CSC (dst)
            perm: original → CSC edge permutation
            rowptr: CSR-style prefix sum over src
            edge_id_per_src: indices mapping CSR order → CSC order
            edge_dst_csr: destination node per edge, in CSR order
            csr_pos: CSC edge id → its slot in CSR order (inverse of edge_id_per_src)
    """
    perm = None
    if not edges_are_dst_sorted:
        edge_index, perm = sort_edge_index_by_dst(edge_index)
    elif ANEMOI_DEBUG_SHARDING:
        assert is_edge_index_dst_sorted(edge_index), (
            "edge_index_to_csc called with edges_are_dst_sorted=True but edge_index is not sorted by destination node. "
            "This indicates a bug in edge sorting/sharding."
        )

    row, col = edge_index
    if num_nodes is None:
        num_nodes = (row.max() + 1, col.max() + 1)

    colptr = index2ptr(col, num_nodes[1])

    if reverse:  # TODO: think about non-bipartite case
        row_sorted, _ = index_sort(row, max_value=num_nodes[0])
        rowptr = index2ptr(row_sorted, num_nodes[0])
        edge_id_per_src = torch.argsort(row, stable=True)
        edge_dst = col
        # csr_pos is the inverse permutation of edge_id_per_src. The backward dst pass
        # scatters its per-edge (alpha, dS) through it into CSR slots, so that the src
        # pass can then read them back as a contiguous stream.
        csr_pos = torch.empty_like(edge_id_per_src)
        csr_pos[edge_id_per_src] = torch.arange(edge_id_per_src.numel(), device=edge_id_per_src.device)
        # dst node per CSR slot, so the src pass needs no edge_id_per_src indirection
        edge_dst_csr = edge_dst[edge_id_per_src]
        return (row, colptr), perm, (rowptr, edge_id_per_src, edge_dst_csr, csr_pos)

    return (row, colptr), perm


def is_triton_available():
    """Checks if triton is available.

    Triton is supported if the triton library is installed and if Anemoi is running on GPU.
    """
    try:
        import triton  # noqa: F401
    except ImportError:
        triton_available = False
    else:
        triton_available = True

    gpus_present = torch.cuda.is_available()

    return triton_available and gpus_present
