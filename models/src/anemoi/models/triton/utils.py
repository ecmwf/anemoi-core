# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional
from typing import Tuple

import torch
from torch_geometric.typing import Adj
from torch_geometric.utils import index_sort
from torch_geometric.utils.sparse import index2ptr


def edge_index_to_csc(edge_index: Adj, num_nodes: Optional[Tuple[int, int]] = None, reverse: bool = True):
    """Convert edge indices to CSC format, optionally also building reverse (CSR-like) metadata.

    Args:
        edge_index (LongTensor): [2, num_edges] edge indices (src, dst).
        num_nodes (Tuple[int, int], optional): (num_src, num_dst).
        reverse (bool): If True, also build CSR-like info for per-source iteration.

    Returns:
        (row, colptr), perm[, (rowptr, edge_id_per_src, edge_dst)]:
            row: source node for each edge (CSC order)
            colptr: column pointers for CSC (dst)
            perm: original → CSC edge permutation
            rowptr: CSR-style prefix sum over src
            edge_id_per_src: indices mapping CSR order → CSC order
            edge_dst: destination node per edge (CSC order)
    """
    row, col = edge_index
    if num_nodes is None:
        num_nodes = (row.max().item() + 1, col.max().item() + 1)

    col, perm = index_sort(col, max_value=num_nodes[1])
    row = row[perm]
    colptr = index2ptr(col, num_nodes[1])

    if reverse:
        row_sorted, _ = index_sort(row, max_value=num_nodes[0])
        rowptr = index2ptr(row_sorted, num_nodes[0])
        edge_id_per_src = torch.argsort(row, stable=True)
        edge_dst = col
        return (row, colptr), perm, (rowptr, edge_id_per_src, edge_dst)

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
