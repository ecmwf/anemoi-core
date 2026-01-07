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

# check if triton is installed
# If pytorch is installed on CPU then torch is not available
try:
    import triton
    import triton.language as tl
except ImportError:
    raise ValueError(
        "Error. The 'triton' backend was selected for the GraphTransformer but Triton is not installed. To use this backend please install Triton. Otherwise, select a different backend for the GraphTransformer in the models config."
    )


def torch_dtype_to_triton(dtype):
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.float32:
        return tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


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


@triton.jit
def build_masks_and_offsets(H: tl.constexpr, C: tl.constexpr, H_pad: tl.constexpr, C_pad: tl.constexpr):
    """Pads H and C to the nearest power of 2 if needed.

    This is required to support non-square numbers of heads and/or channels.
    Returns a mask for H, H*C and an offset for accessing into a 2D H*C matrix, ignoring padded values

    masking apparently has a price, so if H and C are already powers of 2, nothing is returned
    If H is already a power of 2 but C is not, a simpler H*C mask is returned

    This function assumes a matrix layout of shape [H,C] for mask_H_C and H_C_off
    """

    # default mask (assume no padded values)
    H_mask = True
    H_C_mask = True

    if H == H_pad and C == C_pad:
        H_C_off = tl.arange(0, H * C)

    elif H == H_pad:  # just C is not square, we can avoid mask_H
        C_pad_off = tl.arange(0, C_pad)[None, :]  # (1, C_pad)
        H_off = tl.arange(0, H)[:, None]  # (H, 1)

        # 2D mask for H * C
        # e.g 1 2 X X
        #     5 6 X X
        #     X X X X
        # But this kernel loads in 1d, hence we reshape to 1d
        # shape (H_pad, 1) & shape (1, C_pad) => shape (H_pad, C_pad) => shape (H_pad * C_pad, )
        H_C_mask_2d = (C_pad_off < C) & (H_off < H)  # (H, C_pad)
        H_C_mask = tl.reshape(H_C_mask_2d, (H * C_pad,))
        H_C_off = tl.reshape(H_off * C + C_pad_off, (H * C_pad,))

    else:  # H and C both not square
        H_pad_off = tl.arange(0, H_pad)[:, None]
        C_pad_off = tl.arange(0, C_pad)[None, :]

        # mask for H
        H_mask = tl.arange(0, H_pad) < H

        # 2D mask for H * C
        # e.g 1 2 X X
        #     5 6 X X
        #     X X X X
        # But this kernel loads in 1d, hence we reshape to 1d
        # shape (H_pad, 1) & shape (1, C_pad) => shape (H_pad, C_pad) => shape (H_pad * C_pad, )
        H_C_mask_2d = (C_pad_off < C) & (H_pad_off < H)  # (H, C_pad)
        H_C_mask = tl.reshape(H_C_mask_2d, (H_pad * C_pad,))

        # tl.arange(H_pad, C_pad) doesnt work, because the arrays its offseting into aren't padded
        # Therefore we make our own range, using unpadded major dimension (C)
        H_C_off = tl.reshape(H_pad_off * C + C_pad_off, (H_pad * C_pad,))

    return H_mask, H_C_mask, H_C_off
