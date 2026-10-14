# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CSC / reverse-CSR metadata for the Triton GraphTransformer (CPU, runs without a GPU).

The kernels trust these indices blindly (no bounds checks), so wrong metadata gives wrong results or NaNs rather
than an error.
"""

import pytest
import torch

from anemoi.models.triton.utils import edge_index_to_csc


def _graph(n_src, n_dst, max_deg, seed, dst_sorted):
    gen = torch.Generator().manual_seed(seed)
    edges = []
    for dst in range(n_dst - 2):  # the last two dst nodes are isolated (trailing empty CSC columns)
        deg = int(torch.randint(0, max_deg + 1, (1,), generator=gen, device="cpu"))
        edges += [
            (int(s), dst) for s in torch.randperm(n_src - 2, generator=gen, device="cpu")[:deg]
        ]  # last two src isolated
    edge_index = torch.tensor(edges, dtype=torch.long, device="cpu").t().contiguous()
    if not dst_sorted:
        edge_index = edge_index[:, torch.randperm(edge_index.shape[1], generator=gen, device="cpu")]
    return edge_index


def _check_metadata(edge_index, n_src, n_dst, csc, perm, reverse):
    (row, colptr), (rowptr, edge_ids, edge_dst) = csc, reverse
    M = edge_index.shape[1]
    perm = torch.arange(M, device="cpu") if perm is None else perm

    # CSC order is a permutation of the input edges, grouped by dst
    assert sorted(perm.tolist()) == list(range(M))
    assert torch.equal(edge_index[:, perm], torch.stack((row, edge_dst)))
    assert torch.all(edge_dst[1:] >= edge_dst[:-1])

    # colptr: per-dst in-degree, including trailing isolated dst nodes
    assert colptr.numel() == n_dst + 1 and colptr[0] == 0 and colptr[-1] == M
    assert torch.equal(
        colptr[1:] - colptr[:-1],
        torch.bincount(edge_index[1], minlength=n_dst).to(colptr.device),
    )

    # reverse CSR: per-src out-degree, and every listed CSC edge really starts at that src
    assert rowptr.numel() == n_src + 1 and rowptr[0] == 0 and rowptr[-1] == M
    assert torch.equal(
        rowptr[1:] - rowptr[:-1],
        torch.bincount(edge_index[0], minlength=n_src).to(rowptr.device),
    )
    assert sorted(edge_ids.tolist()) == list(range(M))
    for s in range(n_src):
        assert torch.all(row[edge_ids[rowptr[s] : rowptr[s + 1]]] == s)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_edge_index_to_csc_unsorted(seed):
    n_src, n_dst = 23, 17
    edge_index = _graph(n_src, n_dst, 6, seed, dst_sorted=False)
    csc, perm, reverse = edge_index_to_csc(edge_index, num_nodes=(n_src, n_dst), reverse=True)
    _check_metadata(edge_index, n_src, n_dst, csc, perm, reverse)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_edge_index_to_csc_dst_sorted_fast_path(seed):
    """With edges_are_dst_sorted=True no permutation is applied (edge features are used as given)."""
    n_src, n_dst = 23, 17
    edge_index = _graph(n_src, n_dst, 6, seed, dst_sorted=True)
    csc, perm, reverse = edge_index_to_csc(
        edge_index, num_nodes=(n_src, n_dst), reverse=True, edges_are_dst_sorted=True
    )
    assert perm is None
    _check_metadata(edge_index, n_src, n_dst, csc, perm, reverse)


def test_edge_index_to_csc_without_reverse():
    edge_index = _graph(10, 8, 4, 3, dst_sorted=False)
    (row, colptr), perm = edge_index_to_csc(edge_index, num_nodes=(10, 8), reverse=False)
    assert torch.equal(edge_index[0, perm], row)
    assert torch.equal(colptr[1:] - colptr[:-1], torch.bincount(edge_index[1], minlength=8))
