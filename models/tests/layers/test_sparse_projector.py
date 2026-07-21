# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.layers.sparse_projector import SparseProjector


def _make_sparse_csr(n_rows: int, n_cols: int, nnz: int, seed: int = 0) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(seed)
    flat_indices = torch.randperm(n_rows * n_cols, generator=rng)[:nnz]
    rows = flat_indices // n_cols
    cols = flat_indices % n_cols
    values = torch.rand(nnz, generator=rng)
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n_rows, n_cols)).coalesce().to_sparse_csr()


@pytest.fixture
def projector() -> SparseProjector:
    return SparseProjector(autocast=False)


@pytest.fixture
def csr_matrix() -> torch.Tensor:
    return _make_sparse_csr(n_rows=8, n_cols=16, nnz=32, seed=0)


def test_forward_matches_dense_matmul(projector: SparseProjector, csr_matrix: torch.Tensor) -> None:
    """forward result matches the equivalent dense matrix multiplication."""
    x = torch.rand(2, csr_matrix.shape[1], 4)
    out = projector(x, csr_matrix)
    dense = csr_matrix.to_dense()
    expected = torch.einsum("mn,bnc->bmc", dense, x)
    assert torch.allclose(out, expected, atol=1e-5)


def test_forward_handles_arbitrary_leading_dims(projector: SparseProjector, csr_matrix: torch.Tensor) -> None:
    """forward correctly handles inputs with extra leading dimensions."""
    # Shape: [batch, timesteps, nodes, vars]
    x = torch.rand(2, 3, csr_matrix.shape[1], 5)
    out = projector(x, csr_matrix)
    assert out.shape == (2, 3, csr_matrix.shape[0], 5)
    # Verify values match per-sample forward
    for b in range(2):
        for t in range(3):
            expected = projector(x[b, t : t + 1], csr_matrix)
            assert torch.allclose(out[b, t], expected[0], atol=1e-6)
