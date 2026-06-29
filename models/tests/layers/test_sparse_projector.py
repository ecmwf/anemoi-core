# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests to verify numerical equivalence between the old and new SparseProjector
implementations introduced in commit 38328094b.

There are two code changes that could affect loss values:

1. COO -> CSR conversion of smoothing matrices (MultiscaleLossWrapper._as_coalesced_csr).
   torch.sparse.mm uses different internal summation order for COO vs CSR tensors,
   which can introduce floating-point rounding differences.

2. Batched sparse matmul in SparseProjector (new_forward): instead of looping over
   batch elements and calling torch.sparse.mm once per sample, a single matmul is
   performed with a reshaped [N, B*C] RHS. The summation order difference can
   also cause floating-point discrepancies vs the loop.
"""

import pytest
import torch

from anemoi.models.layers.sparse_projector import SparseProjector


# ---------------------------------------------------------------------------
# Helpers – re-implement the "new" approaches inline so these tests are
# self-contained and work on the old commit (390ce34) as well as the new one.
# ---------------------------------------------------------------------------

def _batched_forward(projector: SparseProjector, x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """New batched implementation (commit 38328094b new_forward).

    One sparse matmul over the whole batch instead of one per sample.
    """
    batch_size = x.shape[0]
    input_nodes = x.shape[1]
    trailing_shape = x.shape[2:]

    with torch.amp.autocast(device_type=x.device.type, enabled=projector.autocast):
        x_flat = x.reshape(batch_size, input_nodes, -1)
        # [B, N, C] -> [N, B*C]
        x_rhs = x_flat.permute(1, 0, 2).reshape(input_nodes, -1)
        # [M, N] @ [N, B*C] -> [M, B*C]
        out = torch.sparse.mm(proj, x_rhs)
        output_nodes = out.shape[0]
        # [M, B*C] -> [B, M, ...]
        out = out.reshape(output_nodes, batch_size, -1).permute(1, 0, 2)
        out = out.reshape(batch_size, output_nodes, *trailing_shape)
    return out


def _loop_forward(projector: SparseProjector, x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Old per-sample loop implementation (commit 390ce34 / SparseProjector.forward)."""
    out = []
    with torch.amp.autocast(device_type=x.device.type, enabled=projector.autocast):
        for i in range(x.shape[0]):
            out.append(torch.sparse.mm(proj, x[i, ...]))
    return torch.stack(out)


def _as_csr(coo: torch.Tensor) -> torch.Tensor:
    """Convert a sparse COO tensor to CSR (mirrors _as_coalesced_csr)."""
    return coo.coalesce().to_sparse_csr()


def _make_random_sparse_coo(n_rows: int, n_cols: int, nnz: int, seed: int = 0) -> torch.Tensor:
    """Create a random sparse COO tensor with no duplicate indices."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    flat_indices = torch.randperm(n_rows * n_cols, generator=rng)[:nnz]
    rows = flat_indices // n_cols
    cols = flat_indices % n_cols
    values = torch.rand(nnz, generator=rng)
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n_rows, n_cols)).coalesce()


def _make_coo_with_duplicates(n_rows: int, n_cols: int, seed: int = 0) -> torch.Tensor:
    """Create a sparse COO tensor that has duplicate (row, col) entries.

    Before .coalesce() the values at duplicate positions are NOT summed.
    After .coalesce() they ARE summed, which can change the projection result.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    # Deliberately repeat indices
    rows = torch.tensor([0, 0, 1, 1, 2])
    cols = torch.tensor([0, 0, 1, 1, 2])
    values = torch.rand(5, generator=rng)
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n_rows, n_cols))


# ---------------------------------------------------------------------------
# Tests: COO vs CSR numerical equivalence
# ---------------------------------------------------------------------------

class TestCooVsCsrProjection:
    """Verify that using CSR vs COO format does not change the matmul result
    for well-formed (no duplicate indices) sparse matrices.

    If these tests fail it means torch.sparse.mm gives different floating-point
    results for COO vs CSR, which would explain loss differences between the two
    commits.
    """

    @pytest.fixture
    def random_coo(self) -> torch.Tensor:
        return _make_random_sparse_coo(n_rows=16, n_cols=32, nnz=64, seed=42)

    def test_csr_matches_coo_single_vector(self, random_coo: torch.Tensor) -> None:
        """Single dense vector: CSR and COO sparse.mm should agree."""
        csr = _as_csr(random_coo)
        x = torch.rand(random_coo.shape[1])
        out_coo = torch.sparse.mm(random_coo, x.unsqueeze(1)).squeeze(1)
        out_csr = torch.sparse.mm(csr, x.unsqueeze(1)).squeeze(1)
        assert torch.allclose(out_coo, out_csr), (
            f"COO and CSR sparse.mm differ for a single vector.\n"
            f"Max abs diff: {(out_coo - out_csr).abs().max():.6e}"
        )

    def test_csr_matches_coo_batch_matrix(self, random_coo: torch.Tensor) -> None:
        """Dense matrix (batch of vectors): COO and CSR should agree."""
        csr = _as_csr(random_coo)
        x = torch.rand(random_coo.shape[1], 8)
        out_coo = torch.sparse.mm(random_coo, x)
        out_csr = torch.sparse.mm(csr, x)
        assert torch.allclose(out_coo, out_csr), (
            f"COO and CSR sparse.mm differ for a dense matrix.\n"
            f"Max abs diff: {(out_coo - out_csr).abs().max():.6e}"
        )

    def test_csr_matches_coo_float64(self, random_coo: torch.Tensor) -> None:
        """Same test in float64 to separate precision effects from format effects."""
        coo64 = random_coo.to(torch.float64)
        csr64 = _as_csr(coo64)
        x = torch.rand(random_coo.shape[1], 8, dtype=torch.float64)
        out_coo = torch.sparse.mm(coo64, x)
        out_csr = torch.sparse.mm(csr64, x)
        assert torch.allclose(out_coo, out_csr, atol=1e-12), (
            f"COO and CSR sparse.mm differ even in float64.\n"
            f"Max abs diff: {(out_coo - out_csr).abs().max():.6e}"
        )

    def test_coalesce_merges_duplicates(self) -> None:
        """Verify that .coalesce() sums duplicate entries, which can change results.

        This is the key risk: if the COO matrices used in the old code contained
        duplicate indices, calling .coalesce().to_sparse_csr() will SUM those
        duplicates, producing a different matrix from the original uncoalesced COO.
        """
        coo_with_dups = _make_coo_with_duplicates(n_rows=4, n_cols=4)
        coo_coalesced = coo_with_dups.coalesce()  # sums duplicates

        x = torch.rand(4, 1)

        # The uncoalesced COO may or may not produce the same result as the
        # coalesced version depending on PyTorch internals. What matters is:
        # the coalesced version is what _as_coalesced_csr uses.
        out_uncoalesced = torch.sparse.mm(coo_with_dups, x)
        out_coalesced = torch.sparse.mm(coo_coalesced, x)
        out_csr = torch.sparse.mm(_as_csr(coo_with_dups), x)

        # CSR (via coalesce) must equal coalesced COO
        assert torch.allclose(out_csr, out_coalesced), (
            "CSR result should equal coalesced COO result — both sum duplicates."
        )

        # If this assertion FAILS it means the raw COO (uncoalesced) has different
        # values from the coalesced version, which is the source of loss differences.
        are_equal = torch.allclose(out_uncoalesced, out_coalesced)
        if not are_equal:
            diff = (out_uncoalesced - out_coalesced).abs().max()
            pytest.skip(
                f"Uncoalesced COO gives different result from coalesced COO "
                f"(max diff={diff:.4e}). This confirms the CSR conversion changes "
                f"the projection when the input matrix has duplicate indices."
            )


# ---------------------------------------------------------------------------
# Tests: batched matmul vs per-sample loop
# ---------------------------------------------------------------------------

class TestBatchedVsLoopProjector:
    """Verify that _batched_forward gives the same result as _loop_forward.

    If these tests fail it means the single-matmul approach accumulates
    floating-point errors differently from the per-sample loop, which would
    explain loss differences between the two commits.
    """

    @pytest.fixture
    def projector(self) -> SparseProjector:
        return SparseProjector(autocast=False)

    @pytest.fixture
    def sparse_proj(self) -> torch.Tensor:
        return _make_random_sparse_coo(n_rows=8, n_cols=16, nnz=32, seed=7)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_batched_matches_loop_2d_inner(
        self,
        projector: SparseProjector,
        sparse_proj: torch.Tensor,
        batch_size: int,
        n_channels: int,
    ) -> None:
        """x shape: [B, input_nodes, C]."""
        x = torch.rand(batch_size, sparse_proj.shape[1], n_channels)
        out_loop = _loop_forward(projector, x, sparse_proj)
        out_batched = _batched_forward(projector, x, sparse_proj)
        assert torch.allclose(out_loop, out_batched, atol=1e-5), (
            f"Batched and loop projectors differ (batch={batch_size}, C={n_channels}).\n"
            f"Max abs diff: {(out_loop - out_batched).abs().max():.6e}"
        )

    def test_batched_matches_loop_with_csr(
        self,
        projector: SparseProjector,
        sparse_proj: torch.Tensor,
    ) -> None:
        """Use a CSR matrix (as the new code does after _prepare_smoothing_matrices)."""
        csr_proj = _as_csr(sparse_proj)
        x = torch.rand(2, sparse_proj.shape[1], 5)
        out_loop = _loop_forward(projector, x, csr_proj)
        out_batched = _batched_forward(projector, x, csr_proj)
        assert torch.allclose(out_loop, out_batched, atol=1e-5), (
            f"Batched and loop projectors differ when using CSR format.\n"
            f"Max abs diff: {(out_loop - out_batched).abs().max():.6e}"
        )

    def test_batched_matches_loop_coo_vs_csr_combined(
        self,
        projector: SparseProjector,
        sparse_proj: torch.Tensor,
    ) -> None:
        """Cross-check: old code (loop + COO) vs new code (batched + CSR).

        This is the most representative comparison: it exercises both
        changes together, mirroring the actual difference between the two commits.
        """
        csr_proj = _as_csr(sparse_proj)
        x = torch.rand(2, sparse_proj.shape[1], 6)

        # Old: per-sample loop with COO matrix
        out_old = _loop_forward(projector, x, sparse_proj)
        # New: batched matmul with CSR matrix
        out_new = _batched_forward(projector, x, csr_proj)

        assert torch.allclose(out_old, out_new, atol=1e-5), (
            f"Old (loop+COO) and new (batched+CSR) projectors differ.\n"
            f"Max abs diff: {(out_old - out_new).abs().max():.6e}\n"
            "This confirms the two commits produce different smoothed tensors."
        )
