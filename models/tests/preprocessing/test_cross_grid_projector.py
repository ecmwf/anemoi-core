# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch

from anemoi.models.preprocessing import ProcessorMode
from anemoi.models.preprocessing.cross_grid_projector import CrossGridProjector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_projector(n_src: int = 8, n_dst: int = 4) -> CrossGridProjector:
    """Build a CrossGridProjector from a synthetic .npz sparse matrix.

    _build_from_file expects shape (n_src, n_dst): matrix[i, j] = weight from
    src node i to dst node j (column = destination).
    """
    import pathlib
    import tempfile

    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse import save_npz

    ratio = n_src // n_dst
    # _build_from_file reads shape as (n_dst, n_src): rows = dst, cols = src.
    # (matches the SparseProjector convention: W @ x where W is (n_dst, n_src))
    row = np.repeat(np.arange(n_dst), ratio)  # dst node indices (rows)
    col = np.arange(n_src)  # src node indices (cols)
    data = np.ones(n_src, dtype=np.float32)
    mat = csr_matrix((data, (row, col)), shape=(n_dst, n_src))

    tmp = pathlib.Path(tempfile.mktemp(suffix=".npz"))
    save_npz(str(tmp), mat)
    projector = CrossGridProjector(file_path=tmp, row_normalize=False)
    tmp.unlink()
    return projector


# ---------------------------------------------------------------------------
# ProcessorMode enum
# ---------------------------------------------------------------------------


class TestProcessorMode:
    def test_values(self):
        assert ProcessorMode.STATE == "state"
        assert ProcessorMode.TENDENCY == "tendency"
        assert ProcessorMode.RESIDUAL == "residual"

    def test_is_str(self):
        assert isinstance(ProcessorMode.STATE, str)

    def test_all_members(self):
        members = {m.value for m in ProcessorMode}
        assert members == {"state", "tendency", "residual"}


# ---------------------------------------------------------------------------
# CrossGridProjector
# ---------------------------------------------------------------------------


class TestCrossGridProjector:
    N_SRC = 8
    N_DST = 4
    BATCH, TIME, ENS, VARS = 2, 3, 1, 5

    @pytest.fixture()
    def projector(self):
        return _make_projector(self.N_SRC, self.N_DST)

    def test_output_shape(self, projector):
        x = torch.randn(self.BATCH, self.TIME, self.ENS, self.N_SRC, self.VARS)
        out = projector(x)
        assert out.shape == (self.BATCH, self.TIME, self.ENS, self.N_DST, self.VARS)

    def test_no_trainable_parameters(self, projector):
        params = list(projector.parameters())
        assert len(params) == 0, "CrossGridProjector should have no trainable parameters"

    def test_values_match_sparse_matmul(self, projector):
        """Each dst node should sum its src contributions (row_normalize=False)."""
        ratio = self.N_SRC // self.N_DST  # = 2: each dst sums 2 src nodes with weight 1
        x = torch.ones(1, 1, 1, self.N_SRC, 1)
        out = projector(x)
        expected = torch.full((1, 1, 1, self.N_DST, 1), float(ratio))
        assert torch.allclose(out, expected)

    def test_gradient_does_not_flow_through_matrix(self, projector):
        x = torch.randn(1, 1, 1, self.N_SRC, self.VARS, requires_grad=True)
        out = projector(x)
        out.sum().backward()
        # Gradient flows through x but projector has no parameters to update
        assert x.grad is not None
        assert len(list(projector.parameters())) == 0

    def test_inverse_raises(self, projector):
        x = torch.randn(1, 1, 1, self.N_DST, self.VARS)
        with pytest.raises(NotImplementedError):
            projector.inverse(x)

    def test_checkpoint_roundtrip(self, projector, tmp_path):
        """Saved and loaded projector produces identical output."""
        path = tmp_path / "projector.pt"
        torch.save(projector.state_dict(), path)

        projector2 = _make_projector(self.N_SRC, self.N_DST)
        projector2.load_state_dict(torch.load(path))

        x = torch.randn(self.BATCH, self.TIME, self.ENS, self.N_SRC, self.VARS)
        assert torch.allclose(projector(x), projector2(x))
