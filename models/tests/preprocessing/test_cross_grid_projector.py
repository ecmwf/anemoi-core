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

from anemoi.models.preprocessing.cross_grid_projector import CrossGridProjector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_projector(n_src: int = 8, n_dst: int = 4) -> CrossGridProjector:
    """Build a CrossGridProjector from a synthetic .npz sparse matrix."""
    import pathlib
    import tempfile

    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse import save_npz

    ratio = n_src // n_dst
    # Rows are destinations and columns are sources, matching W @ x.
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
        out, grid_shard_sizes = projector(x)
        assert out.shape == (self.BATCH, self.TIME, self.ENS, self.N_DST, self.VARS)
        assert grid_shard_sizes is None

    def test_grid_sizes(self, projector):
        assert projector.input_grid_size == self.N_SRC
        assert projector.output_grid_size == self.N_DST

    def test_no_trainable_parameters(self, projector):
        params = list(projector.parameters())
        assert len(params) == 0, "CrossGridProjector should have no trainable parameters"

    def test_values_match_sparse_matmul(self, projector):
        """Each dst node should sum its src contributions (row_normalize=False)."""
        ratio = self.N_SRC // self.N_DST  # = 2: each dst sums 2 src nodes with weight 1
        x = torch.ones(1, 1, 1, self.N_SRC, 1)
        out, _ = projector(x)
        expected = torch.full((1, 1, 1, self.N_DST, 1), float(ratio))
        assert torch.allclose(out, expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_preserves_input_dtype(self, projector, dtype):
        x = torch.ones(1, 1, 1, self.N_SRC, 1, dtype=dtype)
        out, _ = projector(x)
        assert out.dtype == dtype

    def test_gradient_does_not_flow_through_matrix(self, projector):
        x = torch.randn(1, 1, 1, self.N_SRC, self.VARS, requires_grad=True)
        out, _ = projector(x)
        out.sum().backward()
        # Gradient flows through x but projector has no parameters to update
        assert x.grad is not None
        assert len(list(projector.parameters())) == 0

    def test_inverse_raises(self, projector):
        x = torch.randn(1, 1, 1, self.N_DST, self.VARS)
        with pytest.raises(NotImplementedError):
            projector.inverse(x)

    def test_whole_model_serialization_roundtrip(self, projector, tmp_path):
        """Whole-model serialization retains the in-memory projection matrix."""
        path = tmp_path / "projector.pt"
        assert not projector.state_dict()
        torch.save(projector, path)

        loaded = torch.load(path, weights_only=False)

        x = torch.randn(self.BATCH, self.TIME, self.ENS, self.N_SRC, self.VARS)
        expected, _ = projector(x)
        actual, _ = loaded(x)
        assert torch.allclose(expected, actual)
