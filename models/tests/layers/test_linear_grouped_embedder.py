# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.layers.embedder import LinearGroupedEmbedder

# every level has the same 2 variables (t, q) - the fixed-variable-set assumption holds.
# no "2t" here deliberately - it parses as variable="t", level=2 (a real gotcha confirmed
# earlier), which would silently become its own uneven 1-variable level group.
FEATURE_NAMES = ["t_850", "q_850", "t_500", "q_500", "lsm"]


class TestLinearGroupedEmbedder:
    def test_rejects_uneven_variable_count_per_level(self):
        # t_850 has no q_850 sibling here - level 850 (1 var) and level 500 (2 vars) differ.
        uneven = ["t_850", "t_500", "q_500"]
        with pytest.raises(ValueError, match="same variable"):
            LinearGroupedEmbedder(uneven, d_model=8, nhead=2)

    def test_forward_shape_and_finite(self):
        d_model = 8
        embedder = LinearGroupedEmbedder(FEATURE_NAMES, d_model=d_model, nhead=2)
        batch, n_time, ensemble, grid = 2, 2, 1, 3
        n_attrs = 4
        x = torch.randn(batch, n_time, ensemble, grid, len(FEATURE_NAMES))
        node_attributes_data = torch.randn(batch * ensemble * grid, n_attrs)

        out = embedder(x, node_attributes_data)

        assert out.shape == (batch * ensemble * grid, n_time * d_model + n_attrs)
        assert torch.isfinite(out).all()
        assert torch.equal(out[:, n_time * d_model :], node_attributes_data)

    def test_output_dim_matches_d_model(self):
        embedder = LinearGroupedEmbedder(FEATURE_NAMES, d_model=8, nhead=2)
        assert embedder.output_dim == 8

    def test_feature_names_subset_not_supported(self):
        embedder = LinearGroupedEmbedder(FEATURE_NAMES, d_model=8, nhead=2)
        x = torch.randn(1, 1, 1, 2, len(FEATURE_NAMES))
        node_attributes_data = torch.zeros(2, 0)

        with pytest.raises(NotImplementedError):
            embedder(x, node_attributes_data, feature_names=FEATURE_NAMES[:2])

    def test_gradients_flow(self):
        embedder = LinearGroupedEmbedder(FEATURE_NAMES, d_model=8, nhead=2)
        x = torch.randn(1, 1, 1, 2, len(FEATURE_NAMES))
        node_attributes_data = torch.zeros(2, 0)

        embedder(x, node_attributes_data).sum().backward()

        assert embedder.level_proj.weight.grad is not None
        assert embedder.surface_proj.weight.grad is not None
        assert embedder.pool.cls_token.grad is not None
