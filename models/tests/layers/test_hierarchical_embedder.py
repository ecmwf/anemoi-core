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

from anemoi.models.layers.embedder import HierarchicalEmbedder

FEATURE_NAMES = ["t_850", "q_850", "t_500", "q_500", "lsm"]


class TestHierarchicalEmbedder:
    def test_groups_variables_by_level(self):
        embedder = HierarchicalEmbedder(FEATURE_NAMES, hidden_dim=7, d_model=8, nhead=2)

        # 3 groups: level 850 (t_850, q_850), level 500 (t_500, q_500), no-level (lsm)
        assert len(embedder.level_groups) == 3
        assert sorted(len(group) for group in embedder.level_groups) == [1, 2, 2]

    def test_forward_shape_and_finite(self):
        d_model = 8
        embedder = HierarchicalEmbedder(FEATURE_NAMES, hidden_dim=7, d_model=d_model, nhead=2)
        batch, n_time, ensemble, grid = 2, 2, 1, 3
        n_attrs = 4
        x = torch.randn(batch, n_time, ensemble, grid, len(FEATURE_NAMES))
        node_attributes_data = torch.randn(batch * ensemble * grid, n_attrs)

        out = embedder(x, node_attributes_data)

        assert out.shape == (batch * ensemble * grid, n_time * d_model + n_attrs)
        assert torch.isfinite(out).all()

    def test_feature_names_subset_not_supported(self):
        embedder = HierarchicalEmbedder(FEATURE_NAMES, hidden_dim=7, d_model=8, nhead=2)
        x = torch.randn(1, 1, 1, 2, len(FEATURE_NAMES))
        node_attributes_data = torch.zeros(2, 0)

        with pytest.raises(NotImplementedError):
            embedder(x, node_attributes_data, feature_names=FEATURE_NAMES[:2])

    def test_row_chunking_matches_unchunked_result(self):
        """Stage 1 splits rows into chunks bounded by ClsSelfAttentionPool._MAX_CHUNK // n_groups
        - forcing a tiny chunk size (so a small batch still spans several chunks) must give the
        same result as the default, effectively unchunked size."""
        torch.manual_seed(0)
        embedder = HierarchicalEmbedder(FEATURE_NAMES, hidden_dim=7, d_model=8, nhead=2)
        embedder.eval()
        x = torch.randn(1, 1, 1, 5, len(FEATURE_NAMES))
        node_attributes_data = torch.randn(5, 3)

        with torch.no_grad():
            out_unchunked = embedder(x, node_attributes_data)
            embedder.stage1_pool._MAX_CHUNK = 2  # forces multiple row-chunks for 5 rows * 3 groups
            out_chunked = embedder(x, node_attributes_data)

        assert torch.allclose(out_unchunked, out_chunked, atol=1e-5)

    def test_gradients_flow(self):
        embedder = HierarchicalEmbedder(FEATURE_NAMES, hidden_dim=7, d_model=8, nhead=2)
        x = torch.randn(1, 1, 1, 2, len(FEATURE_NAMES))
        node_attributes_data = torch.zeros(2, 0)

        embedder(x, node_attributes_data).sum().backward()

        assert embedder.variable_embedding.weight.grad is not None
        assert embedder.stage1_pool.cls_token.grad is not None
        assert embedder.stage2_pool.cls_token.grad is not None
