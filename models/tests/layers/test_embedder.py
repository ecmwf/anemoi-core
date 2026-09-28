# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.models.layers.embedder import PMAEmbedder
from anemoi.models.layers.embedder import FlattenInputTransform

FEATURE_NAMES = ["2t", "q_850", "q_500", "lsm"]


class TestEmbedder:
    """Test suite for PMAEmbedder, including variable-subset and multi-frame support in forward()."""

    def test_forward_full_set_shape(self):
        d_model = 16
        n_attrs = 5
        embedder = PMAEmbedder(FEATURE_NAMES, dim=8, d_model=d_model, nhead=4)
        x = torch.randn(1, 1, 1, 3, len(FEATURE_NAMES))  # batch=1, time=1, ensemble=1, grid=3 nodes
        node_attributes_data = torch.randn(3, n_attrs)

        with torch.no_grad():
            out = embedder(x, node_attributes_data)

        assert out.shape == (3, d_model + n_attrs)
        assert torch.isfinite(out).all()
        assert torch.equal(out[:, d_model:], node_attributes_data)

    def test_forward_subset_shape_and_finite(self):
        d_model = 16
        n_attrs = 5
        embedder = PMAEmbedder(FEATURE_NAMES, dim=8, d_model=d_model, nhead=4)
        kept = ["2t", "q_850", "lsm"]  # excludes q_500
        x = torch.randn(1, 1, 1, 3, len(kept))
        node_attributes_data = torch.randn(3, n_attrs)

        with torch.no_grad():
            out = embedder(x, node_attributes_data, feature_names=kept)

        assert out.shape == (3, d_model + n_attrs)
        assert torch.isfinite(out).all()

    def test_output_dim_matches_d_model(self):
        embedder = PMAEmbedder(FEATURE_NAMES, dim=8, d_model=16, nhead=4)
        assert embedder.output_dim == 16

    def test_forward_multiple_frames_shape(self):
        d_model = 16
        n_attrs = 5
        embedder = PMAEmbedder(FEATURE_NAMES, dim=8, d_model=d_model, nhead=4)
        batch, n_time, ensemble, grid = 2, 3, 1, 4
        x = torch.randn(batch, n_time, ensemble, grid, len(FEATURE_NAMES))
        node_attributes_data = torch.randn(batch * ensemble * grid, n_attrs)

        with torch.no_grad():
            out = embedder(x, node_attributes_data)

        assert out.shape == (batch * ensemble * grid, n_time * d_model + n_attrs)
        assert torch.isfinite(out).all()


class TestFlattenInputTransform:
    """FlattenInputTransform must exactly match what a plain flatten-timesteps-into-features did before."""

    def test_forward_matches_manual_rearrange(self):
        transform = FlattenInputTransform(feature_names=FEATURE_NAMES)
        batch, n_time, ensemble, grid = 2, 2, 1, 3
        n_attrs = 5
        x = torch.randn(batch, n_time, ensemble, grid, len(FEATURE_NAMES))
        node_attributes_data = torch.randn(batch * ensemble * grid, n_attrs)

        out = transform(x, node_attributes_data)

        assert out.shape == (batch * ensemble * grid, n_time * len(FEATURE_NAMES) + n_attrs)
        # timesteps for the same node must be adjacent and unchanged, not shuffled.
        assert torch.equal(out[0, : len(FEATURE_NAMES)], x[0, 0, 0, 0])
        assert torch.equal(out[0, len(FEATURE_NAMES) : 2 * len(FEATURE_NAMES)], x[0, 1, 0, 0])
        assert torch.equal(out[0, 2 * len(FEATURE_NAMES) :], node_attributes_data[0])
