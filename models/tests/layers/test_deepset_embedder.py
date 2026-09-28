# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.models.layers.embedder import DeepSetEmbedder

FEATURE_NAMES = ["2t", "q_850", "q_500", "lsm"]


class TestDeepSetEmbedder:
    """Test suite for DeepSetEmbedder, including variable-subset and multi-frame support."""

    def test_forward_full_set_shape(self):
        d_model = 16
        n_attrs = 5
        embedder = DeepSetEmbedder(FEATURE_NAMES, dim=8, d_model=d_model)
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
        embedder = DeepSetEmbedder(FEATURE_NAMES, dim=8, d_model=d_model)
        kept = ["2t", "q_850", "lsm"]  # excludes q_500
        x = torch.randn(1, 1, 1, 3, len(kept))
        node_attributes_data = torch.randn(3, n_attrs)

        with torch.no_grad():
            out = embedder(x, node_attributes_data, feature_names=kept)

        assert out.shape == (3, d_model + n_attrs)
        assert torch.isfinite(out).all()

    def test_output_dim_matches_d_model(self):
        embedder = DeepSetEmbedder(FEATURE_NAMES, dim=8, d_model=16)
        assert embedder.output_dim == 16

    def test_forward_multiple_frames_shape(self):
        d_model = 16
        n_attrs = 5
        embedder = DeepSetEmbedder(FEATURE_NAMES, dim=8, d_model=d_model)
        batch, n_time, ensemble, grid = 2, 3, 1, 4
        x = torch.randn(batch, n_time, ensemble, grid, len(FEATURE_NAMES))
        node_attributes_data = torch.randn(batch * ensemble * grid, n_attrs)

        with torch.no_grad():
            out = embedder(x, node_attributes_data)

        assert out.shape == (batch * ensemble * grid, n_time * d_model + n_attrs)
        assert torch.isfinite(out).all()

    def test_forward_identical_frames_give_identical_output(self):
        """No frame_idx by design: identical raw values at different timesteps must tokenize
        identically (unlike PMAEmbedder, which deliberately distinguishes them)."""
        d_model = 16
        embedder = DeepSetEmbedder(FEATURE_NAMES, dim=8, d_model=d_model)
        node_attributes_data = torch.randn(1, 2)
        frame = torch.randn(1, 1, 1, 1, len(FEATURE_NAMES))
        x = frame.expand(1, 2, 1, 1, len(FEATURE_NAMES))

        with torch.no_grad():
            out = embedder(x, node_attributes_data)

        first_frame, second_frame = out[:, :d_model], out[:, d_model : 2 * d_model]
        assert torch.allclose(first_frame, second_frame)

    def test_gradients_flow(self):
        embedder = DeepSetEmbedder(FEATURE_NAMES, dim=8, d_model=16)
        x = torch.randn(1, 1, 1, 3, len(FEATURE_NAMES))
        node_attributes_data = torch.randn(3, 5)

        out = embedder(x, node_attributes_data)
        out.sum().backward()

        assert embedder.feature_tokenizer.variable_embedding.weight.grad is not None
        assert embedder.phi[0].weight.grad is not None
        assert embedder.rho[0].weight.grad is not None
