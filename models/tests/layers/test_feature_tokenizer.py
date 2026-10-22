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

from anemoi.models.layers.embedder import FeatureTokenizer

FEATURE_NAMES = ["2t", "q_850", "q_500", "lsm"]


class TestFeatureTokenizer:
    """Test suite for FeatureTokenizer, including variable-subset and frame-idx support in forward()."""

    def test_forward_full_set_shape_and_values(self):
        dim = 8
        tokenizer = FeatureTokenizer(FEATURE_NAMES, dim=dim)
        values = torch.randn(3, len(FEATURE_NAMES))
        frame_idx = torch.zeros(3, dtype=torch.long)

        out = tokenizer(values, frame_idx)

        assert out.shape == (3, len(FEATURE_NAMES), 2 + 3 * dim)
        # no normalization inside the tokenizer - values pass through as-is, normalization is
        # the upstream data preprocessor's job.
        assert torch.equal(out[..., 0], values)

    def test_forward_subset_matches_full_set_rows(self):
        """Excluding a feature must not shift/misalign the remaining ones."""
        dim = 8
        tokenizer = FeatureTokenizer(FEATURE_NAMES, dim=dim)
        values = torch.randn(4, len(FEATURE_NAMES))
        frame_idx = torch.zeros(4, dtype=torch.long)

        out_full = tokenizer(values, frame_idx)

        kept = ["2t", "q_850", "lsm"]  # excludes q_500
        kept_idx = [FEATURE_NAMES.index(name) for name in kept]
        out_subset = tokenizer(values[:, kept_idx], frame_idx, feature_names=kept)

        assert out_subset.shape == (4, len(kept), 2 + 3 * dim)
        assert torch.allclose(out_subset, out_full[:, kept_idx, :])

    def test_forward_subset_order_independent(self):
        """feature_names must be resolved by name, not by position in the original list."""
        dim = 8
        tokenizer = FeatureTokenizer(FEATURE_NAMES, dim=dim)
        values = torch.randn(2, len(FEATURE_NAMES))
        frame_idx = torch.zeros(2, dtype=torch.long)
        out_full = tokenizer(values, frame_idx)

        shuffled = ["lsm", "2t", "q_500"]  # excludes q_850, and reorders the rest
        shuffled_idx = [FEATURE_NAMES.index(name) for name in shuffled]
        out_subset = tokenizer(values[:, shuffled_idx], frame_idx, feature_names=shuffled)

        assert torch.allclose(out_subset, out_full[:, shuffled_idx, :])

    def test_forward_subset_excludes_sole_variable_representative(self):
        """Dropping the only column of a physical variable (no other levels) must still work."""
        dim = 8
        tokenizer = FeatureTokenizer(FEATURE_NAMES, dim=dim)
        values = torch.randn(2, len(FEATURE_NAMES))
        frame_idx = torch.zeros(2, dtype=torch.long)
        out_full = tokenizer(values, frame_idx)

        kept = ["q_850", "q_500", "lsm"]  # excludes "2t", which has no sibling level
        kept_idx = [FEATURE_NAMES.index(name) for name in kept]
        out_subset = tokenizer(values[:, kept_idx], frame_idx, feature_names=kept)

        assert torch.isfinite(out_subset).all()
        assert torch.allclose(out_subset, out_full[:, kept_idx, :])

    def test_forward_unknown_feature_name_raises(self):
        tokenizer = FeatureTokenizer(FEATURE_NAMES, dim=8)
        values = torch.randn(2, 1)
        frame_idx = torch.zeros(2, dtype=torch.long)

        with pytest.raises(KeyError):
            tokenizer(values, frame_idx, feature_names=["not_a_real_feature"])

    def test_forward_mismatched_shape_raises(self):
        tokenizer = FeatureTokenizer(FEATURE_NAMES, dim=8)
        values = torch.randn(2, 2)  # 2 columns, but 3 names given below
        frame_idx = torch.zeros(2, dtype=torch.long)

        with pytest.raises(ValueError):
            tokenizer(values, frame_idx, feature_names=["2t", "q_850", "lsm"])

    def test_forward_single_feature_subset(self):
        dim = 8
        tokenizer = FeatureTokenizer(FEATURE_NAMES, dim=dim)
        values = torch.randn(2, 1)
        frame_idx = torch.zeros(2, dtype=torch.long)

        out = tokenizer(values, frame_idx, feature_names=["lsm"])

        assert out.shape == (2, 1, 2 + 3 * dim)
        assert torch.isfinite(out).all()

    def test_forward_different_frame_idx_gives_different_encoding(self):
        """Two rows with identical values but different frame_idx must not tokenize identically."""
        dim = 8
        tokenizer = FeatureTokenizer(FEATURE_NAMES, dim=dim)
        values = torch.randn(1, len(FEATURE_NAMES)).expand(2, -1)
        frame_idx = torch.tensor([0, 1])

        out = tokenizer(values, frame_idx)

        assert not torch.allclose(out[0], out[1])
