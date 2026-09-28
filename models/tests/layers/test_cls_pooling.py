# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.models.layers.cls_pooling import ClsSelfAttentionPool


class TestClsSelfAttentionPool:
    def test_forward_shape(self):
        pool = ClsSelfAttentionPool(dim=8, nhead=2)
        x = torch.randn(4, 5, 8)  # batch=4, 5 tokens, dim=8

        out = pool(x)

        assert out.shape == (4, 8)
        assert torch.isfinite(out).all()

    def test_forward_single_token(self):
        pool = ClsSelfAttentionPool(dim=8, nhead=2)
        x = torch.randn(3, 1, 8)

        out = pool(x)

        assert out.shape == (3, 8)
        assert torch.isfinite(out).all()

    def test_gradients_flow_to_cls_token_and_input(self):
        pool = ClsSelfAttentionPool(dim=8, nhead=2)
        x = torch.randn(2, 4, 8, requires_grad=True)

        pool(x).sum().backward()

        assert pool.cls_token.grad is not None
        assert torch.isfinite(pool.cls_token.grad).all()
        assert x.grad is not None

    def test_padding_mask_matches_unpadded_result(self):
        """A padded, masked row must give the same result as running its real tokens alone -
        the padding must have zero effect, not just zero attention weight in isolation (the
        residual's mean has to exclude it too)."""
        torch.manual_seed(0)
        pool = ClsSelfAttentionPool(dim=8, nhead=2)
        pool.eval()

        real = torch.randn(1, 3, 8)  # one row, 3 real tokens
        pad = torch.randn(1, 2, 8)  # 2 padding tokens, arbitrary content - must be ignored
        padded = torch.cat([real, pad], dim=1)
        mask = torch.tensor([[False, False, False, True, True]])

        with torch.no_grad():
            out_unpadded = pool(real)
            out_padded = pool(padded, key_padding_mask=mask)

        assert torch.allclose(out_unpadded, out_padded, atol=1e-5)

    def test_padding_mask_batched_matches_per_row_unpadded(self):
        """Batching several differently-sized rows together (via padding) must give the same
        per-row result as running each row's real tokens through unbatched - the scenario
        HierarchicalEmbedder actually uses this for (levels with different variable counts)."""
        torch.manual_seed(1)
        pool = ClsSelfAttentionPool(dim=8, nhead=2)
        pool.eval()

        row0_real = torch.randn(1, 2, 8)
        row1_real = torch.randn(1, 4, 8)
        max_len = 4
        batched = torch.zeros(2, max_len, 8)
        batched[0, :2] = row0_real
        batched[1, :4] = row1_real
        mask = torch.tensor([[False, False, True, True], [False, False, False, False]])

        with torch.no_grad():
            out_batched = pool(batched, key_padding_mask=mask)
            out_row0 = pool(row0_real)
            out_row1 = pool(row1_real)

        assert torch.allclose(out_batched[0], out_row0[0], atol=1e-5)
        assert torch.allclose(out_batched[1], out_row1[0], atol=1e-5)
