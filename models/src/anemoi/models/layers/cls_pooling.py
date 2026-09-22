# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import torch.nn as nn


class ClsSelfAttentionPool(nn.Module):
    """Pools a set of tokens into one, via self-attention with a prepended learned CLS token -
    same mechanism BERT uses to summarize a sequence into a single vector: every token
    (including CLS) attends to every other token, and the CLS row of the output is the pooled
    result. No residual, no feed-forward - just the attention call itself.
    """

    # PyTorch's fused SDPA kernels (flash/efficient attention) hit a CUDA kernel
    # launch limit around 65535 in the batch dimension - our "batch" here is really
    # batch*time*ensemble*grid, which exceeds that at O96 resolution. Chunking keeps
    # each call under the limit without changing the result, since rows are independent.
    _MAX_CHUNK = 32768

    def __init__(self, dim, nhead):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mha = nn.MultiheadAttention(dim, nhead, batch_first=True)

    def forward(self, x):
        # x: (batch, n_tokens, dim) -> (batch, dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        sequence = torch.cat([cls, x], dim=1)
        pooled_chunks = [
            self.mha(chunk, chunk, chunk, need_weights=False)[0][:, 0]
            for chunk in sequence.split(self._MAX_CHUNK, dim=0)
        ]
        return torch.cat(pooled_chunks, dim=0)
