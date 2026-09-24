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
    result. Standard transformer-block pairing: attention sublayer, then feed-forward sublayer,
    each with its own residual + pre-LN - same structure as Set Transformer's MAB/PMA blocks.
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
        # gives the gradient a direct path around the attention call, since the two stacked
        # pooling stages otherwise have no skip connection between them.
        self.residual_proj = nn.Linear(dim, dim)
        # pre-LN: normalizes what the attention sees, not the residual path itself - keeps
        # activation scale from growing across the two stacked pooling stages.
        self.norm = nn.LayerNorm(dim)
        # attention alone is linear in the values it combines (a weighted average) - the FFN
        # sublayer is what gives the block genuine nonlinear capacity, same pairing every
        # well-studied attention-pooling block uses (Set Transformer's MAB/PMA, a plain
        # transformer encoder layer).
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        # x: (batch, n_tokens, dim) -> (batch, dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        sequence = self.norm(torch.cat([cls, x], dim=1))
        pooled_chunks = [
            self.mha(chunk, chunk, chunk, need_weights=False)[0][:, 0]
            for chunk in sequence.split(self._MAX_CHUNK, dim=0)
        ]
        pooled = torch.cat(pooled_chunks, dim=0)
        pooled = pooled + self.residual_proj(x.mean(dim=1))
        return pooled + self.ffn(self.ffn_norm(pooled))
