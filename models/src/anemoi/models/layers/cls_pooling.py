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

    _MAX_CHUNK = 32768

    def __init__(self, dim, nhead):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mha = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.residual_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x, key_padding_mask=None):
        """x: (batch, n_tokens, dim) -> (batch, dim)

        key_padding_mask: (batch, n_tokens) bool, True = that token is padding (a shorter
        set batched alongside longer ones) and must be excluded from attention entirely -
        both as something other tokens attend to, and from the residual's mean. Omit
        (default None) when every row in the batch has the same real token count - treated
        as nothing being padded.
        """
        if key_padding_mask is None:
            key_padding_mask = x.new_zeros(x.shape[0], x.shape[1], dtype=torch.bool)

        output_chunks = []
        for x_chunk, mask_chunk in zip(x.split(self._MAX_CHUNK, dim=0), key_padding_mask.split(self._MAX_CHUNK, dim=0)):
            cls_chunk = self.cls_token.expand(x_chunk.shape[0], -1, -1)
            sequence_chunk = self.norm(torch.cat([cls_chunk, x_chunk], dim=1))
            cls_mask_chunk = mask_chunk.new_zeros(mask_chunk.shape[0], 1)
            full_mask_chunk = torch.cat([cls_mask_chunk, mask_chunk], dim=1)
            pooled_chunk = self.mha(
                sequence_chunk, sequence_chunk, sequence_chunk, key_padding_mask=full_mask_chunk, need_weights=False
            )[0][:, 0]

            valid_chunk = (~mask_chunk).unsqueeze(-1).float()
            residual_chunk = (x_chunk * valid_chunk).sum(dim=1) / valid_chunk.sum(dim=1).clamp(min=1)
            pooled_chunk = pooled_chunk + self.residual_proj(residual_chunk)

            output_chunks.append(pooled_chunk + self.ffn(self.ffn_norm(pooled_chunk)))

        return torch.cat(output_chunks, dim=0)
