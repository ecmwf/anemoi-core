# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0

"""QC flag featurisation utilities.

Implements the data model described in "Unified QC-Aware Observation Interpolator
Repository Design":

  qc_flags: packed integer bitmask (authoritative)
  invalid_mask: integer bitmask defining *hard invalid* bits

Derived features:
  - valid_mask: 1 where (qc_flags & invalid_mask) == 0 else 0
  - decoded bits: selected bits as {0,1} float channels
  - packed embedding: learned embedding of packed qc flags (optionally restricted
    to a subset of bits)

All modules are pure PyTorch and can be used inside the model graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class QCDecodeBits(nn.Module):
    """Decode selected bits from packed qc_flags into float channels.

    Input: qc_flags [...], integer-like
    Output: bits [..., K], float32 in {0,1}
    """

    def __init__(self, bit_indices: list[int]):
        super().__init__()
        if not bit_indices:
            raise ValueError("QCDecodeBits requires a non-empty bit_indices")
        self.register_buffer("bits", torch.tensor(bit_indices, dtype=torch.int64), persistent=False)

    def forward(self, qc_flags: torch.Tensor) -> torch.Tensor:
        q = qc_flags.to(torch.int64)
        q = q.unsqueeze(-1)  # [..., 1]
        b = self.bits.view(*([1] * (q.ndim - 1)), -1)  # [..., K]
        return ((q >> b) & 1).to(torch.float32)


class QCValidMask(nn.Module):
    """Compute valid_mask from packed qc_flags.

    valid_mask = 1 if (qc_flags & invalid_mask) == 0 else 0.

    If invalid_mask == 0, uses strict legacy semantics: valid iff qc_flags == 0.
    """

    def __init__(self, invalid_mask: int):
        super().__init__()
        self.invalid_mask = int(invalid_mask)

    def forward(self, qc_flags: torch.Tensor) -> torch.Tensor:
        q = qc_flags.to(torch.int64)
        if self.invalid_mask == 0:
            valid = q == 0
        else:
            valid = (q & self.invalid_mask) == 0
        return valid.to(torch.float32)


class QCPackedEmbedding(nn.Module):
    """Embed packed qc_flags into a learned vector.

    If bit_indices are provided, qc_flags are first compressed into a dense
    integer in [0, 2^K) using those bits.
    """

    def __init__(self, emb_dim: int, bit_indices: Optional[list[int]] = None):
        super().__init__()
        if emb_dim <= 0:
            raise ValueError("QCPackedEmbedding requires emb_dim > 0")
        self.bit_indices = bit_indices

        if bit_indices is None:
            # safe upper bound; in practice prefer restricting bits.
            vocab = 2**16
        else:
            if len(bit_indices) > 16:
                raise ValueError("QCPackedEmbedding: bit_indices too large; prefer <= 16")
            vocab = 2 ** len(bit_indices)
            self.register_buffer("bits", torch.tensor(bit_indices, dtype=torch.int64), persistent=False)

        self.emb = nn.Embedding(vocab, emb_dim)

    def forward(self, qc_flags: torch.Tensor) -> torch.Tensor:
        q = qc_flags.to(torch.int64)
        if self.bit_indices is None:
            ids = q.clamp_min(0).clamp_max(self.emb.num_embeddings - 1)
        else:
            # pack selected bits into 0..2^K-1
            q_ = q.unsqueeze(-1)  # [..., 1]
            b = self.bits.view(*([1] * (q_.ndim - 1)), -1)  # [..., K]
            packed = ((q_ >> b) & 1).to(torch.int64)  # [..., K]
            shifts = torch.arange(packed.shape[-1], device=packed.device, dtype=torch.int64)
            ids = (packed << shifts).sum(dim=-1)

        return self.emb(ids)


@dataclass(frozen=True)
class QCFeatures:
    valid_mask: torch.Tensor
    bits: Optional[torch.Tensor] = None
    embedding: Optional[torch.Tensor] = None


class QCFeaturizer(nn.Module):
    """Convenience wrapper producing QCFeatures from qc_flags."""

    def __init__(
        self,
        *,
        invalid_mask: int,
        decoded_bits: Optional[list[int]] = None,
        embedding_dim: int = 0,
        embedding_bits: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.valid = QCValidMask(invalid_mask)
        self.decode = QCDecodeBits(decoded_bits) if decoded_bits else None
        self.emb = QCPackedEmbedding(embedding_dim, bit_indices=embedding_bits) if embedding_dim > 0 else None

    def forward(self, qc_flags: torch.Tensor) -> QCFeatures:
        valid_mask = self.valid(qc_flags)
        bits = self.decode(qc_flags) if self.decode is not None else None
        emb = self.emb(qc_flags) if self.emb is not None else None
        return QCFeatures(valid_mask=valid_mask, bits=bits, embedding=emb)
