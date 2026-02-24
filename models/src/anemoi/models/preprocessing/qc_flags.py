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
        q = qc_flags.to(torch.int64).unsqueeze(-1)  # stays on qc_flags.device
        b = self.bits.to(qc_flags.device).view(*([1] * (q.ndim - 1)), -1)
        return ((q >> b) & 1).to(dtype=torch.float32)


class QCValidMask(nn.Module):
    """Compute valid_mask from packed qc_flags.

    valid_mask = 1 if (qc_flags & invalid_mask) == 0 else 0.

    If invalid_mask == 0, uses strict legacy semantics: valid iff qc_flags == 0.
    """

    def __init__(self, bit_indices: list[int]):
        super().__init__()
        if not bit_indices:
            raise ValueError("QCValidMask requires a non-empty bit_indices")
        mask = 0
        for b in bit_indices:
            mask |= 1 << int(b)
        self.register_buffer("invalid_mask", torch.tensor(mask, dtype=torch.int64), persistent=False)

    def forward(self, qc_flags: torch.Tensor) -> torch.Tensor:
        q = qc_flags.to(torch.int64)
        valid = (q & self.invalid_mask.to(qc_flags.device)) == 0
        return valid.to(torch.float32).unsqueeze(-1)


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
            q_ = q.unsqueeze(-1)
            b = self.bits.to(qc_flags.device).view(*([1] * (q_.ndim - 1)), -1)
            packed = ((q_ >> b) & 1).to(torch.int64)
            shifts = torch.arange(packed.shape[-1], device=qc_flags.device, dtype=torch.int64)
            ids = (packed << shifts).sum(dim=-1)
        return self.emb(ids)


class QCFeaturizer(nn.Module):
    def __init__(self, *, method: str, qc_cfg: dict) -> None:
        super().__init__()
        if qc_cfg is None:
            raise ValueError("qc_cfg must be provided")
        if "bit_indices" in qc_cfg and "num_bits" in qc_cfg:
            raise ValueError("qc_cfg cannot have both bit_indices and num_bits")
        if "bit_indices" in qc_cfg and "num_bits" not in qc_cfg:
            bit_indices = qc_cfg["bit_indices"]
        elif "num_bits" in qc_cfg and "bit_indices" not in qc_cfg:
            num_bits = qc_cfg.get("num_bits", None)
            if num_bits is None:
                raise ValueError("qc_cfg needs either bit_indices or num_bits")
            bit_indices = list(range(int(num_bits)))

        if "invalid_bit_indices" in qc_cfg:
            self.mask = QCValidMask(qc_cfg["invalid_bit_indices"])

        if method == "decode_bits":
            self.feat = QCDecodeBits(bit_indices)

        elif method == "embed_bits":
            emb_dim = int(qc_cfg.get("embedding_dim", 0))
            if emb_dim <= 0:
                raise ValueError("embedding_dim must be > 0 for embed_bits")
            self.feat = QCPackedEmbedding(emb_dim=emb_dim, bit_indices=bit_indices)

        else:
            raise ValueError(f"Unknown QC featurizer method: {method}")

    def forward(self, qc_flags: torch.Tensor) -> torch.Tensor:
        qc_features = self.feat(qc_flags)
        if hasattr(self, "mask"):
            valid_mask = self.mask(qc_flags)
            qc_features = torch.cat([qc_features, valid_mask], dim=-1)
        return qc_features
