# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Learned embeddings for categorical input variables (e.g. satellite report types).

A categorical variable is carried through the data pipeline as one raw-code channel
(the normaliser must pass it through unchanged) and replaced by a learned vector
inside the network. Index convention:

- ``MISSING`` (0): raw value 0.0 or NaN. 0.0 is the imputer fill and the
  forecast-phase zeroing of corrector inputs, so it means "no observation here".
- ``UNKNOWN`` (1): a non-zero code not in the vocabulary, or a known code replaced
  during training with probability ``unknown_prob``.
- known codes: ``2 .. K + 1`` in vocabulary list order. The list is append-only, so
  existing rows never move when a code is added.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torch import nn

LOGGER = logging.getLogger(__name__)

# float32 represents every integer up to 2**24 exactly.
MAX_CODE = 2**24


class CategoricalEmbedding(nn.Module):
    """Map raw integer codes to learned embedding vectors.

    Parameters
    ----------
    codes : Sequence[int]
        Vocabulary of known codes. Order defines the embedding rows and must only
        ever be appended to.
    embedding_dim : int
        Width of each embedding vector.
    unknown_prob : float, optional
        Training-only probability of replacing a known code with ``UNKNOWN``. Drawn
        once per (batch sample, vocabulary entry) per forward call, so every cell of
        a sample carrying that code is replaced together. By default 0.0.
    """

    MISSING = 0
    UNKNOWN = 1

    def __init__(self, codes: Sequence[int], embedding_dim: int, unknown_prob: float = 0.0) -> None:
        super().__init__()
        codes = validate_codes(codes)
        if embedding_dim < 1:
            msg = f"embedding_dim must be positive, got {embedding_dim}."
            raise ValueError(msg)
        if not 0.0 <= unknown_prob < 1.0:
            msg = f"unknown_prob must be in [0, 1), got {unknown_prob}."
            raise ValueError(msg)

        codes_t = torch.as_tensor(codes, dtype=torch.long)
        order = torch.argsort(codes_t)
        # Persistent so a checkpoint records which vocabulary its rows belong to.
        self.register_buffer("codes", codes_t)
        self.register_buffer("sorted_codes", codes_t[order].to(torch.float32), persistent=False)
        self.register_buffer("perm", order, persistent=False)
        self.unknown_prob = float(unknown_prob)
        self.embedding = nn.Embedding(len(codes) + 2, embedding_dim)
        self._checked_integer = False

    @property
    def num_codes(self) -> int:
        return self.codes.numel()

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    def extra_repr(self) -> str:
        return f"num_codes={self.num_codes}, embedding_dim={self.embedding_dim}, unknown_prob={self.unknown_prob}"

    def codes_to_index(self, raw: Tensor) -> Tensor:
        """Convert raw codes (float32/float64, any shape) to embedding row indices."""
        if raw.dtype not in (torch.float32, torch.float64):
            msg = (
                f"CategoricalEmbedding needs float32 or float64 codes, got {raw.dtype}. Half-precision "
                "types cannot represent large codes exactly; embed before casting."
            )
            raise TypeError(msg)
        r = torch.nan_to_num(raw, nan=0.0)
        rounded = r.round()
        # One device sync: always outside training, and on the first training call.
        if not self.training or not self._checked_integer:
            self._check_integer(r, rounded)
            self._checked_integer = True
        rounded = rounded.to(torch.float32)

        pos = torch.searchsorted(self.sorted_codes, rounded).clamp(max=self.num_codes - 1)
        known = self.sorted_codes[pos] == rounded
        idx = torch.where(known, self.perm[pos] + 2, self.UNKNOWN)
        return torch.where(rounded == 0, self.MISSING, idx)

    @staticmethod
    def _check_integer(r: Tensor, rounded: Tensor) -> None:
        bad = r != rounded
        if bool(bad.any()):
            sample = r[bad][:5].tolist()
            msg = (
                f"CategoricalEmbedding received non-integer codes, e.g. {sample}. One code per cell is "
                "required; check the dataset (gridding may mix codes) and that the normaliser for this "
                "variable is 'none'."
            )
            raise ValueError(msg)

    def _replace_with_unknown(self, idx: Tensor) -> Tensor:
        """Send each known code to ``UNKNOWN`` with probability ``unknown_prob`` per batch sample.

        The mask has shape ``(batch, K + 2)`` and never depends on the grid, so ranks
        sharing an RNG seed draw the same mask whatever their grid shard.
        """
        batch_size = idx.shape[0]
        mask = torch.rand(batch_size, self.num_codes + 2, device=idx.device) < self.unknown_prob
        mask[:, : self.UNKNOWN + 1] = False  # never replace MISSING (or UNKNOWN)
        flat = idx.reshape(batch_size, -1)
        flat = flat.masked_fill(torch.gather(mask, 1, flat), self.UNKNOWN)
        return flat.reshape(idx.shape)

    def forward(self, raw: Tensor) -> Tensor:
        """Embed raw codes of shape ``(batch, ...)`` to ``(batch, ..., embedding_dim)``."""
        idx = self.codes_to_index(raw)
        if self.training and self.unknown_prob > 0:
            idx = self._replace_with_unknown(idx)
        return self.embedding(idx)

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """Accept a checkpoint whose vocabulary is a prefix of this one.

        Rows for appended codes start as copies of the checkpoint's ``UNKNOWN`` row,
        which is what those codes were mapped to before.
        """
        codes_key, weight_key = prefix + "codes", prefix + "embedding.weight"
        old_codes = state_dict.get(codes_key)
        old_weight = state_dict.get(weight_key)
        if old_codes is not None and old_weight is not None and old_codes.numel() < self.num_codes:
            n_old = old_codes.numel()
            if torch.equal(old_codes.to(self.codes.device), self.codes[:n_old]):
                n_new = self.num_codes - n_old
                pad = old_weight[self.UNKNOWN].unsqueeze(0).expand(n_new, -1)
                state_dict[weight_key] = torch.cat([old_weight, pad], dim=0)
                state_dict[codes_key] = self.codes.clone()
                LOGGER.info("%s: extended vocabulary by %d code(s), initialised from UNKNOWN.", prefix, n_new)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def validate_codes(codes: Sequence[int]) -> list[int]:
    """Check a vocabulary: non-empty, unique, non-zero integers exactly representable in float32."""
    codes = list(codes)
    if not codes:
        msg = "A categorical vocabulary needs at least one code."
        raise ValueError(msg)
    as_int = []
    for code in codes:
        if isinstance(code, bool) or float(code) != int(code):
            msg = f"Categorical codes must be integers, got {code!r}."
            raise ValueError(msg)
        as_int.append(int(code))
    if 0 in as_int:
        msg = "Code 0 is reserved for MISSING and cannot be in the vocabulary."
        raise ValueError(msg)
    if any(abs(c) >= MAX_CODE for c in as_int):
        msg = f"Categorical codes must have |code| < 2**24 to be exact in float32, got {max(as_int, key=abs)}."
        raise ValueError(msg)
    duplicates = sorted({c for c in as_int if as_int.count(c) > 1})
    if duplicates:
        msg = f"Duplicate categorical codes: {duplicates}."
        raise ValueError(msg)
    return as_int


def build_categorical_embeddings(
    specs: Mapping[str, Mapping[str, Any]],
    *,
    embedding_dim: int | None = None,
    unknown_prob: float | None = None,
) -> nn.ModuleDict:
    """Build a ``ModuleDict`` of :class:`CategoricalEmbedding` keyed by variable name.

    Parameters
    ----------
    specs : Mapping[str, Mapping[str, Any]]
        ``{variable: {codes: [...], embedding_dim: int, unknown_prob: float}}``.
    embedding_dim, unknown_prob : optional
        Overrides applied to every variable (e.g. the corrector's own settings).

    Returns
    -------
    nn.ModuleDict
        One embedding per variable, in ``specs`` order.
    """
    embeddings = nn.ModuleDict()
    for name, spec in specs.items():
        dim = embedding_dim if embedding_dim is not None else spec.get("embedding_dim", 8)
        prob = unknown_prob if unknown_prob is not None else spec.get("unknown_prob", 0.0)
        embeddings[name] = CategoricalEmbedding(list(spec["codes"]), int(dim), float(prob))
    return embeddings
