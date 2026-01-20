# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging

import torch

from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)


class QCBitsProcessor(BasePreprocessor):
    """Expand an integer qc_flags variable into bit features + validity flag.

    This is a lightweight, dataset-agnostic preprocessor meant for *nowcasting*
    experiments where qc information should be provided to the network.

    It expects an integer-valued variable called `qc_flags` in the model input.
    The processor will:
      - keep the original qc_flags (optional)
      - append `num_bits` binary channels (bit 0..num_bits-1)
      - append one `valid` channel (1 if qc_flags == 0 else 0)

    The number of bits is configured per dataset.
    """

    def __init__(
        self,
        config=None,
        data_indices=None,
        statistics=None,
    ) -> None:
        super().__init__(config=config or {}, data_indices=data_indices, statistics=statistics)
        cfg = config or {}
        self.num_bits = int(cfg.get("num_bits", 0))
        if self.num_bits <= 0:
            raise ValueError("QCBitsProcessor requires config.num_bits > 0")
        self.keep_qc = bool(cfg.get("keep_qc", False))
        self.add_valid = bool(cfg.get("add_valid", True))
        self.qc_var = str(cfg.get("qc_var", "qc_flags"))

    def transform(self, x, in_place: bool = True, **kwargs):
        # x is expected: (batch, time, ensemble, grid, vars)
        if not in_place:
            x = x.clone()

        name_to_index = getattr(self.data_indices.model.input, "name_to_index", None)
        if name_to_index is None or self.qc_var not in name_to_index:
            raise KeyError(
                f"QCBitsProcessor: '{self.qc_var}' not found in model input variables. "
                f"Known: {list((name_to_index or {}).keys())}"
            )

        qc_idx = int(name_to_index[self.qc_var])
        qc = x[..., qc_idx].to(torch.int64)

        # bits: (..., num_bits)
        bits = ((qc.unsqueeze(-1) >> torch.arange(self.num_bits, device=qc.device)) & 1).to(x.dtype)

        feats = [bits]
        if self.add_valid:
            valid = (qc == 0).to(x.dtype).unsqueeze(-1)
            feats.append(valid)

        extra = torch.cat(feats, dim=-1)

        # remove or keep qc
        if self.keep_qc:
            x_out = torch.cat([x, extra], dim=-1)
        else:
            x_out = torch.cat([x[..., :qc_idx], x[..., qc_idx + 1 :], extra], dim=-1)

        return x_out

    def inverse_transform(self, x, in_place: bool = True, **kwargs):
        # QC features are not invertible; pass-through.
        if not in_place:
            x = x.clone()
        return x
