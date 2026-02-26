# (C) Copyright 2026 Anemoi contributors.
# Licensed under Apache 2.0.

from __future__ import annotations

import os
import time
from typing import Optional

import torch
from torch import nn


def _ddp_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


class _QCDebugPlotter:
    """
    Lightweight, opt-in debug plotting that can be called from forward().
    - Enabled via qc_cfg["debug_plot"] or env QC_DEBUG_PLOT=1
    - Writes PNGs to out_dir (default /tmp/anemoi_qc_debug)
    - Throttled by every_n calls
    - Rank 0 only
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        out_dir: str = "/tmp/anemoi_qc_debug",
        every_n: int = 200,
        source: str = "NORDIC_RADAR",
        bit_colors: Optional[dict[int, tuple[float, float, float]]] = None,
        bit_labels: Optional[dict[int, str]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.out_dir = out_dir
        self.every_n = int(every_n)
        self.source = source
        self.call_i = 0

        if bit_colors is None or bit_labels is None:
            if source == "NORDIC_RADAR":
                self.bit_colors = {
                    0: (0.0, 0.0, 0.0),  # nodata
                    1: (1.0, 0.0, 0.0),  # blocked
                    4: (0.0, 1.0, 1.0),  # sea clutter
                    5: (0.6, 0.4, 0.2),  # ground clutter
                    6: (0.6, 0.6, 0.6),  # other clutter
                    7: (1.0, 0.0, 1.0),  # convective
                    8: (0.0, 1.0, 0.0),  # blocked >50%
                    9: (0.2, 0.2, 1.0),  # extreme
                    10: (0.5, 0.0, 0.8),  # invalid
                }
                self.bit_labels = {
                    0: "nodata",
                    1: "blocked (any)",
                    2: "low elevation",
                    3: "high elevation",
                    4: "sea clutter",
                    5: "ground clutter",
                    6: "other clutter",
                    7: "convective",
                    8: "blocked >50%",
                    9: "extreme (20–50 mm/h)",
                    10: "invalid high (>50 mm/h)",
                }
            else:
                self.bit_colors = {0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0), 2: (1.0, 0.5, 0.0)}
                self.bit_labels = {0: "nodata", 1: "extreme", 2: "invalid"}
        else:
            self.bit_colors = dict(bit_colors)
            self.bit_labels = dict(bit_labels)

    def _should_plot(self) -> bool:
        if not self.enabled:
            return False
        return (self.call_i % self.every_n) == 1

    def _ensure_out_dir(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)

    @torch.no_grad()
    def plot_decode_overlay(self, *, qc_flags_i64: torch.Tensor, tag: str) -> None:
        if not self._should_plot():
            return

        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        self._ensure_out_dir()

        q = qc_flags_i64.detach()
        q = q.reshape(-1).to("cpu")

        P = q.numel()
        print("P", P)
        if P == 0:
            return

        x = np.arange(q.shape[0], dtype=np.float32)
        y = np.zeros_like(x)
        xlabel, ylabel = "point index", ""

        flags = q.numpy().astype(np.int64)

        rgba = np.zeros((flags.size, 4), dtype=np.float32)
        anybit = np.zeros(flags.size, dtype=bool)

        for bit, rgb in self.bit_colors.items():
            m = ((flags >> bit) & 1).astype(bool)
            if not m.any():
                continue
            anybit |= m
            rgba[m, 0] = np.maximum(rgba[m, 0], rgb[0])
            rgba[m, 1] = np.maximum(rgba[m, 1], rgb[1])
            rgba[m, 2] = np.maximum(rgba[m, 2], rgb[2])
        rgba[anybit, 3] = 0.55

        fig = plt.figure(figsize=(7, 7))
        ax = plt.gca()
        ax.scatter(x, y, c=rgba, s=2, linewidths=0, rasterized=True)
        ax.set_title(f"QC flags overlay ({self.source})")
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        present_bits = sorted(bit for bit in self.bit_colors if np.any(((flags >> bit) & 1) == 1))
        handles = [
            Patch(facecolor=(*self.bit_colors[bit], 0.55), edgecolor="none", label=self.bit_labels.get(bit, f"bit {bit}"))
            for bit in present_bits
        ]
        if handles:
            ax.legend(handles=handles, title=f"QC flags ({self.source})", loc="lower left", fontsize=8, title_fontsize=9, framealpha=0.9, ncol=2)

        out = os.path.join(self.out_dir, f"{tag}_qc_overlay_{int(time.time())}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def plot_valid_mask(self, *, valid_mask_f32: torch.Tensor, tag: str) -> None:
        if not self._should_plot():
            return

        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._ensure_out_dir()

        v = valid_mask_f32.detach().reshape(-1).to("cpu")
        if v.numel() == 0:
            return
        
        vv = v.numpy()
        frac_valid = float(np.nanmean(vv)) if vv.size else float("nan")

        fig = plt.figure(figsize=(7, 3))
        ax = plt.gca()
        ax.hist(vv[~np.isnan(vv)], bins=3, range=(-0.1, 1.1))
        ax.set_title(f"Valid mask histogram (mean={frac_valid:.3f})")
        ax.set_xlabel("valid_mask")
        out = os.path.join(self.out_dir, f"{tag}_valid_mask_{int(time.time())}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)


class QCDecodeBits(nn.Module):
    """Decode selected bits from packed qc_flags into float channels.

    Input: qc_flags [...], integer-like
    Output: bits [..., K], float32 in {0,1}
    """

    def __init__(self, bit_indices: list[int], debug_plotter: Optional[_QCDebugPlotter] = None):
        super().__init__()
        if not bit_indices:
            raise ValueError("QCDecodeBits requires a non-empty bit_indices")
        self.register_buffer("bits", torch.tensor(bit_indices, dtype=torch.int64), persistent=False)
        self._dbg = debug_plotter

    def forward(self, qc_flags: torch.Tensor) -> torch.Tensor:
        q = qc_flags.to(torch.int64).unsqueeze(-1)
        b = self.bits.to(qc_flags.device).view(*([1] * (q.ndim - 1)), -1)
        out = ((q >> b) & 1).to(dtype=torch.float32)
        print("DBG", self._dbg, flush=True)
        if self._dbg is not None:
            print("PLOTTING QCDecodeBits overlay", flush=True)
            self._dbg.plot_decode_overlay(qc_flags_i64=qc_flags.to(torch.int64), tag="QCDecodeBits")

        return out


class QCValidMask(nn.Module):
    """Compute valid_mask from packed qc_flags.

    valid_mask = 1 if (qc_flags & invalid_mask) == 0 else 0.
    """

    def __init__(self, bit_indices: list[int], debug_plotter: Optional[_QCDebugPlotter] = None):
        super().__init__()
        if not bit_indices:
            raise ValueError("QCValidMask requires a non-empty bit_indices")
        mask = 0
        for b in bit_indices:
            mask |= 1 << int(b)
        self.register_buffer("invalid_mask", torch.tensor(mask, dtype=torch.int64), persistent=False)
        self._dbg = debug_plotter

    def forward(self, qc_flags: torch.Tensor) -> torch.Tensor:
        q = qc_flags.to(torch.int64)
        valid = (q & self.invalid_mask.to(qc_flags.device)) == 0
        out = valid.to(torch.float32).unsqueeze(-1)
        if self._dbg is not None:
            self._dbg.plot_valid_mask(valid_mask_f32=out, tag="QCValidMask")

        return out


class QCPackedEmbedding(nn.Module):
    """Embed packed qc_flags into a learned vector."""

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
        else:
            raise ValueError("qc_cfg needs either bit_indices or num_bits")

        dbg_cfg = qc_cfg.get("debug_plot", {}) or {}
        print("DBG QCFeaturizer dbg_cfg:", dbg_cfg, flush=True)
        self._dbg = _QCDebugPlotter(
            enabled=bool(dbg_cfg.get("enabled", False)),
            out_dir=str(dbg_cfg.get("out_dir", "/tmp/anemoi_qc_debug")),
            every_n=int(dbg_cfg.get("every_n", 200)),
            source=str(dbg_cfg.get("source", "NORDIC_RADAR")),
        )

        if "invalid_bit_indices" in qc_cfg:
            self.mask = QCValidMask(qc_cfg["invalid_bit_indices"], debug_plotter=self._dbg)

        if method == "decode_bits":
            self.feat = QCDecodeBits(bit_indices, debug_plotter=self._dbg)
        elif method == "embed_bits":
            emb_dim = int(qc_cfg.get("embedding_dim", 0))
            if emb_dim <= 0:
                raise ValueError("embedding_dim must be > 0 for embed_bits")
            self.feat = QCPackedEmbedding(emb_dim=emb_dim, bit_indices=bit_indices)
        else:
            raise ValueError(f"Unknown QC featurizer method: {method}")

    def set_latlons_deg_for_debug(self, latlons_deg: torch.Tensor) -> None:
        """
        Optional: call once from outside (e.g. after graph init) so the debug overlay
        can plot in lon/lat space like your raw-data plot (without cartopy).
        Expected shape [N,2] with columns [lat, lon] OR [lon, lat] as long as consistent
        with your choice in _QCDebugPlotter.set_latlons_deg.
        """
        self._dbg.set_latlons_deg(latlons_deg)

    def forward(self, qc_flags: torch.Tensor) -> torch.Tensor:
        qc_features = self.feat(qc_flags)
        if hasattr(self, "mask"):
            valid_mask = self.mask(qc_flags)
            qc_features = torch.cat([qc_features, valid_mask], dim=-1)
        return qc_features