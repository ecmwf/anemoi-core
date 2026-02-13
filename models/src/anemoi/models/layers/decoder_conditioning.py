# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0

"""Decoder-only conditioning adapters.

These are lightweight modules intended to inject target-time context
(time embedding + future-known target forcings) *only in the decoder*, keeping
the encoder time-invariant w.r.t. target times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from typing import Optional

import einops
import torch
from torch import Tensor
from torch import nn

from anemoi.utils.config import DotDict


@dataclass
class CondConfig:
    enabled: bool
    key: str = "cond"
    method: str = "film"
    apply_to: str = "x_dst"  # "x_dst" | "latent" | "both"
    cond_dim: int = 0


class DecoderConditioning(nn.Module):
    """Owns:
    - how to locate cond tensors in decoder_context
    - how to flatten cond to (B*E*G, C)
    - per-dataset adapters for x_dst and optional adapters for latent
    - optional union-mesh adapter for x_dst(mesh)
    """

    def __init__(
        self,
        cfg: DotDict,
        *,
        xdst_dim_by_ds: Dict[str, int],
        latent_dim: int,
        mesh_xdst_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        cfg = DotDict(cfg)
        self.cfg = CondConfig(
            enabled=bool(cfg),
            key=str(cfg.get("key", "cond")),
            method=str(cfg.get("method", "film")),
            apply_to=str(cfg.get("apply_to", "x_dst")),
            cond_dim=int(cfg.get("cond_dim", 0)),
        )

        self.xdst_adapters = nn.ModuleDict()
        self.mesh_adapter: Optional[nn.Module] = None
        self.latent_adapter: Optional[nn.Module] = None

        if not self.cfg.enabled:
            return

        if self.cfg.apply_to in {"x_dst", "both"}:
            for ds, xdim in xdst_dim_by_ds.items():
                self.xdst_adapters[ds] = build_decoder_conditioner(
                    method=self.cfg.method, x_dim=int(xdim), cond_dim=self.cfg.cond_dim, cfg=cfg
                )
            if mesh_xdst_dim is not None:
                self.mesh_adapter = build_decoder_conditioner(
                    method=self.cfg.method, x_dim=int(mesh_xdst_dim), cond_dim=self.cfg.cond_dim, cfg=cfg
                )

        if self.cfg.apply_to in {"latent", "both"} and latent_dim > 0:
            self.latent_adapter = build_decoder_conditioner(
                method=self.cfg.method, x_dim=int(latent_dim), cond_dim=self.cfg.cond_dim, cfg=cfg
            )

    def infer_t_out(self, decoder_context: Optional[dict], fallback: int) -> int:
        if not self.cfg.enabled or decoder_context is None:
            return fallback
        k = self.cfg.key
        for _, v in decoder_context.items():
            if isinstance(v, dict) and k in v:
                return int(v[k].shape[1])
        return fallback

    def get_cond(
        self,
        decoder_context: Optional[dict],
        *,
        dataset_name: str,
        t: int,
        union_name: Optional[str] = None,
        concat_union_from_datasets: Optional[list[str]] = None,
    ) -> Optional[Tensor]:
        """Returns cond_flat: [(B*E*G), C] or None.
        Supports union mesh:
          - prefer decoder_context[union_name][key]
          - else decoder_context["__union__"][key]
          - else concat datasets along grid dim in the provided order
        """
        if not self.cfg.enabled or decoder_context is None:
            return None

        k = self.cfg.key

        # union mesh requested
        if union_name is not None and dataset_name == union_name:
            if union_name in decoder_context and k in decoder_context[union_name]:
                cond_t = decoder_context[union_name][k][:, t]
            elif "__union__" in decoder_context and k in decoder_context["__union__"]:
                cond_t = decoder_context["__union__"][k][:, t]
            elif concat_union_from_datasets is not None and all(
                (ds in decoder_context and k in decoder_context[ds]) for ds in concat_union_from_datasets
            ):
                cond_t = torch.cat([decoder_context[ds][k][:, t] for ds in concat_union_from_datasets], dim=2)
            else:
                return None
        else:
            if dataset_name not in decoder_context or k not in decoder_context[dataset_name]:
                return None
            cond_t = decoder_context[dataset_name][k][:, t]  # [B,E,G,C]

        return einops.rearrange(cond_t, "b e g c -> (b e g) c")

    def apply(
        self,
        x_latent: Tensor,
        x_dst: Tensor,
        *,
        cond_flat: Optional[Tensor],
        dataset_name: str,
        is_union_mesh: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if cond_flat is None or not self.cfg.enabled:
            return x_latent, x_dst

        if self.latent_adapter is not None:
            x_latent = self.latent_adapter(x_latent, cond_flat)

        if self.cfg.apply_to in {"x_dst", "both"}:
            if is_union_mesh and self.mesh_adapter is not None:
                x_dst = self.mesh_adapter(x_dst, cond_flat)
            elif dataset_name in self.xdst_adapters:
                x_dst = self.xdst_adapters[dataset_name](x_dst, cond_flat)

        return x_latent, x_dst


class DecoderTimeConditioningFiLM(nn.Module):
    """FiLM conditioning: x <- x * (1 + gamma) + beta."""

    def __init__(self, *, x_dim: int, cond_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * x_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class DecoderTimeConditioningCrossAttention(nn.Module):
    """Cross-attention conditioning.

    Intended for x shaped [B, N, D] and cond_tokens shaped [B, M, Dc].
    """

    def __init__(self, *, x_dim: int, cond_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.q = nn.Linear(x_dim, x_dim)
        self.k = nn.Linear(cond_dim, x_dim)
        self.v = nn.Linear(cond_dim, x_dim)
        self.attn = nn.MultiheadAttention(embed_dim=x_dim, num_heads=num_heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(x_dim), nn.Linear(x_dim, x_dim))

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(cond_tokens)
        v = self.v(cond_tokens)
        y, _ = self.attn(q, k, v, need_weights=False)
        return self.out(x + y)


class DecoderTimeConditioningProjConcat(nn.Module):
    """Projected concatenation: x <- MLP([x, cond])."""

    def __init__(self, *, x_dim: int, cond_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, cond], dim=-1))


def build_decoder_conditioner(*, method: str, x_dim: int, cond_dim: int, cfg: dict | None = None) -> nn.Module:
    cfg = cfg or {}
    method = str(method).lower()
    if method in {"film", "fiLM".lower()}:
        return DecoderTimeConditioningFiLM(x_dim=x_dim, cond_dim=cond_dim, hidden=int(cfg.get("hidden", 128)))
    if method in {"cross_attention", "cross-attn", "xattn"}:
        return DecoderTimeConditioningCrossAttention(
            x_dim=x_dim, cond_dim=cond_dim, num_heads=int(cfg.get("num_heads", 4))
        )
    if method in {"proj_concat", "concat", "projected_concat"}:
        return DecoderTimeConditioningProjConcat(x_dim=x_dim, cond_dim=cond_dim, hidden=int(cfg.get("hidden", 128)))
    raise ValueError(f"Unknown decoder conditioning method '{method}'")
