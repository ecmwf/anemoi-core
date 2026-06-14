"""Muon (Newton-Schulz orthogonalized momentum) for >=2D matrix params, with an
AdamW fallback for non-matrix (<2D) params — embeddings/norms/biases/scalars.

Adapter around the vetted ``pytorch_optimizer.Muon``, which requires explicit
param groups each tagged ``use_muon``. The anemoi training loop instantiates an
optimizer as ``instantiate(cfg, params=<flat tensor iterable>, lr=<scaled lr>)``,
so this wrapper accepts that flat iterable, splits it by ``ndim``, builds the
two ``use_muon`` groups, and forwards to ``Muon``. DDP-safe: gradients are
all-reduced before ``step()``, so every rank computes identical updates.

``lr`` is the Muon (matrix) learning rate driven by the lane's cosine schedule;
``adamw_lr`` is the fallback AdamW base rate for the non-matrix group.
"""

from __future__ import annotations

from typing import Any

from pytorch_optimizer import Muon as _Muon


class MuonAuxAdamW(_Muon):
    def __init__(
        self,
        params: Any,
        lr: float = 2e-2,
        adamw_lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_wd: float = 0.01,
        adamw_eps: float = 1e-8,
        ns_steps: int = 5,
        nesterov: bool = True,
        **kwargs: Any,
    ) -> None:
        params = list(params)
        if params and isinstance(params[0], dict):
            groups = params  # already grouped (e.g. on checkpoint reload)
        else:
            matrix = [p for p in params if p.ndim >= 2]
            other = [p for p in params if p.ndim < 2]
            groups = []
            if matrix:
                groups.append({"params": matrix, "use_muon": True})
            if other:
                groups.append({"params": other, "use_muon": False})
        super().__init__(
            groups,
            lr=lr,
            adamw_lr=adamw_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            adamw_betas=adamw_betas,
            adamw_wd=adamw_wd,
            adamw_eps=adamw_eps,
            ns_steps=ns_steps,
            nesterov=nesterov,
            **kwargs,
        )
