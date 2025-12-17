# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

import torch

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.multivariate_kcrps import _slice_grid_if_needed
from anemoi.training.losses.multivariate_kcrps import _split_scalers
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


class VariogramScore(BaseLoss):
    """Variogram score for ensemble forecasts (feature variogram).

    Adapted to anemoi-core loss API:
    - pred:   (bs, ens, grid, var)
    - target: (bs, grid, var)

    This implementation computes a *feature variogram* (pairwise over variables) per grid point:
    - For each grid point, compute the matrix |y_i - y_j|^{beta1}
    - For forecasts, average |x_i - x_j|^{beta1} across ensemble members
    - Score is the average over (i != j) of |V_obs - V_fcst|^{beta2}

    Scaling notes:
    - VAR-including scalers are interpreted inside the variogram metric by scaling components
      by scaler^(1/beta1) before forming pairwise differences.
    - GRID-only scalers are applied to the per-grid score output.
    """

    def __init__(
        self,
        *,
        beta1: float = 1.0,
        beta2: float = 2.0,
        ignore_nans: bool = False,
        eps: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        if beta1 <= 0:
            msg = "beta1 must be > 0"
            raise ValueError(msg)
        if beta2 <= 0:
            msg = "beta2 must be > 0"
            raise ValueError(msg)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def _apply_vector_scalers(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        without_scalers: list[str] | list[int] | None,
        grid_shard_slice: slice | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var_scalers, _ = _split_scalers(self.scaler, without_scalers=without_scalers)
        if len(var_scalers) == 0:
            return pred, target

        w = var_scalers.get_scaler(pred.ndim).to(pred.device)
        w = _slice_grid_if_needed(w, grid_shard_slice)
        if self.eps > 0:
            w = torch.clamp(w, min=self.eps)
        comp = w.pow(1.0 / self.beta1)
        return pred * comp, target * comp

    def _apply_nonvar_scalers(
        self,
        out: torch.Tensor,
        *,
        without_scalers: list[str] | list[int] | None,
        grid_shard_slice: slice | None,
    ) -> torch.Tensor:
        _, non_var_scalers = _split_scalers(self.scaler, without_scalers=without_scalers)
        if len(non_var_scalers) == 0:
            return out
        if TensorDim.GRID.value not in non_var_scalers:
            error_msg = (
                "Scaler tensor must be at least applied to the GRID dimension. "
                "Please add a scaler here, use `UniformWeights` for simple uniform scaling."
            )
            raise RuntimeError(error_msg)
        return non_var_scalers.scale_iteratively(out, subset_indices=None, grid_shard_slice=grid_shard_slice)

    @staticmethod
    def _pairwise_diffs(x: torch.Tensor) -> torch.Tensor:
        """Return pairwise absolute diffs over VARIABLE dim.

        Input:  (..., var)
        Output: (..., var, var)
        """
        return torch.abs(x.unsqueeze(-1) - x.unsqueeze(-2))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: Any = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **_: Any,
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None

        if target.ndim != 3:
            msg = f"target must have shape (bs, grid, var); got ndim={target.ndim}"
            raise ValueError(msg)

        # Optional subsetting (same convention as other losses: `[..., indices]`)
        if scaler_indices is not None:
            pred = pred[scaler_indices]
            target = target[scaler_indices]

        target_e = target.unsqueeze(TensorDim.ENSEMBLE_DIM.value)  # (bs,1,grid,var)
        pred_s, target_s = self._apply_vector_scalers(
            pred,
            target_e,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        # obs variogram: (bs, grid, var, var)
        obs = target_s[:, 0]  # deterministic target
        obs_v = self._pairwise_diffs(obs).pow(self.beta1)

        # forecast variogram: average over ensemble -> (bs, grid, var, var)
        fcst_v = self._pairwise_diffs(pred_s).pow(self.beta1).mean(dim=TensorDim.ENSEMBLE_DIM.value)

        var = obs.shape[-1]
        if var < 2:
            msg = "VariogramScore requires at least 2 variables to form pairwise differences"
            raise ValueError(msg)

        score_mat = torch.abs(obs_v - fcst_v).pow(self.beta2)  # (bs, grid, var, var)
        # diagonal is zero; normalize by var*(var-1) (exclude diagonal count)
        score_grid = score_mat.sum(dim=(-1, -2)) / (var * (var - 1))  # (bs, grid)

        out = score_grid.unsqueeze(TensorDim.ENSEMBLE_DIM.value).unsqueeze(TensorDim.VARIABLE.value)  # (bs,1,grid,1)
        out = self._apply_nonvar_scalers(out, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(out, squash=squash, squash_mode="sum", group=group if is_sharded else None)

    @cached_property
    def name(self) -> str:
        return f"vgram_b1{self.beta1:g}_b2{self.beta2:g}"
