# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import re
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import torch

from anemoi.training.losses.base import BaseLoss
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

    from anemoi.training.losses.scaler_tensor import ScaleTensor

LOGGER = logging.getLogger(__name__)

SubsetIndices = Any  # convention in repo: often passed as `[..., indices]`


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _split_scalers(
    scaler: ScaleTensor,
    *,
    without_scalers: list[str] | list[int] | None,
) -> tuple[ScaleTensor, ScaleTensor]:
    """Split scalers into those affecting VARIABLE and those that do not.

    - VARIABLE scalers must be applied *inside* multivariate distance computations.
    - non-VARIABLE scalers (e.g. GRID/node weights) can be applied to the reduced output.
    """
    active = scaler
    if without_scalers is not None and len(without_scalers) > 0:
        active = active.without(without_scalers)

    var_scalers = active.subset_by_dim(TensorDim.VARIABLE.value)
    non_var_scalers = active.without_by_dim([TensorDim.VARIABLE.value])
    return var_scalers, non_var_scalers


def _slice_grid_if_needed(x: torch.Tensor, grid_shard_slice: slice | None) -> torch.Tensor:
    if grid_shard_slice is None:
        return x
    # mirror ScaleTensor.scale_iteratively() grid slicing behaviour
    if x.ndim > TensorDim.GRID and x.shape[TensorDim.GRID] > 1 and x.shape[TensorDim.GRID] >= grid_shard_slice.stop:
        slicer = [slice(None)] * x.ndim
        slicer[TensorDim.GRID] = grid_shard_slice
        return x[tuple(slicer)]
    return x


class MultivariateKernelCRPS(BaseLoss):
    """Multivariate (kernel) CRPS / Energy-style score using vector norms across variables.

    This implementation is adapted to anemoi-core's loss API:
    - pred:   (bs, ens, grid, var)
    - target: (bs, grid, var)  (deterministic target)

    Important note on scaling:
    - Per-variable (and VAR-including) scalers cannot be applied post-hoc after collapsing the
      variable dimension with a norm. We therefore interpret VAR-including scalers as defining
      a weighted distance metric by scaling the components prior to computing the norm.
    - GRID-only scalers are applied to the per-grid score output.
    """

    def __init__(
        self,
        *,
        fair: bool = True,
        p_norm: float = 1.5,
        beta: float = 1.0,
        implementation: Literal["vectorized", "low_mem"] = "vectorized",
        ignore_nans: bool = False,
        eps: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        if beta <= 0:
            msg = "beta must be > 0"
            raise ValueError(msg)
        if p_norm <= 0:
            msg = "p_norm must be > 0"
            raise ValueError(msg)

        self.fair = fair
        self.p_norm = float(p_norm)
        self.beta = float(beta)
        self.implementation = implementation
        self.eps = float(eps)

        self._impl = {
            "vectorized": self._score_vectorized,
            "low_mem": self._score_low_mem,
        }

    def _apply_vector_scalers(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        scaler_indices: SubsetIndices,
        without_scalers: list[str] | list[int] | None,
        grid_shard_slice: slice | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply VAR-including scalers inside the distance metric.

        We convert the loss-scale tensor to a component-scale tensor by taking ^(1/p_norm),
        so that for p-norm metrics it behaves like weighting inside the norm.
        """
        var_scalers, _ = _split_scalers(self.scaler, without_scalers=without_scalers)
        if len(var_scalers) == 0:
            return pred, target

        w = var_scalers.get_scaler(pred.ndim).to(pred.device)
        w = _slice_grid_if_needed(w, grid_shard_slice)
        if scaler_indices is not None:
            w = w[scaler_indices]

        # convert from "loss scaling" to "component scaling" inside p-norm
        if self.eps > 0:
            w = torch.clamp(w, min=self.eps)
        comp = w.pow(1.0 / self.p_norm)
        return pred * comp, target * comp

    def _apply_nonvar_scalers(
        self,
        out: torch.Tensor,
        *,
        without_scalers: list[str] | list[int] | None,
        grid_shard_slice: slice | None,
    ) -> torch.Tensor:
        """Apply scalers that do not include VARIABLE (typically GRID/node weights)."""
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

    def _score_vectorized(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return per-grid multivariate score (no GRID scaling), shape (bs, grid)."""
        _bs, m, _grid, _ = pred.shape
        n = target.shape[1]

        # E||X - Y||^beta
        diffs_xy = target.unsqueeze(1) - pred.unsqueeze(2)  # (bs, m, n, grid, var)
        dist_xy = torch.linalg.norm(diffs_xy, ord=self.p_norm, dim=-1).pow(self.beta)
        term_xy = dist_xy.mean(dim=(TensorDim.ENSEMBLE_DIM.value, 2))  # mean over m,n -> (bs, grid)

        if m == 1:
            # no spread term
            return term_xy

        # -0.5 E||X - X'||^beta
        diffs_xx = pred.unsqueeze(1) - pred.unsqueeze(2)  # (bs, m, m, grid, var)
        dist_xx = torch.linalg.norm(diffs_xx, ord=self.p_norm, dim=-1).pow(self.beta)
        coef_x = (-0.5 / (m * (m - 1))) if self.fair else (-0.5 / (m * m))
        term_x = dist_xx.sum(dim=(1, 2)) * coef_x  # (bs, grid)

        term_y = 0.0
        if n > 1:
            diffs_yy = target.unsqueeze(1) - target.unsqueeze(2)  # (bs, n, n, grid, var)
            dist_yy = torch.linalg.norm(diffs_yy, ord=self.p_norm, dim=-1).pow(self.beta)
            coef_y = (-0.5 / (n * (n - 1))) if self.fair else (-0.5 / (n * n))
            term_y = dist_yy.sum(dim=(1, 2)) * coef_y

        return term_xy + term_x + term_y

    def _score_low_mem(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Lower-memory variant (still vectorized over grid/vars)."""
        bs, m, grid, _ = pred.shape
        n = target.shape[1]

        # E||X - Y||^beta
        term_xy = 0.0
        for i in range(m):
            diffs = target - pred[:, i : i + 1]
            dist = torch.linalg.norm(diffs, ord=self.p_norm, dim=-1).pow(self.beta)  # (bs, n, grid)
            term_xy = term_xy + dist.mean(dim=1)  # accumulate (bs, grid)
        term_xy = term_xy / m

        if m == 1:
            return term_xy

        coef_x = (-0.5 / (m * (m - 1))) if self.fair else (-0.5 / (m * m))
        term_x = pred.new_zeros((bs, grid))
        for i in range(m):
            diffs = pred[:, i : i + 1] - pred[:, i + 1 :]  # (bs, m-i-1, grid, var)
            if diffs.numel() == 0:
                continue
            dist = torch.linalg.norm(diffs, ord=self.p_norm, dim=-1).pow(self.beta)  # (bs, m-i-1, grid)
            term_x = term_x + dist.sum(dim=1)
        term_x = term_x * coef_x * 2.0  # account for upper triangle only

        term_y = 0.0
        if n > 1:
            coef_y = (-0.5 / (n * (n - 1))) if self.fair else (-0.5 / (n * n))
            term_y_t = target.new_zeros((bs, grid))
            for i in range(n):
                diffs = target[:, i : i + 1] - target[:, i + 1 :]
                if diffs.numel() == 0:
                    continue
                dist = torch.linalg.norm(diffs, ord=self.p_norm, dim=-1).pow(self.beta)
                term_y_t = term_y_t + dist.sum(dim=1)
            term_y = term_y_t * coef_y * 2.0

        return term_xy + term_x + term_y

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: SubsetIndices = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **_: Any,
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None

        if target.ndim == 3:
            target = target.unsqueeze(TensorDim.ENSEMBLE_DIM.value)  # (bs, 1, grid, var)
        elif target.ndim != 4:
            msg = f"target must have ndim 3 or 4, got {target.ndim}"
            raise ValueError(msg)

        # Subset variables early (mirrors how BaseLoss.scale uses scaler_indices for slicing)
        if scaler_indices is not None:
            pred = pred[scaler_indices]
            target = target[scaler_indices]

        pred_s, target_s = self._apply_vector_scalers(
            pred,
            target,
            scaler_indices=None,  # already sliced above
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )

        score_grid = self._impl[self.implementation](pred_s, target_s)  # (bs, grid)
        out = score_grid.unsqueeze(TensorDim.ENSEMBLE_DIM.value).unsqueeze(TensorDim.VARIABLE.value)  # (bs, 1, grid, 1)

        out = self._apply_nonvar_scalers(out, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(out, squash=squash, squash_mode="sum", group=group if is_sharded else None)

    @cached_property
    def name(self) -> str:
        fair_str = "f" if self.fair else ""
        return f"{fair_str}mkcrps_p{self.p_norm:g}_b{self.beta:g}"


class GroupedMultivariateKernelCRPS(MultivariateKernelCRPS):
    """Grouped multivariate kCRPS with feature grouping.

    Supported patch/group methods:
    - group_by_variable
    - group_by_pressurelevel
    """

    _options_feature_patch_method = ("group_by_variable", "group_by_pressurelevel")

    def __init__(
        self,
        *,
        patch_method: str,
        data_indices_model_output: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if patch_method not in self._options_feature_patch_method:
            msg = f"patch_method {patch_method!r} not recognized, options are {self._options_feature_patch_method}"
            raise ValueError(msg)

        self.patch_method = patch_method
        self._patch_set: dict[str, torch.Tensor] | None = None

        if data_indices_model_output is not None:
            name_to_index = getattr(data_indices_model_output, "name_to_index", None)
            if name_to_index is None and isinstance(data_indices_model_output, dict):
                name_to_index = data_indices_model_output.get("name_to_index")
            if name_to_index is not None:
                self._patch_set = self._build_patch_set(name_to_index)

    def set_data_indices(self, data_indices: Any) -> None:
        # Prefer using IndexCollection-like objects from anemoi-core
        try:
            name_to_index = data_indices.model.output.name_to_index
        except Exception:  # noqa: BLE001 - permissive hook
            name_to_index = getattr(data_indices, "name_to_index", None)
        if name_to_index is None:
            msg = "Could not resolve `name_to_index` from data_indices for grouped loss"
            raise ValueError(msg)
        self._patch_set = self._build_patch_set(name_to_index)

    @staticmethod
    def _parse_var_and_level(name: str) -> tuple[str, str | None]:
        # Prefer `var_500` style
        if "_" in name:
            var, last = name.rsplit("_", 1)
            if last.isdigit():
                return var, last
        # Fallback: split embedded digits (e.g. 10u, v10, t25)
        digits = "".join(ch for ch in name if ch.isdigit())
        var = re.sub(r"\d+", "", name).strip("_")
        return var, digits or None

    def _build_patch_set(self, name_to_index: dict[str, int]) -> dict[str, torch.Tensor]:
        if self.patch_method == "group_by_variable":
            groups: dict[str, list[int]] = {}
            for name, idx in name_to_index.items():
                var, _ = self._parse_var_and_level(name)
                groups.setdefault(var, []).append(idx)
            return {k: torch.as_tensor(v, dtype=torch.long) for k, v in sorted(groups.items(), key=lambda kv: kv[0])}

        if self.patch_method == "group_by_pressurelevel":
            groups = {}
            for name, idx in name_to_index.items():
                _var, level = self._parse_var_and_level(name)
                key = (
                    name if level is None or level == "" else ("sfc" if level.isdigit() and int(level) < 50 else level)
                )
                groups.setdefault(key, []).append(idx)
            return {k: torch.as_tensor(v, dtype=torch.long) for k, v in sorted(groups.items(), key=lambda kv: kv[0])}

        msg = "unreachable"
        raise RuntimeError(msg)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: SubsetIndices = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if self._patch_set is None:
            msg = (
                "GroupedMultivariateKernelCRPS requires data indices to build patch sets. "
                "Ensure training calls `set_data_indices`, or pass `data_indices_model_output`."
            )
            raise ValueError(msg)

        # If a subset of variables is requested, filter each patch accordingly
        base_indices = None
        if scaler_indices is not None:
            # expected format is `[..., indices]`
            base_indices = scaler_indices[-1]
            base_indices = torch.as_tensor(base_indices, dtype=torch.long, device=pred.device)

        # compute total per-grid score (bs,1,grid,1) without GRID-only scaling, then scale once
        if target.ndim == 3:
            target = target.unsqueeze(TensorDim.ENSEMBLE_DIM.value)

        total_grid = pred.new_zeros((pred.shape[0], pred.shape[TensorDim.GRID.value]))
        for idxs in self._patch_set.values():
            idxs = idxs.to(pred.device)
            if base_indices is not None:
                # intersect patch indices with requested indices
                mask = torch.isin(idxs, base_indices)
                idxs = idxs[mask]
                if idxs.numel() == 0:
                    continue

            sub = (..., idxs)
            pred_g = pred[sub]
            target_g = target[sub]

            pred_s, target_s = self._apply_vector_scalers(
                pred_g,
                target_g,
                scaler_indices=None,
                without_scalers=without_scalers,
                grid_shard_slice=grid_shard_slice,
            )
            total_grid = total_grid + self._impl[self.implementation](pred_s, target_s)

        out = total_grid.unsqueeze(TensorDim.ENSEMBLE_DIM.value).unsqueeze(TensorDim.VARIABLE.value)  # (bs,1,grid,1)
        out = self._apply_nonvar_scalers(out, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        is_sharded = grid_shard_slice is not None
        return self.reduce(out, squash=squash, squash_mode="sum", group=group if is_sharded else None)

    @cached_property
    def name(self) -> str:
        fair_str = "f" if self.fair else ""
        return f"{fair_str}gmkcrps_{self.patch_method}_p{self.p_norm:g}_b{self.beta:g}"
