# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import torch.nn.functional as F
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss
from anemoi.training.utils.enums import TensorDim


class SSIMLoss(BaseLoss):
    """Structural similarity loss on 2D gridded fields."""

    name: str = "ssim"

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1.0e-6,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
        if x_dim <= 0 or y_dim <= 0:
            raise ValueError(f"x_dim and y_dim must be positive, got {(x_dim, y_dim)}")
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.window_size = int(window_size)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.eps = float(eps)
        self.supports_sharding = False

    def _to_maps(self, x: torch.Tensor) -> torch.Tensor:
        expected_grid = self.x_dim * self.y_dim
        if x.shape[TensorDim.GRID] != expected_grid:
            raise ValueError(
                f"Expected flattened grid size {expected_grid} from x_dim*y_dim, got {x.shape[TensorDim.GRID]}"
            )
        return x.permute(0, 1, 2, 4, 3).reshape(-1, 1, self.y_dim, self.x_dim)

    def _from_maps(self, x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return x.reshape(
            like.shape[TensorDim.BATCH_SIZE],
            like.shape[TensorDim.TIME],
            like.shape[TensorDim.ENSEMBLE_DIM],
            like.shape[TensorDim.VARIABLE],
            self.y_dim * self.x_dim,
        ).permute(0, 1, 2, 4, 3)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if self.ignore_nans:
            nan_mask = torch.isnan(target)
            target = target.masked_fill(nan_mask, 0.0)
            pred = pred.masked_fill(nan_mask, 0.0)

        pred_maps = self._to_maps(pred)
        target_maps = self._to_maps(target)

        kernel = pred_maps.new_full((1, 1, self.window_size, self.window_size), 1.0 / (self.window_size**2))
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred_maps, kernel, padding=pad)
        mu_target = F.conv2d(target_maps, kernel, padding=pad)

        sigma_pred = F.conv2d(pred_maps * pred_maps, kernel, padding=pad) - mu_pred.square()
        sigma_target = F.conv2d(target_maps * target_maps, kernel, padding=pad) - mu_target.square()
        sigma_cross = F.conv2d(pred_maps * target_maps, kernel, padding=pad) - mu_pred * mu_target

        sigma_pred = torch.clamp(sigma_pred, min=0.0)
        sigma_target = torch.clamp(sigma_target, min=0.0)

        dynamic_max = torch.maximum(pred_maps, target_maps).amax(dim=(-2, -1), keepdim=True)
        dynamic_min = torch.minimum(pred_maps, target_maps).amin(dim=(-2, -1), keepdim=True)
        data_range = (dynamic_max - dynamic_min).clamp_min(self.eps)

        c1 = (self.k1 * data_range).square()
        c2 = (self.k2 * data_range).square()

        numerator = (2.0 * mu_pred * mu_target + c1) * (2.0 * sigma_cross + c2)
        denominator = (mu_pred.square() + mu_target.square() + c1) * (sigma_pred + sigma_target + c2)
        ssim_map = numerator / (denominator + self.eps)
        loss_map = torch.clamp((1.0 - ssim_map) * 0.5, min=0.0)

        out = self._from_maps(loss_map, pred)
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(out, squash, group=group if grid_shard_slice is not None else None)
