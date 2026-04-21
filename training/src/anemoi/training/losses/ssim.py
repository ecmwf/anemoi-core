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



def _dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    return F.max_pool2d(mask.to(dtype=torch.float32), kernel_size=kernel_size, stride=1, padding=kernel_size // 2).gt(0.0)


def _masked_reduce_scaled(
    loss: BaseLoss,
    out_scaled: torch.Tensor,
    mask_scaled: torch.Tensor,
    squash: bool,
    *,
    squash_mode: str = "avg",
    group: ProcessGroup | None = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    if squash:
        if squash_mode == "avg":
            out_scaled = loss.avg_function(out_scaled, dim=TensorDim.VARIABLE, keepdim=True)
            mask_scaled = loss.avg_function(mask_scaled, dim=TensorDim.VARIABLE, keepdim=True)
        elif squash_mode == "sum":
            out_scaled = loss.sum_function(out_scaled, dim=TensorDim.VARIABLE, keepdim=True)
            mask_scaled = loss.sum_function(mask_scaled, dim=TensorDim.VARIABLE, keepdim=True)
        else:
            raise ValueError(f"Invalid squash_mode '{squash_mode}'. Supported modes are: 'avg', 'sum'")

    numerator = loss.sum_function(
        out_scaled,
        dim=(TensorDim.TIME, TensorDim.GRID),
        keepdim=True,
    )
    denominator = loss.sum_function(
        mask_scaled,
        dim=(TensorDim.TIME, TensorDim.GRID),
        keepdim=True,
    ).clamp_min(eps)
    reduced = numerator / denominator
    out = loss.avg_function(
        reduced,
        dim=(TensorDim.BATCH_SIZE, TensorDim.TIME, TensorDim.ENSEMBLE_DIM),
    ).squeeze()
    return out if group is None else loss.reduce(out, squash=squash, group=group)


class MaskedLogSSIMLoss(SSIMLoss):
    """SSIM on log-rain maps, restricted to rainy structure instead of the dry background."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        window_size: int = 5,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1.0e-6,
        rain_threshold: float = 0.1,
        mask_dilation: int | None = None,
        include_pred_mask: bool = False,
        log_scale: float = 1.0,
        normalise_by_mask: bool = True,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(
            x_dim=x_dim,
            y_dim=y_dim,
            window_size=window_size,
            k1=k1,
            k2=k2,
            eps=eps,
            ignore_nans=ignore_nans,
        )
        if rain_threshold < 0:
            raise ValueError(f"rain_threshold must be non-negative, got {rain_threshold}")
        if mask_dilation is None:
            mask_dilation = 3
        if mask_dilation <= 0 or mask_dilation % 2 == 0:
            raise ValueError(f"mask_dilation must be a positive odd integer, got {mask_dilation}")
        if log_scale <= 0:
            raise ValueError(f"log_scale must be positive, got {log_scale}")

        self.rain_threshold = float(rain_threshold)
        self.mask_dilation = int(mask_dilation)
        self.include_pred_mask = bool(include_pred_mask)
        self.log_scale = float(log_scale)
        self.normalise_by_mask = bool(normalise_by_mask)

    @property
    def name(self) -> str:
        return "ssim"

    def _rain_mask(self, pred_maps: torch.Tensor, target_maps: torch.Tensor) -> torch.Tensor:
        mask = target_maps.gt(self.rain_threshold)
        if self.include_pred_mask:
            mask.logical_or_(pred_maps.gt(self.rain_threshold))
        return _dilate_mask(mask, self.mask_dilation)

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
        is_sharded = grid_shard_slice is not None

        valid = None
        if self.ignore_nans:
            valid = ~torch.isnan(target)
            target = target.masked_fill(~valid, 0.0)
            pred = pred.masked_fill(~valid, 0.0)

        raw_pred_maps = self._to_maps(pred.clamp_min(0.0))
        raw_target_maps = self._to_maps(target.clamp_min(0.0))
        pred_maps = torch.log1p(raw_pred_maps * self.log_scale)
        target_maps = torch.log1p(raw_target_maps * self.log_scale)

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

        mask_maps = self._rain_mask(raw_pred_maps, raw_target_maps)
        if valid is not None:
            valid_maps = self._to_maps(valid.to(dtype=loss_map.dtype)).gt(0.5)
            loss_map.mul_(valid_maps.to(dtype=loss_map.dtype))
            mask_maps.logical_and_(valid_maps)

        out = self._from_maps(loss_map, pred)
        mask = self._from_maps(mask_maps.to(dtype=out.dtype), pred)

        if self.normalise_by_mask:
            out_scaled = self.scale(
                out * mask,
                scaler_indices,
                without_scalers=without_scalers,
                grid_shard_slice=grid_shard_slice,
            )
            mask_scaled = self.scale(
                mask,
                scaler_indices,
                without_scalers=without_scalers,
                grid_shard_slice=grid_shard_slice,
            )
            return _masked_reduce_scaled(
                self,
                out_scaled,
                mask_scaled,
                squash,
                squash_mode="avg",
                group=group if is_sharded else None,
                eps=self.eps,
            )

        out = self.scale(out * mask, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(out, squash, group=group if is_sharded else None)
