# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


def _gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x

    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d.div_(kernel_1d.sum())
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, 2 * radius + 1, 2 * radius + 1).expand(x.shape[1], 1, -1, -1)
    padded = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel, groups=x.shape[1])


def _shift_with_border(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    _, _, height, width = x.shape
    pad_left = max(-dx, 0)
    pad_right = max(dx, 0)
    pad_top = max(-dy, 0)
    pad_bottom = max(dy, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    x0 = pad_right
    y0 = pad_bottom
    return padded[..., y0 : y0 + height, x0 : x0 + width]


class OpticalFlowConsistencyLoss(BaseLoss):
    """Auxiliary loss that regularizes predictions toward a coarse torch-native optical-flow rollout baseline."""

    name: str = "optical_flow_consistency"

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        *,
        input_variable_index: int = 0,
        patch_size: int = 21,
        search_radius: int = 6,
        seed_spacing: int = 24,
        max_seeds: int = 192,
        rain_threshold: float = 0.1,
        filter_k: int = 6,
        filter_z_thresh: float = 2.5,
        range_fallback: float = 12.0,
        kriging_nugget: float = 0.05,
        downsample: int = 4,
        preprocess_sigma: float = 1.0,
        delta: float = 0.2,
        mask_threshold: float | None = 0.05,
        loss_type: Literal["huber", "l1"] = "huber",
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(ignore_nans=ignore_nans)
        if x_dim <= 0 or y_dim <= 0:
            raise ValueError("x_dim and y_dim must be positive")
        if patch_size < 3 or patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer >= 3")
        if search_radius < 1:
            raise ValueError("search_radius must be >= 1")

        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.input_variable_index = int(input_variable_index)
        self.patch_size = int(patch_size)
        self.search_radius = int(search_radius)
        self.seed_spacing = int(seed_spacing)
        self.max_seeds = int(max_seeds)
        self.rain_threshold = float(rain_threshold)
        self.filter_k = int(filter_k)
        self.filter_z_thresh = float(filter_z_thresh)
        self.range_fallback = float(range_fallback)
        self.kriging_nugget = float(kriging_nugget)
        self.downsample = int(downsample)
        self.preprocess_sigma = float(preprocess_sigma)
        self.delta = float(delta)
        self.mask_threshold = mask_threshold if mask_threshold is None else float(mask_threshold)
        self.loss_type = str(loss_type)
        self.supports_sharding = False

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.y_dim),
            torch.linspace(-1.0, 1.0, self.x_dim),
            indexing="ij",
        )
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        self.register_buffer("_base_grid", base_grid, persistent=False)

    @property
    def name(self) -> str:
        return "optical_flow_consistency"

    def _working_size(self) -> tuple[int, int]:
        return max(1, self.y_dim // self.downsample), max(1, self.x_dim // self.downsample)

    def _prepare_working_frame(self, frame: torch.Tensor) -> torch.Tensor:
        work = frame.to(dtype=torch.float32).clone()
        work.clamp_min_(0.0)
        work.log1p_()
        if self.downsample > 1:
            work = F.interpolate(work, size=self._working_size(), mode="bilinear", align_corners=False)
        if self.preprocess_sigma > 0:
            work = _gaussian_blur2d(work, self.preprocess_sigma)
        return work

    def _estimate_dense_flow(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prev_work = self._prepare_working_frame(prev_frame)
        curr_work = self._prepare_working_frame(curr_frame)
        work_height, work_width = prev_work.shape[-2:]

        curr_rain = curr_frame.to(dtype=torch.float32)
        if curr_rain.shape[-2:] != (work_height, work_width):
            curr_rain = F.interpolate(curr_rain, size=(work_height, work_width), mode="bilinear", align_corners=False)
        rain_mask = curr_rain.ge(self.rain_threshold)

        costs: list[torch.Tensor] = []
        dx_values: list[int] = []
        dy_values: list[int] = []
        radius = self.patch_size // 2

        for dy in range(-self.search_radius, self.search_radius + 1):
            for dx in range(-self.search_radius, self.search_radius + 1):
                shifted = _shift_with_border(curr_work, dx=dx, dy=dy)
                cost = prev_work - shifted
                cost.square_()
                if radius > 0:
                    cost = F.avg_pool2d(cost, kernel_size=self.patch_size, stride=1, padding=radius)
                costs.append(cost)
                dx_values.append(dx)
                dy_values.append(dy)

        cost_volume = torch.cat(costs, dim=1)
        best_indices = cost_volume.argmin(dim=1)
        dx_lookup = torch.tensor(dx_values, device=cost_volume.device, dtype=prev_work.dtype)
        dy_lookup = torch.tensor(dy_values, device=cost_volume.device, dtype=prev_work.dtype)
        flow_x_small = dx_lookup[best_indices].unsqueeze(1)
        flow_y_small = dy_lookup[best_indices].unsqueeze(1)

        if rain_mask.shape != flow_x_small.shape:
            rain_mask = rain_mask.expand_as(flow_x_small)
        flow_x_small.mul_(rain_mask)
        flow_y_small.mul_(rain_mask)

        sample_stride = max(1, self.seed_spacing // max(self.downsample, 1))
        sample_points = math.ceil(work_height / sample_stride) * math.ceil(work_width / sample_stride)
        if sample_points > self.max_seeds:
            stride_scale = int(math.ceil(math.sqrt(sample_points / self.max_seeds)))
            sample_stride *= stride_scale

        if sample_stride > 1:
            mask_float = rain_mask.to(dtype=flow_x_small.dtype)
            pooled_mask = F.avg_pool2d(mask_float, kernel_size=sample_stride, stride=sample_stride, ceil_mode=True)
            pooled_mask.clamp_min_(1.0e-6)

            flow_xy = torch.cat((flow_x_small, flow_y_small), dim=1)
            flow_xy.mul_(mask_float)
            flow_xy = F.avg_pool2d(flow_xy, kernel_size=sample_stride, stride=sample_stride, ceil_mode=True)
            flow_xy.div_(pooled_mask)
            flow_xy = F.interpolate(flow_xy, size=(work_height, work_width), mode="bilinear", align_corners=False)
            flow_x_small = flow_xy[:, 0:1]
            flow_y_small = flow_xy[:, 1:2]

        if self.preprocess_sigma > 0:
            flow_x_small = _gaussian_blur2d(flow_x_small, self.preprocess_sigma)
            flow_y_small = _gaussian_blur2d(flow_y_small, self.preprocess_sigma)
            flow_x_small.mul_(rain_mask)
            flow_y_small.mul_(rain_mask)

        flow_x = F.interpolate(flow_x_small, size=(self.y_dim, self.x_dim), mode="bilinear", align_corners=False)
        flow_y = F.interpolate(flow_y_small, size=(self.y_dim, self.x_dim), mode="bilinear", align_corners=False)
        flow_x.mul_(self.x_dim / work_width)
        flow_y.mul_(self.y_dim / work_height)
        return flow_x, flow_y

    def _advect_torch(
        self,
        field: torch.Tensor,
        flow_x: torch.Tensor,
        flow_y: torch.Tensor,
        step_scale: float,
    ) -> torch.Tensor:
        batch_size = field.shape[0]
        base_grid = self._base_grid.to(device=field.device, dtype=field.dtype)
        grid = torch.empty((batch_size, self.y_dim, self.x_dim, 2), device=field.device, dtype=field.dtype)

        x_scale = 0.0 if self.x_dim <= 1 else 2.0 * step_scale / (self.x_dim - 1)
        y_scale = 0.0 if self.y_dim <= 1 else 2.0 * step_scale / (self.y_dim - 1)
        grid[..., 0] = base_grid[..., 0] - flow_x[:, 0].mul(x_scale)
        grid[..., 1] = base_grid[..., 1] - flow_y[:, 0].mul(y_scale)

        advected = F.grid_sample(
            field,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return advected.clamp_min_(0.0)

    def _teacher_rollout(self, input_context: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if input_context.ndim != 5:
            raise ValueError("input_context must have shape (batch, time, ensemble, grid, vars)")
        if input_context.shape[1] < 2:
            raise ValueError("input_context must include at least two timesteps for optical-flow estimation")
        if pred.shape[-2] != self.x_dim * self.y_dim:
            raise ValueError(
                f"Expected grid size {self.x_dim * self.y_dim}, got {pred.shape[-2]} for optical-flow loss"
            )
        if pred.shape[-1] != 1:
            raise ValueError("OpticalFlowConsistencyLoss expects a single output variable; wrap it with filtering if needed")
        if input_context.shape[-1] <= self.input_variable_index:
            raise ValueError("input_variable_index exceeds available input channels")

        prev_frame = input_context[:, -2, 0, :, self.input_variable_index].reshape(-1, 1, self.y_dim, self.x_dim)
        curr_frame = input_context[:, -1, 0, :, self.input_variable_index].reshape(-1, 1, self.y_dim, self.x_dim)
        prev_frame = prev_frame.to(device=pred.device, dtype=torch.float32)
        curr_frame = curr_frame.to(device=pred.device, dtype=torch.float32)

        try:
            flow_x, flow_y = self._estimate_dense_flow(prev_frame, curr_frame)
        except Exception as exc:  # pragma: no cover - robustness fallback
            LOGGER.warning("Falling back to persistence in optical-flow loss after flow estimation failure: %s", exc)
            flow_x = curr_frame.new_zeros((curr_frame.shape[0], 1, self.y_dim, self.x_dim))
            flow_y = curr_frame.new_zeros((curr_frame.shape[0], 1, self.y_dim, self.x_dim))

        teacher = pred.new_empty((pred.shape[0], pred.shape[1], 1, pred.shape[-2], 1))
        for lead_step in range(1, pred.shape[1] + 1):
            advected = self._advect_torch(curr_frame, flow_x, flow_y, step_scale=float(lead_step))
            teacher[:, lead_step - 1, 0, :, 0].copy_(advected.flatten(start_dim=1).to(dtype=pred.dtype))
        return teacher.expand(-1, -1, pred.shape[2], -1, -1)

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
        input_context: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_context is None:
            raise ValueError("OpticalFlowConsistencyLoss requires input_context to estimate flow")

        is_sharded = grid_shard_slice is not None
        with torch.no_grad():
            teacher = self._teacher_rollout(input_context, pred)

        pred_for_loss = pred
        if self.ignore_nans:
            nan_mask = torch.isnan(target)
            pred_for_loss = pred.masked_fill(nan_mask, 0.0)
            teacher.masked_fill_(nan_mask, 0.0)

        if self.loss_type == "l1":
            out = torch.abs(pred_for_loss - teacher)
        elif self.loss_type == "huber":
            out = F.huber_loss(pred_for_loss, teacher, reduction="none", delta=self.delta)
        else:
            raise ValueError(f"Unsupported optical-flow loss_type: {self.loss_type}")

        if self.mask_threshold is not None:
            mask = teacher.gt(self.mask_threshold)
            mask.logical_or_(target > self.mask_threshold)
            out.mul_(mask.to(dtype=out.dtype))

        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(
            out,
            squash,
            group=group if is_sharded else None,
            squash_mode=kwargs.get("squash_mode", "avg"),
        )
