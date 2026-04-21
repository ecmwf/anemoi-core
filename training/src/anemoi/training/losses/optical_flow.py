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
from anemoi.training.utils.enums import TensorDim

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

def _dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    return F.max_pool2d(
        mask.to(dtype=torch.float32),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).gt(0.0)

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
    return out if group is None else reduce_tensor(out, group)


class SoftWetMaskAdvectiveConsistencyLoss(OpticalFlowConsistencyLoss):
    """
    Short-lead auxiliary term encouraging predicted rainy support to move
    consistently with input-derived optical flow.
    """

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
        ignore_nans: bool = False,
        wet_threshold: float = 0.1,
        wet_temperature: float = 0.05,
        teacher_support_threshold: float = 0.20,
        teacher_mask_dilation: int = 3,
        loss_type: Literal["huber", "l1", "bce"] = "huber",
        delta: float = 0.10,
        lead_decay: float = 0.7,
        max_reg_leads: int = 6,
        normalise_by_mask: bool = True,
        include_target_mask: bool = False,
        flow_history: int = 4,
        history_decay: float = 0.8,
    ) -> None:
        super().__init__(
            x_dim=x_dim,
            y_dim=y_dim,
            input_variable_index=input_variable_index,
            patch_size=patch_size,
            search_radius=search_radius,
            seed_spacing=seed_spacing,
            max_seeds=max_seeds,
            rain_threshold=rain_threshold,
            filter_k=filter_k,
            filter_z_thresh=filter_z_thresh,
            range_fallback=range_fallback,
            kriging_nugget=kriging_nugget,
            downsample=downsample,
            preprocess_sigma=preprocess_sigma,
            delta=delta,
            mask_threshold=None,
            loss_type="huber",
            ignore_nans=ignore_nans,
        )
        if wet_temperature <= 0:
            raise ValueError(f"wet_temperature must be positive, got {wet_temperature}")
        if not 0 <= teacher_support_threshold <= 1:
            raise ValueError(
                f"teacher_support_threshold must be in [0, 1], got {teacher_support_threshold}"
            )
        if teacher_mask_dilation <= 0 or teacher_mask_dilation % 2 == 0:
            raise ValueError(
                f"teacher_mask_dilation must be a positive odd integer, got {teacher_mask_dilation}"
            )
        if lead_decay <= 0:
            raise ValueError(f"lead_decay must be positive, got {lead_decay}")
        if max_reg_leads <= 0:
            raise ValueError(f"max_reg_leads must be positive, got {max_reg_leads}")
        if flow_history < 2:
            raise ValueError(f"flow_history must be >= 2, got {flow_history}")
        if history_decay <= 0:
            raise ValueError(f"history_decay must be positive, got {history_decay}")

        self.wet_threshold = float(wet_threshold)
        self.wet_temperature = float(wet_temperature)
        self.teacher_support_threshold = float(teacher_support_threshold)
        self.teacher_mask_dilation = int(teacher_mask_dilation)
        self.aux_loss_type = str(loss_type)
        self.delta = float(delta)
        self.lead_decay = float(lead_decay)
        self.max_reg_leads = int(max_reg_leads)
        self.normalise_by_mask = bool(normalise_by_mask)
        self.include_target_mask = bool(include_target_mask)
        self.flow_history = int(flow_history)
        self.history_decay = float(history_decay)

    @property
    def name(self) -> str:
        return "wetmask_advective_consistency"

    def _maps_sequence_to_like(self, seq_maps: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        batch_size, _, ensemble_size, grid_size, _ = like.shape
        seq_maps = seq_maps.reshape(batch_size, ensemble_size, like.shape[1], self.y_dim * self.x_dim)
        return seq_maps.permute(0, 2, 1, 3).unsqueeze(-1)

    def _soft_wet_mask(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32).clamp_min_(0.0)
        return torch.sigmoid((x - self.wet_threshold) / self.wet_temperature)

    def _lead_weights(self, n_leads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        lead_ids = torch.arange(n_leads, device=device, dtype=dtype)
        weights = torch.pow(torch.tensor(self.lead_decay, device=device, dtype=dtype), lead_ids)
        if self.max_reg_leads < n_leads:
            weights[self.max_reg_leads:] = 0.0
        return weights

    def _estimate_history_flow_from_inputs(
        self,
        input_context: torch.Tensor,
        pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate a mean per-step flow from the last `flow_history` input frames.

        Returns:
            mean_flow_x: (B*E, 1, Y, X)
            mean_flow_y: (B*E, 1, Y, X)
            last_frame:   (B*E, 1, Y, X)
        """
        if input_context.ndim != 5:
            raise ValueError("input_context must have shape (batch, time, ensemble, grid, vars)")
        if input_context.shape[-1] <= self.input_variable_index:
            raise ValueError("input_variable_index exceeds available input channels")

        batch_size, _, ensemble_size, _, _ = input_context.shape

        n_hist = min(self.flow_history, input_context.shape[1])
        if n_hist < 2:
            raise ValueError("Need at least two input frames to estimate history flow")

        hist = input_context[:, -n_hist:, :, :, self.input_variable_index]
        hist = hist.reshape(batch_size, n_hist, ensemble_size, self.y_dim, self.x_dim)
        hist = hist.permute(0, 2, 1, 3, 4).reshape(batch_size * ensemble_size, n_hist, 1, self.y_dim, self.x_dim)
        hist = hist.to(device=pred.device, dtype=torch.float32)

        flow_xs = []
        flow_ys = []
        weights = []

        for pair_idx in range(1, n_hist):
            prev_frame = hist[:, pair_idx - 1]
            curr_frame = hist[:, pair_idx]
            try:
                flow_x, flow_y = self._estimate_dense_flow(prev_frame, curr_frame)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning(
                    "Skipping history flow pair %d/%d after flow estimation failure: %s",
                    pair_idx,
                    n_hist - 1,
                    exc,
                )
                continue

            # More recent pairs get larger weight.
            age = (n_hist - 1) - pair_idx
            weight = self.history_decay ** age

            flow_xs.append(flow_x)
            flow_ys.append(flow_y)
            weights.append(weight)

        if not flow_xs:
            last_frame = hist[:, -1]
            zeros = last_frame.new_zeros((last_frame.shape[0], 1, self.y_dim, self.x_dim))
            return zeros, zeros.clone(), last_frame

        weight_tensor = hist.new_tensor(weights).view(-1, 1, 1, 1, 1)
        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1.0e-6)

        mean_flow_x = (torch.stack(flow_xs, dim=0) * weight_tensor).sum(dim=0)
        mean_flow_y = (torch.stack(flow_ys, dim=0) * weight_tensor).sum(dim=0)
        last_frame = hist[:, -1]

        return mean_flow_x, mean_flow_y, last_frame

    def _teacher_rollout_masks(
        self,
        input_context: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build an advected wet-mask teacher from the last `flow_history` input frames.
        Output shape: (B*E, T, 1, Y, X)
        """
        if pred.shape[-2] != self.x_dim * self.y_dim:
            raise ValueError(
                f"Expected grid size {self.x_dim * self.y_dim}, got {pred.shape[-2]}"
            )

        _, lead_steps, _, _, _ = pred.shape

        flow_x, flow_y, last_frame = self._estimate_history_flow_from_inputs(input_context, pred)
        last_mask = self._soft_wet_mask(last_frame)

        teacher_masks = []
        for lead_step in range(1, lead_steps + 1):
            advected = self._advect_torch(last_mask, flow_x, flow_y, step_scale=float(lead_step))
            teacher_masks.append(advected.clamp_(0.0, 1.0))

        return torch.stack(teacher_masks, dim=1)  # (B*E, T, 1, Y, X)

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
            raise ValueError("SoftWetMaskAdvectiveConsistencyLoss requires input_context")
        if pred.shape[-1] != 1 or target.shape[-1] != 1:
            raise ValueError("SoftWetMaskAdvectiveConsistencyLoss expects a single output variable")

        is_sharded = grid_shard_slice is not None
        batch_size, lead_steps, ensemble_size, _, _ = pred.shape

        with torch.no_grad():
            teacher_masks = self._teacher_rollout_masks(input_context, pred)

        pred_maps = pred[..., 0].reshape(batch_size * ensemble_size, lead_steps, 1, self.y_dim, self.x_dim)
        pred_masks = self._soft_wet_mask(pred_maps)

        target_maps = target[..., 0].reshape(batch_size * ensemble_size, lead_steps, 1, self.y_dim, self.x_dim)
        valid_mask = None
        if self.ignore_nans:
            valid_mask = ~torch.isnan(target_maps)
            target_maps = target_maps.masked_fill(~valid_mask, 0.0)
            pred_masks = pred_masks.masked_fill(~valid_mask, 0.0)
            teacher_masks = teacher_masks.masked_fill(~valid_mask, 0.0)

        support = teacher_masks.gt(self.teacher_support_threshold)
        support = _dilate_mask(
            support.reshape(batch_size * ensemble_size * lead_steps, 1, self.y_dim, self.x_dim),
            self.teacher_mask_dilation,
        ).reshape(batch_size * ensemble_size, lead_steps, 1, self.y_dim, self.x_dim)

        if self.include_target_mask:
            target_support = self._soft_wet_mask(target_maps).gt(self.teacher_support_threshold)
            support.logical_or_(target_support)

        if valid_mask is not None:
            support.logical_and_(valid_mask)

        if self.aux_loss_type == "l1":
            loss_maps = torch.abs(pred_masks - teacher_masks)
        elif self.aux_loss_type == "bce":
            teacher_prob = teacher_masks.clamp(1.0e-4, 1.0 - 1.0e-4)
            pred_prob = pred_masks.clamp(1.0e-4, 1.0 - 1.0e-4)
            loss_maps = F.binary_cross_entropy(pred_prob, teacher_prob, reduction="none")
        elif self.aux_loss_type == "huber":
            loss_maps = F.huber_loss(pred_masks, teacher_masks, reduction="none", delta=self.delta)
        else:
            raise ValueError(f"Unsupported loss_type: {self.aux_loss_type}")

        lead_weights = self._lead_weights(lead_steps, device=loss_maps.device, dtype=loss_maps.dtype)
        loss_maps = loss_maps * lead_weights.view(1, lead_steps, 1, 1, 1)
        loss_maps = loss_maps * support.to(dtype=loss_maps.dtype)

        out = self._maps_sequence_to_like(loss_maps, pred)
        mask = self._maps_sequence_to_like(
            support.to(dtype=loss_maps.dtype) * lead_weights.view(1, lead_steps, 1, 1, 1),
            pred,
        )

        squash_mode = kwargs.get("squash_mode", "avg")

        if self.normalise_by_mask:
            out_scaled = self.scale(
                out,
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
                squash_mode=squash_mode,
                group=group if is_sharded else None,
            )

        out = self.scale(
            out,
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        return self.reduce(
            out,
            squash,
            group=group if is_sharded else None,
            squash_mode=squash_mode,
        )