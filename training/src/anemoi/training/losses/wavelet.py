# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Regional, area/mask-aware wavelet AMSE."""

import math
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import torch
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.wavelet_transforms import DTCWT2D
from anemoi.models.layers.wavelet_transforms import WaveletPyramid
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.utils.compile import subset_tensor
from anemoi.training.utils.enums import TensorDim


class RegionalWaveletLoss(BaseLoss):
    """Band-summed DTCWT AMSE over oriented subbands plus low-pass MSE.

    Fields have shape ``[batch,time,ensemble,grid,variable]``; targets may have
    a singleton ensemble dimension or omit it. The selected grid must be an
    H-by-W rectangle flattened in row-major order.

    GRID-bearing scalers supply absolute spatial weights. Their product's sum
    is retained as a multiplier, while its relative weights are pooled onto each
    coefficient grid. With no spatial scalers the default is uniform 1/(H*W).
    Thus unit-sum node weights preserve the unweighted result, and configured
    regional totals are not normalized away. Mixed-dimension masks are supported.

    Power and cross terms are weighted and summed spatially before AMSE.
    Coefficient weighting is not sqrt-weighting of physical fields
    or exact wavelet-support masking. Zero-weight input values are excluded
    before transforming. ``ignore_nans`` additionally pools finite-pair validity
    through the pyramid and corrects by the valid input fraction.

    ``transform`` contains backend/filter options for :class:`DTCWT2D`; J is
    specified on the loss. Compilation through Anemoi's existing configuration
    compiles the tensor scoring kernel, not backend preparation or collectives.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] | list[int],
        J: int | None = None,  # noqa: N803
        *,
        transform: Mapping[str, Any] | None = None,
        scale_weights: list[float] | None = None,
        eps: float = 1e-8,
        ignore_nans: bool = False,
        subgrid: tuple[int, int | None] | None = None,
        separate_pred_target_xfm: bool | None = None,
        separate_pred_target_xfm_threshold: int = 1_000_000,
        xfm_batch_chunk_size: int | None = None,
    ) -> None:
        """Configure the regional grid, DTCWT, scale weights and chunking.

        If omitted, J is selected from the shortest grid dimension. Separate
        prediction/target transforms default to the configured grid-size threshold.
        """
        super().__init__(ignore_nans=ignore_nans)
        if len(grid_shape) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in grid_shape):
            msg = "grid_shape must contain two integers >= 2"
            raise ValueError(msg)
        self.grid_shape = tuple(grid_shape)
        self.n_gridpoints = math.prod(grid_shape)
        self.J = max(1, int(math.log2(min(grid_shape))) - 3) if J is None else J
        if not math.isfinite(eps) or eps <= 0:
            msg = "eps must be finite and positive"
            raise ValueError(msg)
        if scale_weights is not None and (
            len(scale_weights) != self.J or any(not math.isfinite(w) or w < 0 for w in scale_weights)
        ):
            msg = "scale_weights must contain J finite, nonnegative weights"
            raise ValueError(msg)
        if xfm_batch_chunk_size is not None and (
            isinstance(xfm_batch_chunk_size, bool)
            or not isinstance(xfm_batch_chunk_size, int)
            or xfm_batch_chunk_size < 1
        ):
            msg = "xfm_batch_chunk_size must be a positive integer or None"
            raise ValueError(msg)
        if separate_pred_target_xfm_threshold < 1:
            msg = "separate_pred_target_xfm_threshold must be positive"
            raise ValueError(msg)
        if subgrid is not None and (
            not isinstance(subgrid, Sequence)
            or len(subgrid) != 2
            or isinstance(subgrid[0], bool)
            or not isinstance(subgrid[0], int)
            or subgrid[0] < 0
            or (subgrid[1] is not None and (not isinstance(subgrid[1], int) or subgrid[1] <= subgrid[0]))
        ):
            msg = "subgrid must be a (nonnegative start, end or None) pair resolved before instantiation"
            raise ValueError(msg)
        self.subgrid = slice(*(subgrid or (0, None)))
        self.scale_weights = tuple(scale_weights) if scale_weights is not None else (1.0,) * self.J
        self.eps = eps
        self.xfm_batch_chunk_size = xfm_batch_chunk_size
        self.separate_pred_target_xfm = (
            self.n_gridpoints > separate_pred_target_xfm_threshold
            if separate_pred_target_xfm is None
            else separate_pred_target_xfm
        )
        self.transform = DTCWT2D(J=self.J, **dict(transform or {}))

    @property
    def name(self) -> str:
        """Return the identifier used to label this loss."""
        return "regional_wavelet"

    @property
    def needs_shard_layout_info(self) -> bool:
        """Request grid-shard metadata so regional fields can be reconstructed."""
        return True

    def compile_for_training(self, **options: Any) -> None:
        """Compile tensor scoring; real backend transforms intentionally stay eager."""
        self._score_pyramid = torch.compile(self._score_pyramid, **options)

    @staticmethod
    def _validate_spatial_scaler(dimension: int | tuple[int, ...], scaler: torch.Tensor) -> None:
        """Check layout axes and require finite, nonnegative GRID-bearing scalers."""
        dims = tuple(range(dimension, dimension + scaler.ndim)) if isinstance(dimension, int) else dimension
        if any(d < -5 or d >= 5 for d in dims):
            msg = "Wavelet scaler dimensions must index the five-dimensional input layout"
            raise ValueError(msg)
        dims = tuple(d % 5 for d in dims)
        if TensorDim.GRID in dims and (not torch.isfinite(scaler).all() or (scaler < 0).any()):
            msg = "Wavelet spatial scalers must be finite and nonnegative"
            raise ValueError(msg)

    def add_scaler(
        self,
        dimension: int | tuple[int, ...],
        scaler: torch.Tensor,
        *,
        name: str | None = None,
    ) -> None:
        """Validate and register a spatial or non-spatial loss scaler."""
        self._validate_spatial_scaler(dimension, scaler)
        self.scaler.add_scaler(dimension=dimension, scaler=scaler, name=name)

    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        """Validate a replacement scaler against its registered axes and update it."""
        dims = self.scaler.specified_dimensions[name]
        self._validate_spatial_scaler(dims, scaler)
        self.scaler.update_scaler(name=name, scaler=scaler, override=override)

    def _prepare_inputs(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        grid_shard_slice: slice | None,
        grid_shard_sizes: list[int] | None,
        group: ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int] | None]:
        """Validate fields, detach/broadcast targets and reconstruct sharded grids.

        Sharded inputs are transposed from grid shards to variable shards.
        Return their variable-shard sizes, or None for unsharded inputs.
        """
        if pred.ndim != 5 or target.ndim not in (4, 5) or not all((*pred.shape[:3], pred.shape[-1])):
            msg = "Wavelet inputs must have nonempty B,T,E,V dimensions in [B,T,E,G,V] (target may omit E)"
            raise ValueError(msg)
        if not pred.is_floating_point() or not target.is_floating_point():
            msg = "Wavelet inputs must be real floating-point tensors"
            raise TypeError(msg)
        if pred.device != target.device:
            msg = "Prediction and target must be on the same device"
            raise ValueError(msg)
        if target.ndim == 4:
            target = target.unsqueeze(2)
        if target.shape[:2] != pred.shape[:2] or target.shape[3:] != pred.shape[3:]:
            msg = "Prediction and target batch/time/grid/variable shapes must match"
            raise ValueError(msg)
        if target.shape[2] not in (1, pred.shape[2]):
            msg = "Target ensemble dimension must be singleton or match prediction"
            raise ValueError(msg)
        target = target.detach().expand_as(pred)
        if (grid_shard_slice is None) != (grid_shard_sizes is None):
            msg = "grid_shard_slice and grid_shard_sizes must be provided together"
            raise ValueError(msg)
        if grid_shard_sizes is None:
            return pred, target, None
        if group is None:
            msg = "Sharded wavelet inputs require a process group"
            raise ValueError(msg)
        sizes = get_shard_sizes(pred, TensorDim.VARIABLE, group)
        pred = all_to_all_transpose(pred, 4, sizes, 3, grid_shard_sizes, group)
        target = all_to_all_transpose(target, 4, sizes, 3, grid_shard_sizes, group)
        return pred, target, sizes

    @staticmethod
    def _reshard_scaler(
        name: str,
        values: torch.Tensor,
        full_grid_size: int,
        variable_shard_sizes: list[int],
        grid_shard_sizes: list[int],
        group: ProcessGroup,
    ) -> torch.Tensor:
        """Match a scaler to the complete-grid, local-variable field layout.

        Full-grid scalers only need variable slicing. Rank-local spatial scalers
        are gathered or transposed depending on whether they vary by variable.
        """
        rank = torch.distributed.get_rank(group)
        if values.shape[3] not in (1, full_grid_size):
            if values.shape[3] != grid_shard_sizes[rank]:
                msg = f"Spatial scaler {name!r} must contain the full grid or the rank-local grid shard"
                raise ValueError(msg)
            if values.shape[-1] == 1:
                values = gather_tensor(values, 3, grid_shard_sizes, group)
            else:
                if values.shape[-1] != sum(variable_shard_sizes):
                    msg = f"Scaler {name!r} must contain the full variable dimension"
                    raise ValueError(msg)
                # Gathering after slicing variables would mix different variables across ranks.
                return all_to_all_transpose(values, 4, variable_shard_sizes, 3, grid_shard_sizes, group)
        if values.shape[-1] != 1:
            if values.shape[-1] != sum(variable_shard_sizes):
                msg = f"Scaler {name!r} must contain the full variable dimension"
                raise ValueError(msg)
            start = sum(variable_shard_sizes[:rank])
            values = values[..., start : start + variable_shard_sizes[rank]]
        return values

    def _align_scaler(
        self,
        name: str,
        dims: tuple[int, ...],
        values: torch.Tensor,
        pred: torch.Tensor,
        variable_shard_sizes: list[int] | None,
        grid_shard_sizes: list[int] | None,
        group: ProcessGroup | None,
    ) -> torch.Tensor:
        """Place scaler axes in [B,T,E,G,V] order without expanding singleton axes.

        Move values to FP32 on the input device, align distributed shards,
        validate broadcasting and select the configured spatial subgrid.
        """
        if len(dims) != values.ndim or len(set(dims)) != len(dims):
            msg = f"Scaler {name!r} has incompatible dimensions {dims} and shape {values.shape}"
            raise ValueError(msg)
        shape = [1] * 5
        for d, n in zip(dims, values.shape, strict=True):
            shape[d] = n
        values = values.to(device=pred.device, dtype=torch.float32)
        values = values.permute(sorted(range(len(dims)), key=dims.__getitem__)).reshape(shape)
        if variable_shard_sizes is not None:
            if group is None or grid_shard_sizes is None:
                msg = "Aligning sharded scalers requires a process group and grid shard sizes"
                raise ValueError(msg)
            values = self._reshard_scaler(name, values, pred.shape[3], variable_shard_sizes, grid_shard_sizes, group)
        if any(values.shape[d] not in (1, pred.shape[d]) for d in (0, 1, 2, 4)):
            msg = f"Scaler {name!r} does not broadcast to the prediction batch/time/ensemble/variable dimensions"
            raise ValueError(msg)
        if TensorDim.GRID in dims:
            if values.shape[3] not in (1, pred.shape[3]):
                msg = f"Spatial scaler {name!r} does not match the input grid"
                raise ValueError(msg)
            if values.shape[3] != 1:
                values = values[:, :, :, self.subgrid]
        return values

    def _weights(
        self,
        pred: torch.Tensor,
        without_scalers: list[str] | list[int] | None,
        variable_shard_sizes: list[int] | None,
        grid_shard_sizes: list[int] | None,
        group: ProcessGroup | None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Separate active compact spatial scalers from non-spatial multipliers.

        Spatial products are deferred until chunk processing. An empty spatial
        tuple denotes implicit uniform weights with total weight one.
        """
        spatial = []
        non_spatial_weights = torch.ones((1, 1, 1, 1, 1), device=pred.device, dtype=torch.float32)
        for name, (dims, values) in self.scaler.tensors.items():
            dims = tuple(d % 5 for d in dims)
            if without_scalers and (name in without_scalers or any(d in without_scalers for d in dims)):
                continue
            values = self._align_scaler(name, dims, values, pred, variable_shard_sizes, grid_shard_sizes, group)
            if TensorDim.GRID in dims:
                spatial.append(values)
            else:
                non_spatial_weights = non_spatial_weights * values
        return tuple(spatial), non_spatial_weights

    @staticmethod
    def _pool_spatial_weights(relative_spatial_weights: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        """Pool relative weights to a coefficient grid and restore unit mean.

        Preserve compact singleton unit weights and leave all-zero maps zero.
        """
        if relative_spatial_weights.shape[-2:] == (1, 1):
            return relative_spatial_weights
        pooled = functional.adaptive_avg_pool2d(relative_spatial_weights, shape)
        mean = pooled.mean(dim=(-2, -1), keepdim=True)
        return pooled / torch.where(mean > 0, mean, torch.ones_like(mean))

    def _amse(self, pred_power: torch.Tensor, target_power: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        """Compute epsilon-regularized AMSE from band power and real cross terms."""
        a = torch.sqrt(pred_power + self.eps)
        b = torch.sqrt(target_power + self.eps)
        return (a - b).square() + 2 * torch.maximum(pred_power, target_power) * (1 - cross / (a * b + self.eps))

    def _score_pyramid(
        self,
        pred: WaveletPyramid,
        target: WaveletPyramid,
        relative_spatial_weights: torch.Tensor,
        valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Combine weighted subband AMSE and low-pass MSE into per-field scores.

        Pool relative weights and optional validity masks to coefficient grids.
        Normalize by grid size or valid count; the caller applies absolute
        spatial totals and non-spatial weights separately.
        """
        total = relative_spatial_weights.new_zeros(relative_spatial_weights.shape[0])
        mask = valid
        for scale_weight, wp, wt in zip(self.scale_weights, pred.highpasses, target.highpasses, strict=True):
            shape = (wp.shape[-3], wp.shape[-2])
            weight = self._pool_spatial_weights(relative_spatial_weights, shape)
            if mask is not None:
                mask = functional.adaptive_avg_pool2d(mask, shape)
                weight = weight * mask
            weight = weight.unsqueeze(2)
            pp = (wp.square().sum(dim=-1) * weight).sum(dim=(-2, -1))
            pt = (wt.square().sum(dim=-1) * weight).sum(dim=(-2, -1))
            cross = ((wp * wt).sum(dim=-1) * weight).sum(dim=(-2, -1))
            amse = self._amse(pp, pt, cross)
            total = total + scale_weight * amse.flatten(1).sum(dim=1)
        low_shape = (pred.lowpass.shape[-2], pred.lowpass.shape[-1])
        weight = self._pool_spatial_weights(relative_spatial_weights, low_shape)
        if mask is not None:
            weight = weight * functional.adaptive_avg_pool2d(mask, low_shape)
        total = total + ((pred.lowpass - target.lowpass).square() * weight).flatten(1).sum(dim=1)
        if valid is None:
            return total / self.n_gridpoints
        count = valid.flatten(1).sum(dim=1)
        return torch.where(count > 0, total / count.clamp_min(1e-6), torch.zeros_like(total))

    def _chunk_fields(self, values: torch.Tensor, indices: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Gather selected [B,T,E,V] fields as FP32 [chunk,1,H,W] tensors."""
        return values[indices[0], indices[1], indices[2], :, indices[3]].reshape(-1, 1, *self.grid_shape).float()

    def _chunk_weights(
        self,
        weights: tuple[torch.Tensor, ...],
        indices: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Build relative spatial weights, absolute totals and inclusion masks.

        For each field, total_spatial_weight is sum(w) and
        relative_spatial_weights is H*W*w/sum(w), with zero-safe normalization.
        Without spatial scalers, return compact unit weights, unit totals and
        no inclusion mask. Explicit zero-weight points are excluded.
        """
        if not weights:
            relative_spatial_weights = torch.ones(
                (indices[0].numel(), 1, 1, 1),
                device=indices[0].device,
                dtype=torch.float32,
            )
            return relative_spatial_weights, relative_spatial_weights.flatten(), None
        wc = self._chunk_fields(weights[0], indices)
        for values in weights[1:]:
            wc = wc * self._chunk_fields(values, indices)
        total_spatial_weight = wc.flatten(1).sum(dim=1)
        if not torch.isfinite(total_spatial_weight).all():
            msg = "Wavelet spatial scaler products and total weights must be finite"
            raise ValueError(msg)
        safe_total_spatial_weight = torch.where(
            total_spatial_weight > 0,
            total_spatial_weight,
            torch.ones_like(total_spatial_weight),
        )[:, None, None, None]
        relative_spatial_weights = (wc / safe_total_spatial_weight) * self.n_gridpoints
        return relative_spatial_weights, total_spatial_weight, wc > 0

    def _compute_fields(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Transform and score cropped fields in memory-bounded forward chunks.

        Prepare weights and finite-pair masks per chunk, restore each field's
        absolute spatial weight, and return scores with shape [B,T,E,1,V].
        Autograd may retain coefficient storage across chunks until backward.
        """
        field_shape = (*pred.shape[:3], pred.shape[-1])
        n_fields = math.prod(field_shape)
        if n_fields == 0:
            return pred.float().sum(dim=3, keepdim=True)
        weights = tuple(w.expand_as(pred) for w in weights)
        outputs = []
        chunk_size = self.xfm_batch_chunk_size or n_fields
        for start in range(0, n_fields, chunk_size):
            indices = torch.unravel_index(
                torch.arange(start, min(start + chunk_size, n_fields), device=pred.device),
                field_shape,
            )
            pc, tc = self._chunk_fields(pred, indices), self._chunk_fields(target, indices)
            relative_spatial_weights, total_spatial_weight, included = self._chunk_weights(weights, indices)
            valid = None
            if self.ignore_nans:
                finite = torch.isfinite(pc) & torch.isfinite(tc)
                valid = (finite if included is None else ~included | finite).to(pc.dtype)
                included = finite if included is None else included & finite
            if included is not None:
                pc = torch.where(included, pc, 0.0)
                tc = torch.where(included, tc, 0.0)
            if self.separate_pred_target_xfm:
                pp = self.transform(pc)
                with torch.no_grad():
                    pt = self.transform(tc)
            else:
                combined = self.transform(torch.cat((pc, tc), dim=0))
                n = pc.shape[0]
                pp = WaveletPyramid(combined.lowpass[:n], tuple(h[:n] for h in combined.highpasses))
                pt = WaveletPyramid(combined.lowpass[n:], tuple(h[n:] for h in combined.highpasses))
            outputs.append(self._score_pyramid(pp, pt, relative_spatial_weights, valid) * total_spatial_weight)
        return torch.cat(outputs).reshape(*pred.shape[:3], 1, pred.shape[-1])

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        grid_shard_sizes: list[int] | None = None,
        grid_dim: int | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode = "avg",
    ) -> torch.Tensor:
        """Compute the regional wavelet loss with scalers and final reduction.

        Reconstruct sharded grids before subgrid selection, score FP32 chunks,
        apply non-spatial weights, then gather variable scores if necessary.
        Apply optional result subsetting and the BaseLoss reduction last.
        """
        if grid_dim not in (None, 3, -2):
            msg = "RegionalWaveletLoss requires grid_dim=3 for [B,T,E,G,V] inputs"
            raise ValueError(msg)
        if scaler_indices is not None and not isinstance(scaler_indices, tuple):
            msg = "scaler_indices must be a tuple"
            raise TypeError(msg)
        pred, target, sizes = self._prepare_inputs(pred, target, grid_shard_slice, grid_shard_sizes, group)
        start, stop, _ = self.subgrid.indices(pred.shape[3])
        if (
            self.subgrid.start >= pred.shape[3]
            or (self.subgrid.stop is not None and self.subgrid.stop > pred.shape[3])
            or stop - start != self.n_gridpoints
        ):
            msg = "Selected subgrid point count must equal grid_shape[0] * grid_shape[1]"
            raise ValueError(msg)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            weights, non_spatial_weights = self._weights(pred, without_scalers, sizes, grid_shard_sizes, group)
            result = (
                self._compute_fields(pred[:, :, :, self.subgrid], target[:, :, :, self.subgrid], weights)
                * non_spatial_weights
            )
        if sizes is not None:
            result = gather_tensor(result, 4, sizes, group)
        result, _, _ = subset_tensor(result, scaler_indices)
        return self.reduce(result, squash=squash, squash_mode=squash_mode)
