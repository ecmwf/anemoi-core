# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Band-summed AMSE using a two-dimensional dual-tree complex wavelet transform."""

import math
from collections.abc import Mapping
from typing import Any
from typing import NamedTuple

import torch
from omegaconf import OmegaConf
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.models.layers.wavelet_transforms import DTCWT2D
from anemoi.models.layers.wavelet_transforms import WaveletPyramid
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.schemas.training import WaveletProjectionConfigSchema
from anemoi.training.utils.compile import subset_tensor
from anemoi.training.utils.enums import TensorDim


class WaveletCoefficientValidity(NamedTuple):
    """Validity masks for low-pass and high-pass wavelet coefficients."""

    lowpass: torch.Tensor
    highpasses: tuple[torch.Tensor, ...]


class WaveletAMSELoss(BaseLoss):
    """Band-summed AMSE over 2D dual-tree complex wavelet subbands, plus low-pass MSE.

    Inputs are expected in ``[batch,time,ensemble,grid,variable]`` layout;
    targets may have a singleton ensemble dimension or omit it. Shapes, dtypes
    and devices are assumed to be compatible rather than explicitly checked.
    The selected grid must be an H-by-W rectangle flattened in row-major order.
    Optional ``projection_config`` interpolates arbitrary source nodes to that
    target grid, per field chunk. With ``ignore_nans=True``, NaN values are
    replaced by zero before transforming, and coefficients whose filter support
    intersects a missing value are excluded from the score. Infinities propagate.
    Area weights and GRID-bearing scalers are not supported; non-spatial scalers
    are applied after the wavelet score.

    Notes
    -----
    Area weights, including target-grid weights, and GRID-bearing scalers are
    not supported. The score uses uniform spatial weighting; non-spatial
    scalers such as variable or time weights are applied after scoring.

    With ``ignore_nans=False``, NaNs and infinities propagate through the
    transform and may make the loss nonfinite. With ``ignore_nans=True``, NaNs
    are excluded using coefficient-level validity masks; infinities still
    propagate.

    ``transform`` contains backend/filter options for :class:`DTCWT2D`; J is
    specified on the loss. Compilation through Anemoi's existing configuration
    compiles the tensor scoring kernel, not backend preparation or collectives.
    """

    needs_graph_data: bool = True
    needs_data_node_name: bool = True

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
        projection_config: Mapping[str, Any] | WaveletProjectionConfigSchema | None = None,
        graph_data: HeteroData | None = None,
        data_node_name: str = DEFAULT_DATASET_NAME,
        transform_prediction_and_target_separately: bool | None = None,
        separate_transform_gridpoint_threshold: int = 1_000_000,
        transform_field_chunk_size: int | None = None,
    ) -> None:
        """Configure the 2D scoring grid, DTCWT, scale weights and chunking.

        If omitted, J is selected from the shortest grid dimension. Set
        ``transform_prediction_and_target_separately`` to control whether
        prediction and target use separate transform calls. If it is None,
        ``separate_transform_gridpoint_threshold`` selects the mode based on
        grid size. ``transform_field_chunk_size`` limits fields per transform
        chunk; None processes all fields together.
        """
        super().__init__(ignore_nans=ignore_nans)
        self.grid_shape = tuple(grid_shape)
        self.n_gridpoints = math.prod(grid_shape)
        self.J = max(1, int(math.log2(min(grid_shape))) - 3) if J is None else J
        if scale_weights is not None and len(scale_weights) != self.J:
            msg = "scale_weights must contain one weight per wavelet level"
            raise ValueError(msg)
        self.subgrid = slice(*(subgrid or (0, None)))
        self.scale_weights = tuple(scale_weights) if scale_weights is not None else (1.0,) * self.J
        self.eps = eps
        self.transform_field_chunk_size = transform_field_chunk_size
        self.transform_prediction_and_target_separately = (
            self.n_gridpoints > separate_transform_gridpoint_threshold
            if transform_prediction_and_target_separately is None
            else transform_prediction_and_target_separately
        )
        self.projection_provider = self._build_projection(projection_config, graph_data, data_node_name, subgrid)
        if self.projection_provider is not None:
            self.projector = SparseProjector()
        self.transform = DTCWT2D(J=self.J, **dict(transform or {}))

    def _build_projection(
        self,
        config: Mapping[str, Any] | WaveletProjectionConfigSchema | None,
        graph_data: HeteroData | None,
        data_node_name: str,
        subgrid: tuple[int, int | None] | None,
    ) -> ProjectionGraphProvider | None:
        """Construct and validate a constant, nonnegative interpolation matrix."""
        if config is None:
            return None
        if subgrid is not None:
            msg = "Wavelet projection requires subgrid=None"
            raise ValueError(msg)
        if isinstance(config, WaveletProjectionConfigSchema):
            raw_config = config.model_dump()
        elif OmegaConf.is_config(config):
            raw_config = OmegaConf.to_container(config, resolve=True)
        else:
            raw_config = dict(config)
        raw_config.setdefault("row_normalize", True)
        if (
            raw_config.get("num_nearest_neighbours") is not None
            and graph_data is not None
            and data_node_name not in graph_data.node_types
        ):
            msg = f"Source node type {data_node_name!r} is missing from graph_data"
            raise ValueError(msg)
        provider = ProjectionGraphProvider.from_config(
            raw_config,
            graph_data=graph_data,
            data_node_name=data_node_name,
            validate_weights=True,
        )
        assert provider is not None, "Projection config must construct a provider"
        matrix = provider.get_edges(device=torch.device("cpu"), dtype=torch.float32)
        if matrix.shape[0] != self.n_gridpoints:
            msg = "Projection destination point count must equal grid_shape[0] * grid_shape[1]"
            raise ValueError(msg)
        row_sums = torch.segment_reduce(matrix.values(), "sum", lengths=matrix.crow_indices().diff())
        if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-6):
            msg = "Projection requires positive support and rows summing to one; use row_normalize=True"
            raise ValueError(msg)
        return provider

    @property
    def name(self) -> str:
        """Return the identifier used to label this loss."""
        return "wavelet_amse"

    @property
    def needs_shard_layout_info(self) -> bool:
        """Request grid-shard metadata so complete spatial fields can be reconstructed."""
        return True

    def compile_for_training(self, **options: Any) -> None:
        """Compile tensor scoring; real backend transforms intentionally stay eager."""
        self._score_pyramid = torch.compile(self._score_pyramid, **options)

    @staticmethod
    def _validate_scaler_dimensions(dimension: int | tuple[int, ...], scaler: torch.Tensor) -> None:
        """Reject GRID scalers because the wavelet score has no spatial axis to weight.

        Anemoi's normal loss scaling expects a score at every grid point. Wavelet
        scoring collapses the grid into subband scores, so only non-spatial
        scalers can be applied meaningfully.
        """
        scaler_dimensions = (
            tuple(range(dimension, dimension + scaler.ndim)) if isinstance(dimension, int) else dimension
        )
        if any(dim < -5 or dim >= 5 for dim in scaler_dimensions):
            msg = "Wavelet scaler dimensions must index the five-dimensional input layout"
            raise ValueError(msg)
        scaler_dimensions = tuple(dim % 5 for dim in scaler_dimensions)
        if TensorDim.GRID in scaler_dimensions:
            msg = "WaveletAMSELoss does not support area weights or GRID-bearing scalers"
            raise ValueError(msg)
        if len(scaler_dimensions) != scaler.ndim or len(set(scaler_dimensions)) != len(scaler_dimensions):
            msg = f"Scaler dimensions {scaler_dimensions} are incompatible with scaler shape {scaler.shape}"
            raise ValueError(msg)

    def add_scaler(
        self,
        dimension: int | tuple[int, ...],
        scaler: torch.Tensor,
        *,
        name: str | None = None,
    ) -> None:
        """Validate the wavelet-specific restriction, then delegate registration to Anemoi.

        This override gives config-driven scaler registration an immediate,
        actionable error for unsupported GRID weights.
        """
        self._validate_scaler_dimensions(dimension, scaler)
        super().add_scaler(dimension=dimension, scaler=scaler, name=name)

    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        """Validate updated dimensions before delegating dynamic scaler updates to Anemoi.

        Scalers can change shape at runtime, so the GRID restriction must also
        be enforced when Anemoi updates a previously registered scaler.
        """
        scaler_dimensions = self.scaler.specified_dimensions[name]
        self._validate_scaler_dimensions(scaler_dimensions, scaler)
        super().update_scaler(name=name, scaler=scaler, override=override)

    def _prepare_inputs(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        grid_shard_slice: slice | None,
        grid_shard_sizes: list[int] | None,
        group: ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int] | None]:
        """Broadcast the target and reconstruct grid-sharded inputs.

        Sharded inputs are transposed from grid shards to variable shards.
        Return their variable-shard sizes, or None for unsharded inputs.
        """
        if target.ndim == 4:
            target = target.unsqueeze(2)
        target = target.detach().expand_as(pred)
        if (grid_shard_slice is None) != (grid_shard_sizes is None):
            msg = "grid_shard_slice and grid_shard_sizes must be provided together"
            raise ValueError(msg)
        if grid_shard_sizes is None:
            return pred, target, None
        if group is None:
            msg = "Sharded wavelet inputs require a process group"
            raise ValueError(msg)
        variable_shard_sizes = get_shard_sizes(pred, TensorDim.VARIABLE, group)
        pred = all_to_all_transpose(
            pred,
            TensorDim.VARIABLE,
            variable_shard_sizes,
            TensorDim.GRID,
            grid_shard_sizes,
            group,
        )
        target = all_to_all_transpose(
            target,
            TensorDim.VARIABLE,
            variable_shard_sizes,
            TensorDim.GRID,
            grid_shard_sizes,
            group,
        )
        return pred, target, variable_shard_sizes

    def _scale_scores(
        self,
        scores: torch.Tensor,
        without_scalers: list[str] | list[int] | None,
    ) -> torch.Tensor:
        """Apply Anemoi's non-spatial scalers to the collapsed wavelet scores.

        ``BaseLoss.scale`` requires a GRID scaler, but wavelet scores have
        already collapsed that dimension. Use ``ScaleTensor`` directly on the
        retained batch/time/ensemble/variable axes, after gathering variable
        shards so variable scalers remain globally aligned.
        """
        scaler_specs = self.scaler.tensors
        for scaler_dimensions, scaler in scaler_specs.values():
            self._validate_scaler_dimensions(scaler_dimensions, scaler)
        if not scaler_specs:
            return scores

        scalers = self.scaler.resolve(scores.ndim)
        if without_scalers:
            scalers = scalers.without(without_scalers)
        scalers = scalers.to(device=scores.device, dtype=scores.dtype)
        for name, (scaler_dimensions, scaler) in scalers.tensors.items():
            scaler_dimensions = tuple(int(dim) % scores.ndim for dim in scaler_dimensions)
            if len(scaler_dimensions) != scaler.ndim or len(set(scaler_dimensions)) != len(scaler_dimensions):
                msg = f"Scaler {name!r} has incompatible dimensions {scaler_dimensions} and shape {scaler.shape}"
                raise ValueError(msg)
            if any(
                scaler_size not in (1, scores.shape[dimension])
                for dimension, scaler_size in zip(scaler_dimensions, scaler.shape, strict=True)
            ):
                msg = f"Scaler {name!r} does not broadcast to the wavelet score dimensions"
                raise ValueError(msg)
        return scalers.scale_iteratively(scores)

    def _amse(
        self,
        pred_power: torch.Tensor,
        target_power: torch.Tensor,
        cross_power: torch.Tensor,
    ) -> torch.Tensor:
        """Compute epsilon-regularized AMSE from band power and real cross terms."""
        pred_amplitude = torch.sqrt(pred_power + self.eps)
        target_amplitude = torch.sqrt(target_power + self.eps)
        return (pred_amplitude - target_amplitude).square() + 2 * torch.maximum(
            pred_power,
            target_power,
        ) * (1 - cross_power / (pred_amplitude * target_amplitude + self.eps))

    def _rescale_valid_coefficient_sum(self, scores: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        """Rescale the sum over valid coefficients to the all-valid grid size."""
        valid_count = validity.flatten(1).sum(dim=1).to(dtype=scores.dtype)
        total_count = validity[0].numel()
        valid_score_sum = scores.masked_fill(~validity, 0).flatten(1).sum(dim=1)
        return valid_score_sum / valid_count.clamp_min(1) * total_count

    def _score_pyramid(
        self,
        pred: WaveletPyramid,
        target: WaveletPyramid,
        coefficient_validity: WaveletCoefficientValidity | None = None,
    ) -> torch.Tensor:
        """Combine uniformly weighted subband AMSE and low-pass MSE per field."""
        total_score = pred.lowpass.new_zeros(pred.lowpass.shape[0])
        for level_index, (scale_weight, pred_highpass, target_highpass) in enumerate(
            zip(self.scale_weights, pred.highpasses, target.highpasses, strict=True),
        ):
            pred_power = pred_highpass.square().sum(dim=-1)
            target_power = target_highpass.square().sum(dim=-1)
            cross_power = (pred_highpass * target_highpass).sum(dim=-1)
            if coefficient_validity is not None:
                validity = coefficient_validity.highpasses[level_index]
                valid_count = validity.sum(dim=(-2, -1)).to(dtype=pred_power.dtype)
                total_count = validity.shape[-2] * validity.shape[-1]
                normalization = total_count / valid_count.clamp_min(1)
                pred_power = pred_power.masked_fill(~validity, 0).sum(dim=(-2, -1)) * normalization
                target_power = target_power.masked_fill(~validity, 0).sum(dim=(-2, -1)) * normalization
                cross_power = cross_power.masked_fill(~validity, 0).sum(dim=(-2, -1)) * normalization
                band_score = self._amse(pred_power, target_power, cross_power)
                band_score = band_score.masked_fill(valid_count == 0, 0)
            else:
                band_score = self._amse(
                    pred_power.sum(dim=(-2, -1)),
                    target_power.sum(dim=(-2, -1)),
                    cross_power.sum(dim=(-2, -1)),
                )
            band_score = band_score.flatten(1).sum(dim=1)
            total_score = total_score + scale_weight * band_score
        lowpass_error = (pred.lowpass - target.lowpass).square()
        if coefficient_validity is None:
            lowpass_error = lowpass_error.flatten(1).sum(dim=1)
        else:
            lowpass_error = self._rescale_valid_coefficient_sum(lowpass_error, coefficient_validity.lowpass)
        total_score = total_score + lowpass_error
        return total_score / self.n_gridpoints

    @staticmethod
    def _flat_fields(values: torch.Tensor, field_indices: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Gather only the requested [B,T,E,V] fields as FP32 [chunk,grid]."""
        return values[field_indices[0], field_indices[1], field_indices[2], :, field_indices[3]].float()

    def _chunk_fields(self, values: torch.Tensor, field_indices: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Gather selected [B,T,E,V] fields as FP32 [chunk,1,H,W] tensors."""
        return self._flat_fields(values, field_indices).reshape(-1, 1, *self.grid_shape)

    def _chunk_inputs(
        self,
        values: torch.Tensor,
        field_indices: tuple[torch.Tensor, ...],
        projection_matrix: torch.Tensor | None,
    ) -> torch.Tensor:
        """Optionally interpolate a source-field chunk before its target reshape."""
        if projection_matrix is None:
            return self._chunk_fields(values, field_indices)
        fields = self._flat_fields(values, field_indices).unsqueeze(-1)
        return self.projector(fields, projection_matrix).reshape(-1, 1, *self.grid_shape)

    def _chunk_inputs_with_validity(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        field_indices: tuple[torch.Tensor, ...],
        projection_matrix: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask matching NaNs and project values and validity together."""
        pred_fields = self._flat_fields(pred, field_indices)
        target_fields = self._flat_fields(target, field_indices)
        valid_source_fields = ~torch.isnan(pred_fields) & ~torch.isnan(target_fields)
        pred_fields = pred_fields.masked_fill(~valid_source_fields, 0.0)
        target_fields = target_fields.masked_fill(~valid_source_fields, 0.0)

        if projection_matrix is not None:
            pred_fields = self.projector(pred_fields.unsqueeze(-1), projection_matrix).squeeze(-1)
            target_fields = self.projector(target_fields.unsqueeze(-1), projection_matrix).squeeze(-1)
            valid_fraction = self.projector(
                valid_source_fields.to(dtype=torch.float32).unsqueeze(-1),
                projection_matrix,
            ).squeeze(-1)
            valid_source_fields = torch.isclose(
                valid_fraction,
                torch.ones_like(valid_fraction),
                rtol=1e-5,
                atol=1e-6,
            )

        return (
            pred_fields.reshape(-1, 1, *self.grid_shape),
            target_fields.reshape(-1, 1, *self.grid_shape),
            valid_source_fields.reshape(-1, 1, *self.grid_shape),
        )

    def _transform_coefficient_validity(self, valid_grid: torch.Tensor) -> WaveletCoefficientValidity:
        """Propagate invalid input support through the DTCWT filter bank."""
        mask_input = torch.zeros_like(valid_grid, dtype=torch.float32)
        mask_input = mask_input.masked_fill(~valid_grid, float("nan"))
        with torch.no_grad():
            mask_pyramid = self.transform(mask_input)
        return WaveletCoefficientValidity(
            lowpass=torch.isfinite(mask_pyramid.lowpass),
            highpasses=tuple(torch.isfinite(level).all(dim=-1) for level in mask_pyramid.highpasses),
        )

    def _compute_fields(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        projection_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform and score cropped fields in memory-bounded forward chunks.

        Return scores with shape [B,T,E,1,V].
        Autograd may retain coefficient storage across chunks until backward.
        """
        field_shape = (*pred.shape[:3], pred.shape[-1])
        n_fields = math.prod(field_shape)
        if n_fields == 0:
            return pred.float().sum(dim=3, keepdim=True)
        chunk_scores = []
        chunk_size = n_fields if self.transform_field_chunk_size is None else self.transform_field_chunk_size
        for start in range(0, n_fields, chunk_size):
            field_indices = torch.unravel_index(
                torch.arange(start, min(start + chunk_size, n_fields), device=pred.device),
                field_shape,
            )
            coefficient_validity = None
            if self.ignore_nans:
                pred_chunk, target_chunk, valid_grid = self._chunk_inputs_with_validity(
                    pred,
                    target,
                    field_indices,
                    projection_matrix,
                )
                coefficient_validity = self._transform_coefficient_validity(valid_grid)
            else:
                pred_chunk = self._chunk_inputs(pred, field_indices, projection_matrix)
                target_chunk = self._chunk_inputs(target, field_indices, projection_matrix)
            if self.transform_prediction_and_target_separately:
                pred_pyramid = self.transform(pred_chunk)
                with torch.no_grad():
                    target_pyramid = self.transform(target_chunk)
            else:
                combined_pyramid = self.transform(torch.cat((pred_chunk, target_chunk), dim=0))
                n_chunk_fields = pred_chunk.shape[0]
                pred_pyramid = WaveletPyramid(
                    combined_pyramid.lowpass[:n_chunk_fields],
                    tuple(highpass[:n_chunk_fields] for highpass in combined_pyramid.highpasses),
                )
                target_pyramid = WaveletPyramid(
                    combined_pyramid.lowpass[n_chunk_fields:],
                    tuple(highpass[n_chunk_fields:] for highpass in combined_pyramid.highpasses),
                )
            chunk_scores.append(
                self._score_pyramid(pred_pyramid, target_pyramid, coefficient_validity),
            )
        return torch.cat(chunk_scores).reshape(*pred.shape[:3], 1, pred.shape[-1])

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
        """Compute wavelet AMSE with optional grid projection, scalers and final reduction.

        Reconstruct sharded grids before subgrid selection and score FP32
        chunks. Gather variable scores before applying non-spatial scalers, then
        apply optional result subsetting and the BaseLoss reduction.
        """
        if grid_dim not in (None, 3, -2):
            msg = "WaveletAMSELoss requires grid_dim=3 for [B,T,E,G,V] inputs"
            raise ValueError(msg)
        if scaler_indices is not None and not isinstance(scaler_indices, tuple):
            msg = "scaler_indices must be a tuple"
            raise TypeError(msg)
        pred, target, variable_shard_sizes = self._prepare_inputs(
            pred,
            target,
            grid_shard_slice,
            grid_shard_sizes,
            group,
        )
        with torch.autocast(device_type=pred.device.type, enabled=False):
            projection_matrix = (
                self.projection_provider.get_edges(device=pred.device, dtype=torch.float32)
                if self.projection_provider is not None
                else None
            )
            result = self._compute_fields(
                pred[:, :, :, self.subgrid],
                target[:, :, :, self.subgrid],
                projection_matrix,
            )
        if variable_shard_sizes is not None:
            result = gather_tensor(result, 4, variable_shard_sizes, group)
        result = self._scale_scores(result, without_scalers)
        result, _, _ = subset_tensor(result, scaler_indices)
        return self.reduce(result, squash=squash, squash_mode=squash_mode)
