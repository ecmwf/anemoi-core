# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math
from collections.abc import Mapping
from typing import Any
from typing import Literal
from typing import NamedTuple

import torch
import torch.nn.functional as functional
from einops import rearrange
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


class DTCWT2DLoss(BaseLoss):
    r"""Dual-tree complex wavelet loss for scale- and orientation-aware comparison.

    The transform requires a regular 2D grid; an optional sparse projection can be used to
    map irregular source nodes onto an appropriate grid before scoring.

    For level :math:`j` and orientation :math:`o`, let :math:`P_{j,o}` and
    :math:`T_{j,o}` be prediction and target powers summed over spatial
    coefficients, and :math:`C_{j,o}` their real cross-power. Each detail band
    contributes

    .. math::

        \ell_{j,o} =
        \left(\sqrt{P_{j,o}+\varepsilon}-\sqrt{T_{j,o}+\varepsilon}\right)^2
        + 2\max(P_{j,o},T_{j,o})
          \left(1-\frac{C_{j,o}}
          {\sqrt{P_{j,o}+\varepsilon}\sqrt{T_{j,o}+\varepsilon}+\varepsilon}\right).

    The loss sums over orientations, weights levels, adds a low-pass squared-error
    term, and is normalized by the input grid size. Spatially localized wavelet
    filters make the comparison sensitive to structure across scales and orientations.

    Implementation from Chappell et al. (2026), arXiv:2610.37325.
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
        nan_handling: Literal["strict", "pooled"] | None = None,
        subgrid: tuple[int, int | None] | None = None,
        projection_config: Mapping[str, Any] | WaveletProjectionConfigSchema | None = None,
        graph_data: HeteroData | None = None,
        data_node_name: str = DEFAULT_DATASET_NAME,
        transform_prediction_and_target_separately: bool | None = None,
        separate_transform_gridpoint_threshold: int = 1_000_000,
        transform_field_chunk_size: int | None = None,
    ) -> None:
        """Create a 2D DTCWT loss.

        Parameters
        ----------
        grid_shape
            Height and width of the rectangular transform grid. After optional
            projection, the transform input must contain ``H * W`` nodes in
            row-major order.
        J
            Number of DTCWT levels. If None, use
            ``max(1, floor(log2(min(grid_shape))) - 3)``.
        transform
            Backend and filter options forwarded to :class:`DTCWT2D`.
        scale_weights
            Optional weight for each wavelet level. Defaults to one per level;
            if provided, must contain exactly ``J`` weights.
        eps
            Positive regularization constant used in the subband score.
        ignore_nans
            Enable strict NaN handling for backward compatibility. NaN-affected
            coefficients are excluded; infinities still propagate.
        nan_handling
            NaN handling strategy: ``"strict"`` excludes coefficients whose
            filter support intersects missing inputs, while ``"pooled"`` weights
            coefficients by fractional valid support. Either option enables NaN
            handling. If omitted, ``ignore_nans=True`` selects ``"strict"``.
        subgrid
            Optional ``(start, end)`` slice of the flattened input grid, applied
            before projection and transformation; ``end=None`` selects through
            the final node. With projection, its source columns must match the
            selected nodes and their ordering.
        projection_config
            Optional sparse projection from the selected input nodes to the
            ``grid_shape`` target. Matrix columns must match the selected source
            nodes, and destination rows must match ``H * W``. See
            :meth:`~anemoi.models.layers.graph_provider.ProjectionGraphProvider.from_config`
            for supported modes.
        graph_data
            Model graph required for edge-based and target-grid projection modes;
            not needed when ``projection_config`` uses a saved matrix.
        data_node_name
            Graph node type containing the projection's source nodes.
        transform_prediction_and_target_separately
            Whether to transform predictions and targets in separate calls to
            reduce memory usage. If None, select separate calls when the grid exceeds
            ``separate_transform_gridpoint_threshold``.
        separate_transform_gridpoint_threshold
            Grid-point threshold for selecting separate transform calls when
            ``transform_prediction_and_target_separately`` is None.
        transform_field_chunk_size
            Maximum number of fields transformed per chunk. If None, process all
            fields together.
        """
        super().__init__(ignore_nans=ignore_nans or nan_handling is not None)
        self.nan_handling = nan_handling or ("strict" if ignore_nans else None)
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
        self.projection_provider = self._build_projection(projection_config, graph_data, data_node_name)
        if self.projection_provider is not None:
            self.projector = SparseProjector()
        self.transform = DTCWT2D(J=self.J, **dict(transform or {}))

    def _build_projection(
        self,
        config: Mapping[str, Any] | WaveletProjectionConfigSchema | None,
        graph_data: HeteroData | None,
        data_node_name: str,
    ) -> ProjectionGraphProvider | None:
        """Construct and validate a constant, nonnegative interpolation matrix."""
        if config is None:
            return None
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
        # Each destination row is a weighted average, so a constant source field stays constant.
        row_sums = torch.segment_reduce(matrix.values(), "sum", lengths=matrix.crow_indices().diff())
        if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-6):
            msg = "Projection requires positive support and rows summing to one; use row_normalize=True"
            raise ValueError(msg)
        return provider

    @property
    def name(self) -> str:
        return "dtcwt_2d"

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

        The base loss scaling implementation expects a score at every grid point. Wavelet
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
            msg = "DTCWT2DLoss does not support area weights or GRID-bearing scalers"
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
        """Validate the wavelet-specific restriction, then delegate registration to the base class.

        This override gives config-driven scaler registration an immediate,
        actionable error for unsupported GRID weights.
        """
        self._validate_scaler_dimensions(dimension, scaler)
        super().add_scaler(dimension=dimension, scaler=scaler, name=name)

    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        """Validate updated dimensions before delegating dynamic scaler updates to the base class.

        Scalers can change shape at runtime, so the GRID restriction must also
        be enforced when the base class updates a previously registered scaler.
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
        """Apply non-spatial scalers to the collapsed wavelet scores.

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

    def _band_score(
        self,
        pred_power: torch.Tensor,
        target_power: torch.Tensor,
        cross_power: torch.Tensor,
    ) -> torch.Tensor:
        """Score band magnitude and alignment using regularized power terms.

        The first term penalizes magnitude differences; the second penalizes
        disagreement in normalized cross-power.
        """
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
        valid_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine uniformly weighted subband scores and low-pass MSE per field."""
        pooled_support = self.nan_handling == "pooled" and coefficient_validity is not None
        if pooled_support and valid_grid is None:
            msg = "Pooled NaN handling requires the input-grid validity mask"
            raise ValueError(msg)
        total_score = pred.lowpass.new_zeros(pred.lowpass.shape[0])
        for level_index, (scale_weight, pred_highpass, target_highpass) in enumerate(
            zip(self.scale_weights, pred.highpasses, target.highpasses, strict=True),
        ):
            # Collapse the coefficient-component axis but retain spatial positions for validity weighting.
            pred_power = pred_highpass.square().sum(dim=-1)
            target_power = target_highpass.square().sum(dim=-1)
            cross_power = (pred_highpass * target_highpass).sum(dim=-1)
            if coefficient_validity is not None:
                validity = coefficient_validity.highpasses[level_index]
                if pooled_support:
                    # Broadcast fractional spatial support over the high-pass orientations.
                    support = validity.unsqueeze(2)
                    pred_power = (pred_power * support).sum(dim=(-2, -1))
                    target_power = (target_power * support).sum(dim=(-2, -1))
                    cross_power = (cross_power * support).sum(dim=(-2, -1))
                else:
                    valid_count = validity.sum(dim=(-2, -1)).to(dtype=pred_power.dtype)
                    total_count = validity.shape[-2] * validity.shape[-1]
                    # Rescale the valid-only average to the coefficient count of a complete field.
                    normalization = total_count / valid_count.clamp_min(1)
                    pred_power = pred_power.masked_fill(~validity, 0).sum(dim=(-2, -1)) * normalization
                    target_power = target_power.masked_fill(~validity, 0).sum(dim=(-2, -1)) * normalization
                    cross_power = cross_power.masked_fill(~validity, 0).sum(dim=(-2, -1)) * normalization
                band_score = self._band_score(pred_power, target_power, cross_power)
                if not pooled_support:
                    band_score = band_score.masked_fill(valid_count == 0, 0)
            else:
                band_score = self._band_score(
                    pred_power.sum(dim=(-2, -1)),
                    target_power.sum(dim=(-2, -1)),
                    cross_power.sum(dim=(-2, -1)),
                )
            # Sum orientations at this level before applying its configured weight.
            band_score = band_score.flatten(1).sum(dim=1)
            total_score = total_score + scale_weight * band_score
        lowpass_error = (pred.lowpass - target.lowpass).square()
        if coefficient_validity is None:
            lowpass_error = lowpass_error.flatten(1).sum(dim=1)
        elif pooled_support:
            lowpass_error = (lowpass_error * coefficient_validity.lowpass).flatten(1).sum(dim=1)
        else:
            lowpass_error = self._rescale_valid_coefficient_sum(lowpass_error, coefficient_validity.lowpass)
        total_score = total_score + lowpass_error
        if pooled_support:
            # Pooled scores are normalized by valid input cells, not by the full grid area.
            valid_count = valid_grid.flatten(1).sum(dim=1).to(dtype=total_score.dtype).clamp_min(1)
            return total_score / valid_count
        return total_score / self.n_gridpoints

    @staticmethod
    def _flat_fields(values: torch.Tensor, field_indices: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Gather only the requested [B,T,E,V] fields as FP32 [chunk,grid]."""
        # Paired indices select one B/T/E/V combination per row; ':' retains every grid point.
        return values[field_indices[0], field_indices[1], field_indices[2], :, field_indices[3]].float()

    def _chunk_inputs(
        self,
        values: torch.Tensor,
        field_indices: tuple[torch.Tensor, ...],
        projection_matrix: torch.Tensor | None,
    ) -> torch.Tensor:
        """Gather a field chunk, optionally interpolate it, and reshape to the target grid."""
        fields = self._flat_fields(values, field_indices)
        if projection_matrix is not None:
            # SparseProjector expects node features with a trailing channel dimension.
            fields = self.projector(fields.unsqueeze(-1), projection_matrix).squeeze(-1)
        # Restore the ordered H*W destination nodes to the rectangular transform layout.
        return rearrange(fields, "chunk (h w) -> chunk 1 h w", h=self.grid_shape[0], w=self.grid_shape[1])

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
        # Compare only source locations that are non-NaN in both fields; zero-fill them before projection.
        valid_source_fields = ~torch.isnan(pred_fields) & ~torch.isnan(target_fields)
        pred_fields = pred_fields.masked_fill(~valid_source_fields, 0.0)
        target_fields = target_fields.masked_fill(~valid_source_fields, 0.0)

        if projection_matrix is not None:
            pred_fields = self.projector(pred_fields.unsqueeze(-1), projection_matrix).squeeze(-1)
            target_fields = self.projector(target_fields.unsqueeze(-1), projection_matrix).squeeze(-1)
            # A projected destination is valid only if all of its interpolation support is valid.
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
            rearrange(pred_fields, "chunk (h w) -> chunk 1 h w", h=self.grid_shape[0], w=self.grid_shape[1]),
            rearrange(target_fields, "chunk (h w) -> chunk 1 h w", h=self.grid_shape[0], w=self.grid_shape[1]),
            rearrange(
                valid_source_fields,
                "chunk (h w) -> chunk 1 h w",
                h=self.grid_shape[0],
                w=self.grid_shape[1],
            ),
        )

    def _transform_coefficient_validity(self, valid_grid: torch.Tensor) -> WaveletCoefficientValidity:
        """Propagate invalid input support through the DTCWT filter bank."""
        # NaNs propagate through the filters, marking coefficients touched by missing inputs.
        mask_input = torch.zeros_like(valid_grid, dtype=torch.float32)
        mask_input = mask_input.masked_fill(~valid_grid, float("nan"))
        with torch.no_grad():
            mask_pyramid = self.transform(mask_input)
        return WaveletCoefficientValidity(
            lowpass=torch.isfinite(mask_pyramid.lowpass),
            highpasses=tuple(torch.isfinite(level).all(dim=-1) for level in mask_pyramid.highpasses),
        )

    def _pooled_coefficient_support(
        self,
        valid_grid: torch.Tensor,
        pyramid: WaveletPyramid,
    ) -> WaveletCoefficientValidity:
        """Pool input validity recursively to each wavelet coefficient resolution."""
        support = valid_grid.to(dtype=torch.float32)
        highpasses = []
        # Pool successively so fractional support follows each coarser wavelet resolution.
        for level in pyramid.highpasses:
            support = functional.adaptive_avg_pool2d(support, output_size=level.shape[-3:-1])
            highpasses.append(support)
        lowpass_support = functional.adaptive_avg_pool2d(support, output_size=pyramid.lowpass.shape[-2:])
        return WaveletCoefficientValidity(lowpass=lowpass_support, highpasses=tuple(highpasses))

    def _compute_fields(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        projection_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform and score fields in chunks to manage memory usage.

        Return scores with shape [B,T,E,1,V] for scaling and reducing.
        """
        # Treat each batch/time/ensemble/variable combination as an independent field.
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
            valid_grid = None
            if self.ignore_nans:
                pred_chunk, target_chunk, valid_grid = self._chunk_inputs_with_validity(
                    pred,
                    target,
                    field_indices,
                    projection_matrix,
                )
            else:
                pred_chunk = self._chunk_inputs(pred, field_indices, projection_matrix)
                target_chunk = self._chunk_inputs(target, field_indices, projection_matrix)
            if self.transform_prediction_and_target_separately:
                # Separate calls reduce peak transform-batch memory for large fields.
                pred_pyramid = self.transform(pred_chunk)
                with torch.no_grad():
                    target_pyramid = self.transform(target_chunk)
            else:
                # One combined call is cheaper when the doubled field batch fits in memory.
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
            coefficient_validity = None
            if valid_grid is not None:
                if self.nan_handling == "pooled":
                    coefficient_validity = self._pooled_coefficient_support(valid_grid, pred_pyramid)
                else:
                    coefficient_validity = self._transform_coefficient_validity(valid_grid)
            chunk_scores.append(
                self._score_pyramid(pred_pyramid, target_pyramid, coefficient_validity, valid_grid),
            )
        return rearrange(
            torch.cat(chunk_scores),
            "(b t e v) -> b t e 1 v",
            b=pred.shape[0],
            t=pred.shape[1],
            e=pred.shape[2],
            v=pred.shape[-1],
        )

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
        """Compute the wavelet score with optional projection, scalers and final reduction.

        Reconstructs sharded grids before subgrid selection and scores FP32 chunks.
        Gathers variable scores before applying non-spatial scalers, then
        applies optional result subsetting and the BaseLoss reduction.
        """
        if grid_dim not in (None, 3, -2):
            msg = "DTCWT2DLoss requires grid_dim=3 for [B,T,E,G,V] inputs"
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
            # Projection columns must match this selected source domain; full-grid matrices are not sliced here.
            selected_pred = pred[:, :, :, self.subgrid]
            selected_target = target[:, :, :, self.subgrid]
            if projection_matrix is not None and projection_matrix.shape[1] != selected_pred.shape[3]:
                msg = (
                    f"Projection source point count ({projection_matrix.shape[1]}) must match "
                    f"selected input grid size ({selected_pred.shape[3]})"
                )
                raise ValueError(msg)
            result = self._compute_fields(
                selected_pred,
                selected_target,
                projection_matrix,
            )
        if variable_shard_sizes is not None:
            result = gather_tensor(result, 4, variable_shard_sizes, group)
        result = self._scale_scores(result, without_scalers)
        result, _, _ = subset_tensor(result, scaler_indices)
        return self.reduce(result, squash=squash, squash_mode=squash_mode)
