# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spectral-domain losses.

This module consolidates spectral losses that were historically split across
`spatial.py` and `spectral.py`.

Notes
-----
* These losses operate on tensors whose *spatial* dimension is flattened
  (i.e. `(..., grid, variables)`), and internally reshape to 2D grids for FFT2D.
* For backwards compatibility, legacy class names (e.g. ``LogFFT2Distance``)
  are kept.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import einops
import torch
from omegaconf import OmegaConf

from anemoi.graphs.builders import build_node_to_node_projection_subgraph
from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.graph_provider import create_projection_graph_provider
from anemoi.models.layers.graph_provider import normalize_projection_edges_name
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.models.layers.sparse_projector import apply_sparse_projector_with_reshaping
from anemoi.models.layers.spectral_transforms import DCT2D
from anemoi.models.layers.spectral_transforms import FFT2D
from anemoi.models.layers.spectral_transforms import OctahedralSHT
from anemoi.models.layers.spectral_transforms import ReducedSHT
from anemoi.models.layers.spectral_transforms import SpectralTransform
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.kcrps import AlmostFairKernelCRPS
from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


def _ensure_without_scalers_has_grid_dimension(without_scalers: list[str] | list[int] | None) -> list[str] | list[int]:
    """Temporary fix for https://github.com/ecmwf/anemoi-core/issues/725.

    Some pipelines pass numeric scaler indices and rely on excluding scalers over grid dimension
    by default. Ensure this exclusion is present for numeric lists.
    """
    if without_scalers is None:
        return [TensorDim.GRID.value]
    if len(without_scalers) == 0:
        return [TensorDim.GRID.value]
    if not isinstance(without_scalers[0], str) and TensorDim.GRID.value not in without_scalers:
        without_scalers.append(TensorDim.GRID.value)  # type: ignore[arg-type]
    return without_scalers


def _normalise_mapping(config: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    """Convert config-like objects to a plain mapping."""
    if config is None:
        return {}
    if OmegaConf.is_config(config):
        return dict(OmegaConf.to_container(config, resolve=True))
    return dict(config)


def _require_graph_data(graph_data: HeteroData | None, message: str) -> HeteroData:
    """Return graph data or raise a config error."""
    if graph_data is None:
        raise ValueError(message)
    return graph_data


def _projection_provider_from_edges(
    *,
    graph_data: HeteroData | None,
    edges_name: tuple[str, str, str],
    edge_weight_attribute: str | None,
    src_node_weight_attribute: str | None,
    row_normalize: bool,
) -> ProjectionGraphProvider:
    """Create a provider from existing graph edges."""
    graph_data = _require_graph_data(graph_data, "graph_data must be provided when projection edges are configured.")
    return create_projection_graph_provider(
        graph=graph_data,
        edges_name=edges_name,
        edge_weight_attribute=edge_weight_attribute,
        src_node_weight_attribute=src_node_weight_attribute,
        row_normalize=row_normalize,
    )


def _projection_provider_from_matrix(
    *,
    matrix_path: str | Path,
    row_normalize: bool,
) -> ProjectionGraphProvider:
    """Create a provider from a serialized sparse matrix."""
    return create_projection_graph_provider(
        file_path=matrix_path,
        row_normalize=row_normalize,
    )


def _projection_subgraph_from_target_grid_config(
    *,
    graph_data: HeteroData,
    data_node_name: str,
    projection_node_name: str,
    projection_config: Mapping[str, Any],
) -> HeteroData:
    """Build projection edges from the source node to a configured target grid."""
    return build_node_to_node_projection_subgraph(
        graph_data,
        data_node_name,
        projection_node_name,
        projection_config,
        target_grid_keys=("grid", "target_grid"),
    )


def _projection_provider_from_target_grid_config(
    *,
    graph_data: HeteroData | None,
    data_node_name: str,
    projection_config: Mapping[str, Any],
    edge_weight_attribute: str | None,
    src_node_weight_attribute: str | None,
    row_normalize: bool,
) -> ProjectionGraphProvider:
    """Build a target-grid projection subgraph and create its provider."""
    graph_data = _require_graph_data(graph_data, "graph_data must be provided for on-the-fly projection_config.")
    projection_node_name = projection_config.get("projection_node_name", "projection")
    subgraph = _projection_subgraph_from_target_grid_config(
        graph_data=graph_data,
        data_node_name=data_node_name,
        projection_node_name=projection_node_name,
        projection_config=projection_config,
    )
    return create_projection_graph_provider(
        graph=subgraph,
        edges_name=(data_node_name, "to", projection_node_name),
        edge_weight_attribute=edge_weight_attribute or DEFAULT_EDGE_WEIGHT_ATTRIBUTE,
        src_node_weight_attribute=src_node_weight_attribute,
        row_normalize=row_normalize,
    )


def _resolve_spectral_projection_provider(
    *,
    graph_data: HeteroData | None,
    data_node_name: str,
    projection_config: Mapping[str, Any] | Any | None,
) -> ProjectionGraphProvider | None:
    """Resolve spectral projection inputs into a provider.

    Supported modes are matrix file, existing graph edges, and on-the-fly
    target-grid projection.

    Config examples::

        # File-backed sparse matrix.
        projection_config:
          matrix_path: /path/to/projection.npz

        # Existing graph edges.
        projection_config:
          edges_name: [data, to, projection]
          edge_weight_attribute: gauss_weight

        # Build a projection from the source node to a target grid.
        projection_config:
          projection_node_name: projection
          node_builder: ...
          num_nearest_neighbours: 4
          sigma: 0.1
          row_normalize: false
    """
    projection_config = _normalise_mapping(projection_config)
    if not projection_config:
        return None

    edge_weight_attribute = projection_config.get("edge_weight_attribute")
    src_node_weight_attribute = projection_config.get("src_node_weight_attribute")
    row_normalize = bool(projection_config.get("row_normalize", False))

    # File-backed sparse matrix.
    if projection_config.get("matrix_path") is not None:
        return _projection_provider_from_matrix(
            matrix_path=projection_config["matrix_path"],
            row_normalize=row_normalize,
        )

    # Existing graph edges.
    if projection_config.get("edges_name") is not None:
        return _projection_provider_from_edges(
            graph_data=graph_data,
            edges_name=normalize_projection_edges_name(projection_config["edges_name"]),
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            row_normalize=row_normalize,
        )

    # On-the-fly target-grid projection.
    return _projection_provider_from_target_grid_config(
        graph_data=graph_data,
        data_node_name=data_node_name,
        projection_config=projection_config,
        edge_weight_attribute=edge_weight_attribute,
        src_node_weight_attribute=src_node_weight_attribute,
        row_normalize=row_normalize,
    )


class SpectralLoss(BaseLoss):
    """Base class for spectral losses."""

    needs_graph_data: bool = True
    transform: SpectralTransform

    def __init__(
        self,
        transform: Literal[
            "fft2d",
            "reduced_sht",
            "octahedral_sht",
            "dct2d",
        ] = "fft2d",
        *,
        ignore_nans: bool = False,
        scalers: list | None = None,
        graph_data: HeteroData | None = None,
        data_node_name: str | None = None,
        nodes_slice: tuple[int | None, ...] | list[int | None] | None = None,
        projection_config: object | None = None,
        projection_autocast: bool = False,
        **kwargs,
    ) -> None:
        """Create a spectral loss.

        Parameters
        ----------
        transform
            Spectral transform type.
        ignore_nans
            Whether to ignore NaNs in the loss computation.
        scalers
            Kept for Hydra/config compatibility. Scaling is handled by BaseLoss.
        graph_data
            Graph used to resolve projection edges or build a projection subgraph.
        data_node_name
            Source node type for projection config.
        nodes_slice
            Slice bounds applied to the grid dimension before projection.
        projection_config
            Matrix file, existing-edge, or target-grid projection config.
        projection_autocast
            Use automatic mixed precision for sparse projection.
        kwargs
            Additional arguments for the spectral transform.
        """
        super().__init__(ignore_nans=ignore_nans)

        _ = scalers

        self.nodes_slice = slice(*(nodes_slice or (0, None))) if nodes_slice is not None else None

        self.projection_provider = _resolve_spectral_projection_provider(
            graph_data=graph_data,
            data_node_name=data_node_name or DEFAULT_DATASET_NAME,
            projection_config=projection_config,
        )
        self.projection = SparseProjector(autocast=projection_autocast)

        self.supports_sharding = True

        if transform == "fft2d":
            LOGGER.info("Using FFT2D spectral transform in spectral loss.")
            self.transform = FFT2D(**kwargs)
        elif transform == "dct2d":
            LOGGER.info("Using DCT2D spectral transform in spectral loss.")
            self.transform = DCT2D(**kwargs)
        elif transform == "reduced_sht":
            # expected additional args: grid
            # optional args: truncation
            LOGGER.info("Using ReducedSHT spectral transform in spectral loss.")
            self.transform = ReducedSHT(**kwargs)
        elif transform == "octahedral_sht":
            # expected additional args: nlat
            # optional args: truncation
            LOGGER.info("Using Octahedral SHT spectral transform in spectral loss.")
            self.transform = OctahedralSHT(**kwargs)
        else:
            msg = f"Unknown transform type: {transform}"
            raise ValueError(msg)

    @property
    def needs_shard_layout_info(self) -> bool:
        return True

    def _apply_nodes_slice(self, x: torch.Tensor) -> torch.Tensor:
        if self.nodes_slice is None:
            return x
        return torch.index_select(x, -2, torch.arange(*self.nodes_slice.indices(x.size(-2)), device=x.device))

    def _prepare_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial preprocessing shared by all spectral transforms."""
        x = self._apply_nodes_slice(x)
        if self.projection_provider is not None:
            x = apply_sparse_projector_with_reshaping(self.projection, x, self.projection_provider)
        return x

    def _to_spectral_flat(
        self,
        x: torch.Tensor,
        *,
        grid_shard_sizes: ShardSizes = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Transform to spectral domain and flatten spectral dimensions."""
        variable_shard_sizes = None
        if grid_shard_sizes is not None:
            variable_shard_sizes = get_shard_sizes(x, -1, group)
            x = all_to_all_transpose(
                x,
                -1,
                variable_shard_sizes,
                -2,
                grid_shard_sizes,
                group,
            )

        x = self._prepare_spatial(x)
        x_spec = self.transform.forward(x)
        # flatten only transformed spatial/spectral dims into one "mode" axis
        spatial_start_dim = x.ndim - 2
        x_spec = x_spec.flatten(start_dim=spatial_start_dim, end_dim=-2)

        if variable_shard_sizes is not None:
            mode_shard_sizes = get_shard_sizes(x_spec, -2, group)
            x_spec = all_to_all_transpose(
                x_spec,
                -2,
                mode_shard_sizes,
                -1,
                variable_shard_sizes,
                group,
            )
        return x_spec


class SpectralL2Loss(SpectralLoss):
    r"""L2 loss in spectral domain.

    .. math::
        \lVert F - \hat F \rVert_2^2
    """

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
        grid_shard_sizes: ShardSizes = None,
        grid_dim: int | None = None,
        squash_mode: str = "avg",
        **kwargs,
    ) -> torch.Tensor:
        del grid_dim, kwargs  # unused
        is_sharded = grid_shard_sizes is not None or grid_shard_slice is not None
        group = group if is_sharded else None
        scale_grid_shard_slice = None if grid_shard_sizes is not None else grid_shard_slice

        pred_spectral = self._to_spectral_flat(pred, grid_shard_sizes=grid_shard_sizes, group=group)
        target_spectral = self._to_spectral_flat(target, grid_shard_sizes=grid_shard_sizes, group=group)

        diff = torch.abs(pred_spectral - target_spectral) ** 2

        result = self.scale(
            diff,
            scaler_indices,
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=scale_grid_shard_slice,
        )
        return self.reduce(result, squash=squash, group=group, squash_mode=squash_mode)


class LogSpectralDistance(SpectralLoss):
    r"""Log Spectral Distance (LSD)."""

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
        grid_shard_sizes: ShardSizes = None,
        grid_dim: int | None = None,
        squash_mode: str = "avg",
    ) -> torch.Tensor:
        del grid_dim  # unused
        is_sharded = grid_shard_sizes is not None or grid_shard_slice is not None
        group = group if is_sharded else None
        scale_grid_shard_slice = None if grid_shard_sizes is not None else grid_shard_slice
        eps = torch.finfo(pred.dtype).eps

        pred_spectral = self._to_spectral_flat(pred, grid_shard_sizes=grid_shard_sizes, group=group)
        target_spectral = self._to_spectral_flat(target, grid_shard_sizes=grid_shard_sizes, group=group)

        power_pred = torch.abs(pred_spectral) ** 2
        power_tgt = torch.abs(target_spectral) ** 2

        log_diff = torch.log(power_tgt + eps) - torch.log(power_pred + eps)

        result = self.scale(
            log_diff**2,
            scaler_indices,
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=scale_grid_shard_slice,
        )
        return torch.sqrt(self.reduce(result, squash=squash, group=group, squash_mode=squash_mode) + eps)


class FourierCorrelationLoss(SpectralLoss):
    r"""Fourier Correlation Loss (FCL)."""

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
        grid_shard_sizes: ShardSizes = None,
        grid_dim: int | None = None,
        squash_mode: str = "avg",
    ) -> torch.Tensor:
        del grid_dim  # unused
        is_sharded = grid_shard_sizes is not None or grid_shard_slice is not None
        group = group if is_sharded else None
        scale_grid_shard_slice = None if grid_shard_sizes is not None else grid_shard_slice
        eps = torch.finfo(pred.dtype).eps

        pred_spectral = self._to_spectral_flat(pred, grid_shard_sizes=grid_shard_sizes, group=group)
        target_spectral = self._to_spectral_flat(target, grid_shard_sizes=grid_shard_sizes, group=group)
        n_modes = pred_spectral.size(dim=TensorDim.GRID.value)

        # compute correlation per mode before applying any external weighting
        # keeps the ratio bounded by Cauchy-Schwarz (up to numerical error)
        cross = torch.real(pred_spectral * torch.conj(target_spectral))
        denom = torch.sqrt(torch.abs(pred_spectral) ** 2 * torch.abs(target_spectral) ** 2 + eps)
        correlation = torch.clamp(cross / denom, min=-1.0, max=1.0)

        # apply weighting/scaling after correlation is computed
        result = (1 - correlation) / n_modes
        result = self.scale(
            result,
            scaler_indices,
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=scale_grid_shard_slice,
        )
        return self.reduce(result, squash=squash, group=group, squash_mode=squash_mode)


class LogFFT2Distance(LogSpectralDistance):
    """Backwards compatible alias for log spectral distance on FFT2D grids."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        ignore_nans: bool = False,
        scalers: list | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            transform="fft2d",
            x_dim=x_dim,
            y_dim=y_dim,
            ignore_nans=ignore_nans,
            scalers=scalers,
            **kwargs,
        )


class SpectralCRPSLoss(SpectralLoss, AlmostFairKernelCRPS):
    """CRPS computed in spectral space using arbitrary spectral transforms.

    Works with:
      - FFT2D
      - DCT2D
      - Reduced SHT
      - Octahedral SHT
    """

    def __init__(
        self,
        transform: Literal[
            "fft2d",
            "dct2d",
            "reduced_sht",
            "octahedral_sht",
        ] = "fft2d",
        *,
        x_dim: int | None = None,
        y_dim: int | None = None,
        alpha: float = 1.0,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        scalers: list | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            transform=transform,
            x_dim=x_dim,
            y_dim=y_dim,
            ignore_nans=ignore_nans,
            scalers=scalers,
            **kwargs,
        )
        self.alpha = alpha
        self.no_autocast = no_autocast

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
        grid_shard_sizes: ShardSizes = None,
        grid_dim: int | None = None,
        squash_mode: str = "avg",
    ) -> torch.Tensor:
        del grid_dim  # unused
        is_sharded = grid_shard_sizes is not None or grid_shard_slice is not None
        group = group if is_sharded else None
        scale_grid_shard_slice = None if grid_shard_sizes is not None else grid_shard_slice

        # → [..., modes, vars]
        pred_spec = self._to_spectral_flat(pred, grid_shard_sizes=grid_shard_sizes, group=group)
        tgt_spec = self._to_spectral_flat(target, grid_shard_sizes=grid_shard_sizes, group=group)

        pred_spec = einops.rearrange(pred_spec, "b t e m v -> b t v m e")  # ensemble dim last for preds
        tgt_spec = einops.rearrange(tgt_spec, "... m v -> (...) v m")  # remove ensemble dim for targets
        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                crps = self._kernel_crps(pred_spec, tgt_spec, alpha=self.alpha)
        else:
            crps = self._kernel_crps(pred_spec, tgt_spec, alpha=self.alpha)
        crps = einops.rearrange(crps, "b t v m -> b t 1 m v")  # consistent with tensordim

        scaled = self.scale(
            crps,
            scaler_indices,
            without_scalers=_ensure_without_scalers_has_grid_dimension(without_scalers),
            grid_shard_slice=scale_grid_shard_slice,
        )
        return self.reduce(scaled, squash=squash, group=group, squash_mode=squash_mode)

    @property
    def name(self) -> str:
        return "CRPS-Spectral"
