# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from abc import abstractmethod
from typing import Optional

import einops
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform
from anemoi.models.layers.spectral_transforms import InverseOctahedralSHT
from anemoi.models.layers.spectral_transforms import InverseRegularSHT


class BaseResidualConnection(nn.Module, ABC):
    """Base class for residual connection modules."""

    def __init__(self, graph: HeteroData | None = None, **_) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        grid_shard_sizes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        """Define the residual connection operation.

        Should be overridden by subclasses.
        """
        pass

    @staticmethod
    def _expand_time(x: torch.Tensor, n_step_output: int | None) -> torch.Tensor:
        if n_step_output is None:
            return x
        return x.unsqueeze(1).expand(-1, n_step_output, -1, -1, -1)


class SkipConnection(BaseResidualConnection):
    """Skip connection module

    This layer returns the most recent timestep from the input sequence.

    This module is used to bypass processing layers and directly pass the latest input forward.
    """

    def __init__(self, step: int = -1, **_) -> None:
        super().__init__()
        self.step = step

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_sizes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        """Return the last timestep of the input sequence."""
        x_skip = x[:, self.step, ...]  # x shape: (batch, time, ens, nodes, features)
        return self._expand_time(x_skip, n_step_output)


class TruncatedConnection(BaseResidualConnection):
    """Truncated skip connection.

    Applies a coarse-graining and reconstruction of input features using sparse
    projections to truncate high-frequency features.

    Edge names and the edge-weight attribute are expected to be pre-resolved by
    ``ProjectionCreator`` and passed in directly.  File-path loading is still
    supported as an alternative to the graph-based path.

    Parameters
    ----------
    graph : HeteroData, optional
        Graph containing the truncation subgraphs.
    src_node_weight_attribute : str, optional
        Source-node attribute used as additional projection weights.
    edge_weight_attribute : str, optional
        Edge attribute used as projection weights (default: ``gauss_weight``).
    truncation_config : dict, optional
        Configuration used to build or load the truncation projections.
    truncation_up_edges_name : tuple[str, str, str], optional
        Pre-resolved ``(src, relation, dst)`` edge type for the up-projection.
    truncation_down_edges_name : tuple[str, str, str], optional
        Pre-resolved ``(src, relation, dst)`` edge type for the down-projection.
    data_node_name : str, default "data"
        Name of the data nodes in ``graph``.
    autocast : bool, default False
        Whether to use automatic mixed precision for the projections.
    row_normalize : bool, optional
        Normalize projection weights per target node so each row sums to 1.
    truncation_up_file_path : str, optional
        Deprecated path to an ``.npz`` file for the up-projection matrix.
    truncation_down_file_path : str, optional
        Deprecated path to an ``.npz`` file for the down-projection matrix.

    Examples
    --------
    >>> # Graph-based path (edge names supplied by ProjectionCreator)
    >>> conn = TruncatedConnection(
    ...     graph=graph,
    ...     data_node_name="data",
    ...     truncation_down_edges_name=("data", "to", "truncation"),
    ...     truncation_up_edges_name=("truncation", "to", "data"),
    ...     edge_weight_attribute="gauss_weight",
    ... )
    >>> x = torch.randn(2, 4, 1, 40192, 44)  # (batch, time, ens, nodes, features)
    >>> out = conn(x)
    >>> print(out.shape)
    torch.Size([2, 4, 1, 40192, 44])

    >>> # File-based path
    >>> conn = TruncatedConnection(
    ...     truncation_down_file_path="n320_to_o96.npz",
    ...     truncation_up_file_path="o96_to_n320.npz",
    ... )
    >>> x = torch.randn(2, 4, 1, 40192, 44)
    >>> out = conn(x)
    >>> print(out.shape)
    torch.Size([2, 4, 1, 40192, 44])
    """

    def __init__(
        self,
        graph: Optional[HeteroData] = None,
        src_node_weight_attribute: Optional[str] = None,
        edge_weight_attribute: Optional[str] = None,
        truncation_config: Optional[dict] = None,
        truncation_up_edges_name: Optional[tuple[str, str, str]] = None,
        truncation_down_edges_name: Optional[tuple[str, str, str]] = None,
        data_node_name: str = "data",
        autocast: bool = False,
        row_normalize: bool = False,
        # Deprecated: pass inside truncation_config instead.
        truncation_up_file_path: Optional[str] = None,
        truncation_down_file_path: Optional[str] = None,
        **_,
    ) -> None:
        super().__init__()

        truncation_config = self._normalise_truncation_config(
            truncation_config,
            truncation_up_file_path,
            truncation_down_file_path,
        )

        if truncation_config is not None:
            up_path = truncation_config.get("truncation_up_file_path")
            down_path = truncation_config.get("truncation_down_file_path")
            has_file = up_path is not None or down_path is not None
            from anemoi.models.schemas.residual import TruncationConfigOnTheFlySchema

            onthefly_keys = set(TruncationConfigOnTheFlySchema.model_fields)
            has_onthefly = bool(set(truncation_config) & onthefly_keys)
            if has_file and has_onthefly:
                msg = "truncation_config mixes file-based and on-the-fly keys. Use one mode only."
                raise ValueError(msg)
            if up_path is not None and down_path is not None:
                truncation_up_file_path = up_path
                truncation_down_file_path = down_path
            else:
                from anemoi.graphs.builders import build_truncation_subgraph

                graph = build_truncation_subgraph(graph, data_node_name, truncation_config)
                truncation_down_edges_name = (data_node_name, "to", "truncation")
                truncation_up_edges_name = ("truncation", "to", data_node_name)

        _edge_weight_attr = (
            edge_weight_attribute if edge_weight_attribute is not None else DEFAULT_EDGE_WEIGHT_ATTRIBUTE
        )

        up_edges, down_edges = self._resolve_edges(
            graph=graph,
            truncation_up_file_path=truncation_up_file_path,
            truncation_down_file_path=truncation_down_file_path,
            truncation_up_edges_name=truncation_up_edges_name,
            truncation_down_edges_name=truncation_down_edges_name,
        )

        self.provider_down = ProjectionGraphProvider(
            graph=graph,
            edges_name=down_edges,
            edge_weight_attribute=_edge_weight_attr,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_down_file_path,
            row_normalize=row_normalize,
        )

        self.provider_up = ProjectionGraphProvider(
            graph=graph,
            edges_name=up_edges,
            edge_weight_attribute=_edge_weight_attr,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_up_file_path,
            row_normalize=row_normalize,
        )

        self.projector = SparseProjector(autocast=autocast)

    @staticmethod
    def _normalise_truncation_config(
        truncation_config: Optional[dict],
        truncation_up_file_path: Optional[str],
        truncation_down_file_path: Optional[str],
    ) -> Optional[dict]:
        """Forward deprecated top-level file-path kwargs into truncation_config."""
        has_files = truncation_up_file_path is not None or truncation_down_file_path is not None
        if not has_files:
            return truncation_config
        import logging

        logging.getLogger(__name__).warning(
            "Passing 'truncation_up_file_path' / 'truncation_down_file_path' as top-level kwargs "
            "is deprecated. Move them inside 'truncation_config' instead."
        )
        cfg = dict(truncation_config) if truncation_config is not None else {}
        if truncation_up_file_path is not None:
            cfg.setdefault("truncation_up_file_path", truncation_up_file_path)
        if truncation_down_file_path is not None:
            cfg.setdefault("truncation_down_file_path", truncation_down_file_path)
        return cfg

    @staticmethod
    def _resolve_edges(
        *,
        graph: HeteroData | None,
        truncation_up_file_path: str | None,
        truncation_down_file_path: str | None,
        truncation_up_edges_name: tuple[str, str, str] | None,
        truncation_down_edges_name: tuple[str, str, str] | None,
    ) -> tuple[tuple[str, str, str] | None, tuple[str, str, str] | None]:
        """Validate and return the (up, down) edge tuples."""
        files_specified = truncation_up_file_path is not None and truncation_down_file_path is not None
        if files_specified:
            assert (
                truncation_up_edges_name is None and truncation_down_edges_name is None
            ), "Specify either file paths or edge names for truncation, not both."
            return None, None

        assert graph is not None, "graph must be provided when file paths are not specified."
        assert (
            truncation_up_edges_name is not None and truncation_down_edges_name is not None
        ), "Both truncation_up_edges_name and truncation_down_edges_name must be provided."
        up_edges = tuple(truncation_up_edges_name)
        down_edges = tuple(truncation_down_edges_name)
        assert up_edges in graph.edge_types, f"Graph must contain edges {up_edges} for up-projection."
        assert down_edges in graph.edge_types, f"Graph must contain edges {down_edges} for down-projection."
        return up_edges, down_edges

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_sizes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        """Apply truncated skip connection."""
        batch_size = x.shape[0]
        x = x[:, -1, ...]  # pick latest step

        x = einops.rearrange(x, "batch ensemble grid features -> (batch ensemble) grid features")
        channel_shard_sizes = get_shard_sizes(x, -1, model_comm_group)
        if grid_shard_sizes is not None:  # grids sharding -> channel sharding
            x = all_to_all_transpose(x, -1, channel_shard_sizes, -2, grid_shard_sizes, model_comm_group)
        x = self.projector(x, self.provider_down.get_edges(device=x.device))
        x = self.projector(x, self.provider_up.get_edges(device=x.device))
        if grid_shard_sizes is not None:  # channel sharding -> grid sharding
            x = all_to_all_transpose(x, -2, grid_shard_sizes, -1, channel_shard_sizes, model_comm_group)
        x = einops.rearrange(x, "(batch ensemble) grid features -> batch ensemble grid features", batch=batch_size)

        return self._expand_time(x, n_step_output)


def _ornstein_init_theta(
    theta_init: float,
    theta_buff: float,
    statistics: dict,
) -> np.ndarray:
    """Best-guess initialization of theta from per-variable tendency statistics.

    If ``theta_init`` is zero and both ``stdev`` and ``stdev_tend`` are present
    in the statistics dict, falls back to ``0.5 * (stdev_tend / stdev) ** 2``.
    The returned value is reparameterized into the (theta_buff, 1) interval and
    clipped to (0.01, 0.99) for numerical stability before being inverted into
    sigmoid-space.
    """
    statistics = statistics or {}
    if theta_init == 0 and {"stdev", "stdev_tend"}.issubset(statistics):
        theta_init = 0.5 * (statistics["stdev_tend"] / statistics["stdev"]) ** 2

    theta_init = (np.asarray(theta_init) - theta_buff) / (1 - theta_buff)
    theta_init = np.where(theta_init < 1, theta_init, 0.99)
    theta_init = np.where(theta_init > 0, theta_init, 0.01)
    return theta_init


def _grid_shape_from_graph(graph: HeteroData, dataset_name: str) -> tuple[int, int]:
    """Derive (nlat, nlon) from the unique latitude/longitude coordinates of a node set."""
    assert graph is not None and dataset_name is not None, (
        "Ornstein residuals need both `graph` and `dataset_name` to derive nlat/nlon. "
        "These are passed by `BaseGraphModel._build_residual`."
    )
    node_x = graph[dataset_name].x
    nlat = int(torch.unique(node_x[:, 0]).numel())
    nlon = int(torch.unique(node_x[:, 1]).numel())
    return nlat, nlon


def _slice_statistics_to_prognostic(statistics: dict | None, data_indices) -> dict:
    if not statistics:
        return {}
    idx = data_indices.data.input.prognostic
    return {k: v[idx] for k, v in statistics.items() if hasattr(v, "__getitem__")}


class ScalarOrnsteinConnection(BaseResidualConnection):
    """Ornstein residual with learnable scalars theta and mu.

    ``residual(x) = (1 - theta) * x + mu + sum_i beta_i * f_i``

    where theta is in (theta_buff, 1) and learned independently for each
    prognostic variable. ``f_i`` are forcing variables listed in
    ``regressors`` No spatial or spectral structure.

    Parameters
    ----------
    theta_init : float
        Initial value for theta. If 0 and statistics are available, auto-initialized
        from tendency statistics.
    theta_buff : float
        Lower bound buffer for theta. Theta is constrained to (theta_buff, 1).
    theta_train : bool
        Whether theta is a trainable parameter.
    regressors : list[str] | None
        Variable names to use as regressors.
    """

    def __init__(
        self,
        theta_init: float = 0.00,
        theta_buff: float = 0.00,
        theta_train: bool = True,
        regressors: list[str] | None = None,
        graph: HeteroData | None = None,
        statistics: dict | None = None,
        data_indices=None,
        dataset_name: str | None = None,
        **_,
    ) -> None:
        super().__init__()
        regressors = regressors or []
        assert data_indices is not None, "ScalarOrnsteinConnection needs `data_indices`."

        self._internal_input_idx = list(data_indices.model.input.prognostic)
        variables = data_indices.model.input.name_to_index
        self._regressors_input_idx = [variables[f] for f in regressors]

        sliced_stats = _slice_statistics_to_prognostic(statistics, data_indices)
        theta = _ornstein_init_theta(theta_init, theta_buff, sliced_stats)
        theta = np.log(theta / (1 - theta))

        weight = torch.zeros(len(regressors) + 2, len(self._internal_input_idx))
        weight[0, :] = torch.from_numpy(np.broadcast_to(theta, weight[0, :].shape).copy())

        self.weight = Parameter(weight, theta_train)
        self.theta_buff = theta_buff

    def _learnable(self, x_last: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        gain = 1 - torch.sigmoid(weight[0, :]) * (1 - self.theta_buff) - self.theta_buff
        out = gain * x_last[..., self._internal_input_idx] + weight[1, :]
        for i, k in enumerate(self._regressors_input_idx):
            out = out + weight[i + 2, :] * x_last[..., k].unsqueeze(-1)
        return out

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_sizes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        x_last = x[:, -1, ...]
        out = torch.zeros_like(x_last)
        out[..., self._internal_input_idx] = self._learnable(x_last)
        return self._expand_time(out, n_step_output)


class SpectralOrnsteinConnection(BaseResidualConnection):
    """Ornstein residual with learnable spatially-varying theta and mu defined via spherical harmonics.

    ``residual(x) = (1 - theta(s)) * x + mu(s) + sum_i beta_i(s) * f_i``

    where theta/mu/beta_i are stored as
    ``lmax x lmax`` complex SH coefficients (per prognostic variable), and the
    spatial fields are obtained via inverse SHT. ``f_i`` are forcing variables
    listed in ``regressors``.

    When ``truncate=True``, a learnable spectral low-pass filter is applied to
    the input fields before computing the residual. This truncates high-frequency
    content from the skip connection.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree for the theta/mu coefficients.
    grid : str
        Grid type: ``"regular"`` for regular lat-lon, ``"octahedral"`` for
        octahedral reduced grids. Other types are not currently supported and will raise an error.
    theta_init : float
        Initial value for theta.
    theta_buff : float
        Lower bound buffer for theta.
    use_mean : bool
        Whether to include a the mean (mu) term.
    regressors : list[str] | None
        Variable names to use as spatially-varying regressors.
    truncate : bool
        If True, apply a learnable spectral low-pass filter to the input fields.
    skip_truncate_variables : list[str] | None
        Variable names to exclude from spectral truncation (only used when
        ``truncate=True``).
    anti_aliasing : bool
        If True (and ``truncate=True``), use anti-aliasing blending in the filter.
    """

    def __init__(
        self,
        lmax: int = 2,
        grid: str = "regular",
        theta_init: float = 0.00,
        theta_buff: float = 0.00,
        use_mean: bool = True,
        regressors: list[str] | None = None,
        truncate: bool = False,
        skip_truncate_variables: list[str] | None = None,
        anti_aliasing: bool = True,
        graph: HeteroData | None = None,
        statistics: dict | None = None,
        data_indices=None,
        dataset_name: str | None = None,
        **_,
    ) -> None:
        super().__init__()
        regressors = regressors or []
        assert data_indices is not None, "SpectralOrnsteinConnection needs `data_indices`."

        self._internal_input_idx = list(data_indices.model.input.prognostic)
        variables = data_indices.model.input.name_to_index
        self._regressors_input_idx = [variables[f] for f in regressors]

        self.nlat, self.nlon = _grid_shape_from_graph(graph, dataset_name)

        sliced_stats = _slice_statistics_to_prognostic(statistics, data_indices)
        theta = _ornstein_init_theta(theta_init, theta_buff, sliced_stats)
        theta = 4 * np.pi * np.log(theta / (1 - theta))

        weight = torch.zeros(len(regressors) + 2, len(self._internal_input_idx), lmax, lmax, 2)
        weight[0, :, 0, 0, 0] = torch.from_numpy(np.broadcast_to(theta, weight[0, :, 0, 0, 0].shape).copy())
        self.weight = Parameter(weight)

        if grid == "octahedral":
            self.isht = InverseOctahedralSHT(self.nlat, truncation=lmax - 1)
        elif grid == "regular":
            self.isht = InverseRegularSHT(self.nlat, truncation=lmax - 1)
        else:
            raise ValueError(f"Unsupported grid type {grid!r}. Supported types: ['octahedral', 'regular']")

        muzero = torch.ones_like(weight)
        muzero[1, :, :, :, :] = 1.0 if use_mean else 0.0
        self.register_buffer("muzero", muzero)
        self.theta_buff = theta_buff

        # Spectral truncation (low-pass filtering) of input fields
        self.truncate = truncate
        if truncate:
            self._init_truncation(grid, lmax, theta_init, anti_aliasing, skip_truncate_variables or [], variables)

    def _init_truncation(self, grid, lmax, theta_init, anti_aliasing, skip_truncate_variables, variables):
        """Initialize spectral truncation parameters."""
        if grid == "octahedral":
            oct_lons = [20 + 4 * i for i in range(self.nlat // 2)]
            oct_lons += list(reversed(oct_lons))
            trunc = self.nlat - 1
            self.x_fsht = SphericalHarmonicTransform(lons_per_lat=oct_lons, truncation=trunc)
            self.x_isht = InverseSphericalHarmonicTransform(lons_per_lat=oct_lons, truncation=trunc)
        elif grid == "regular":
            reg_lons = [self.nlon] * self.nlat
            trunc = self.nlat - 1
            self.x_fsht = SphericalHarmonicTransform(lons_per_lat=reg_lons, truncation=trunc)
            self.x_isht = InverseSphericalHarmonicTransform(lons_per_lat=reg_lons, truncation=trunc)
        else:
            raise ValueError(f"Unsupported grid type {grid!r}.")

        skip_idx = {variables[v] for v in skip_truncate_variables if v in variables}
        self._truncation_input_idx = [int(idx) for idx in self._internal_input_idx if idx not in skip_idx]

        blur_lmax = self.x_fsht.truncation + 1

        filt = torch.ones(len(self._truncation_input_idx), blur_lmax)
        filt = filt * max(theta_init, 0.01) / (0.5 - max(theta_init, 0.01))
        filt = torch.sqrt(filt / blur_lmax)

        walias = torch.zeros(len(self._truncation_input_idx), lmax, lmax, 2)

        self.filter = Parameter(filt)
        self.walias = Parameter(walias)

        self.lpass_filter = self._truncate_with_anti_aliasing if anti_aliasing else self._truncate_without_anti_aliasing

    def _x_filter(self) -> torch.Tensor:
        f = torch.square(self.filter)
        f = torch.cumsum(f, -1)
        return f / (1 + f)

    def _w_filter(self) -> torch.Tensor:
        walias = self.isht(torch.view_as_complex(self.walias))
        return torch.sigmoid(walias)

    def _truncate_without_anti_aliasing(self, x: torch.Tensor) -> torch.Tensor:
        x = self.x_fsht(x)
        f = self._x_filter()
        x = x * (1 - f.unsqueeze(-1))
        return self.x_isht(x)

    def _truncate_with_anti_aliasing(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = self.x_fsht(x)
        f = self._x_filter()
        walias = self._w_filter()

        x_skip = x_skip * (1 - f.unsqueeze(-1))
        return walias * x + (1 - walias) * self.x_isht(x_skip)

    def _apply_truncation(self, x_last: torch.Tensor) -> torch.Tensor:
        x_last = einops.rearrange(x_last, "... values var -> ... var values")
        x_last[..., self._truncation_input_idx, :] = self.lpass_filter(x_last[..., self._truncation_input_idx, :])
        return einops.rearrange(x_last, "... var values -> ... values var")

    def _learnable(self, x_last: torch.Tensor) -> torch.Tensor:
        if self.truncate:
            x_last = self._apply_truncation(x_last)

        weight = self.isht(torch.view_as_complex(self.weight * self.muzero))
        weight = einops.rearrange(weight, "... var values -> ... values var")

        gain = 1 - torch.sigmoid(weight[0, ...]) * (1 - self.theta_buff) - self.theta_buff
        out = gain * x_last[..., self._internal_input_idx] + weight[1, ...]
        for i, k in enumerate(self._regressors_input_idx):
            out = out + weight[i + 2, ...] * x_last[..., k].unsqueeze(-1)
        return out

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_sizes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        x_last = x[:, -1, ...]
        out = torch.zeros_like(x_last)
        out[..., self._internal_input_idx] = self._learnable(x_last)
        return self._expand_time(out, n_step_output)
