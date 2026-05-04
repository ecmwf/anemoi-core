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

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
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
        grid_shard_shapes=None,
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
        grid_shard_shapes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        """Return the last timestep of the input sequence."""
        x_skip = x[:, self.step, ...]  # x shape: (batch, time, ens, nodes, features)
        return self._expand_time(x_skip, n_step_output)


class TruncatedConnection(BaseResidualConnection):
    """Truncated skip connection

    This connection applies a coarse-graining and reconstruction of input features using sparse
    projections to truncate high frequency features.

    This module uses two projection operators: one to map features from the full-resolution
    grid to a truncated (coarse) grid, and another to project back to the original resolution.

    Parameters
    ----------
    graph : HeteroData, optional
        The graph containing the subgraphs for down and up projections.
    data_nodes : str, optional
        Name of the nodes representing the data nodes.
    truncation_nodes : str, optional
        Name of the nodes representing the truncated (coarse) nodes.
    edge_weight_attribute : str, optional
        Name of the edge attribute to use as weights for the projections.
    src_node_weight_attribute : str, optional
        Name of the source node attribute to use as weights for the projections.
    autocast : bool, default False
        Whether to use automatic mixed precision for the projections.
    truncation_up_file_path : str, optional
        File path (.npz) to load the up-projection matrix from.
    truncation_down_file_path : str, optional
        File path (.npz) to load the down-projection matrix from.
    row_normalize : bool, optional
        Whether to normalize weights per row (target node) so each row sums to 1

    Example
    -------
    >>> from torch_geometric.data import HeteroData
    >>> import torch
    >>> # Assume graph is a HeteroData object with the required edges and node types
    >>> graph = HeteroData()
    >>> # ...populate graph with nodes and edges for 'data' and 'int'...
    >>> # Example creating the projection matrices from the graph
    >>> conn = TruncatedConnection(
    ...     graph=graph,
    ...     data_nodes="data",
    ...     truncation_nodes="int",
    ...     edge_weight_attribute="gauss_weight",
    ... )
    >>> x = torch.randn(2, 4, 1, 40192, 44)  # (batch, time, ens, nodes, features)
    >>> out = conn(x)
    >>> print(out.shape)
    torch.Size([2, 4, 1, 40192, 44])

    >>> # Example specifying .npz files for projection matrices
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
        data_nodes: Optional[str] = None,
        truncation_nodes: Optional[str] = None,
        edge_weight_attribute: Optional[str] = None,
        src_node_weight_attribute: Optional[str] = None,
        truncation_up_file_path: Optional[str] = None,
        truncation_down_file_path: Optional[str] = None,
        autocast: bool = False,
        row_normalize: bool = False,
        **_,
    ) -> None:
        super().__init__()
        up_edges, down_edges = self._get_edges_name(
            graph,
            data_nodes,
            truncation_nodes,
            truncation_up_file_path,
            truncation_down_file_path,
            edge_weight_attribute,
        )

        self.provider_down = ProjectionGraphProvider(
            graph=graph,
            edges_name=down_edges,
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_down_file_path,
            row_normalize=row_normalize,
        )

        self.provider_up = ProjectionGraphProvider(
            graph=graph,
            edges_name=up_edges,
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=truncation_up_file_path,
            row_normalize=row_normalize,
        )

        self.projector = SparseProjector(autocast=autocast)

    def _get_edges_name(
        self,
        graph,
        data_nodes,
        truncation_nodes,
        truncation_up_file_path,
        truncation_down_file_path,
        edge_weight_attribute,
    ):
        are_files_specified = truncation_up_file_path is not None and truncation_down_file_path is not None
        if not are_files_specified:
            assert graph is not None, "graph must be provided if file paths are not specified."
            assert data_nodes is not None, "data nodes name must be provided if file paths are not specified."
            assert (
                truncation_nodes is not None
            ), "truncation nodes name must be provided if file paths are not specified."
            up_edges = (truncation_nodes, "to", data_nodes)
            down_edges = (data_nodes, "to", truncation_nodes)
            assert up_edges in graph.edge_types, f"Graph must contain edges {up_edges} for up-projection."
            assert down_edges in graph.edge_types, f"Graph must contain edges {down_edges} for down-projection."
        else:
            assert (
                data_nodes is None or truncation_nodes is None or edge_weight_attribute is None
            ), "If file paths are specified, node and attribute names should not be provided."
            up_edges = down_edges = None  # Not used when loading from files
        return up_edges, down_edges

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_shapes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        """Apply truncated skip connection."""
        batch_size = x.shape[0]
        x = x[:, -1, ...]  # pick latest step
        shard_shapes = apply_shard_shapes(x, 0, grid_shard_shapes) if grid_shard_shapes is not None else None

        x = einops.rearrange(x, "batch ensemble grid features -> (batch ensemble) grid features")
        x = self._to_channel_shards(x, shard_shapes, model_comm_group)
        x = self.projector(x, self.provider_down.get_edges(device=x.device))
        x = self.projector(x, self.provider_up.get_edges(device=x.device))
        x = self._to_grid_shards(x, shard_shapes, model_comm_group)
        x = einops.rearrange(x, "(batch ensemble) grid features -> batch ensemble grid features", batch=batch_size)

        return self._expand_time(x, n_step_output)

    def _to_channel_shards(self, x, shard_shapes=None, model_comm_group=None):
        return self._reshard(x, shard_channels, shard_shapes, model_comm_group)

    def _to_grid_shards(self, x, shard_shapes=None, model_comm_group=None):
        return self._reshard(x, gather_channels, shard_shapes, model_comm_group)

    def _reshard(self, x, fn, shard_shapes=None, model_comm_group=None):
        if shard_shapes is not None:
            x = fn(x, shard_shapes, model_comm_group)
        return x


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
    """Mean-reverting residual with a single learnable scalar theta per prognostic variable.

    ``residual(x) = (1 - theta) * x`` where theta is in (theta_buff, 1) and learned
    independently for each prognostic variable. No spatial or spectral structure.

    Parameters
    ----------
    theta_init : float
        Initial value for theta. If 0 and statistics are available, auto-initialized
        from tendency statistics.
    theta_buff : float
        Lower bound buffer for theta. Theta is constrained to (theta_buff, 1).
    theta_train : bool
        Whether theta is a trainable parameter.
    """

    def __init__(
        self,
        theta_init: float = 0.00,
        theta_buff: float = 0.00,
        theta_train: bool = True,
        graph: HeteroData | None = None,
        statistics: dict | None = None,
        data_indices=None,
        dataset_name: str | None = None,
        **_,
    ) -> None:
        super().__init__()
        assert data_indices is not None, "ScalarOrnsteinConnection needs `data_indices`."
        self._internal_input_idx = list(data_indices.model.input.prognostic)

        sliced_stats = _slice_statistics_to_prognostic(statistics, data_indices)
        theta = _ornstein_init_theta(theta_init, theta_buff, sliced_stats)
        theta = np.log(theta / (1 - theta))

        weight = torch.zeros(len(self._internal_input_idx))
        weight[:] = torch.from_numpy(np.broadcast_to(theta, weight.shape).copy())

        self.weight = Parameter(weight, theta_train)
        self.theta_buff = theta_buff

    def _learnable(self, x_last: torch.Tensor) -> torch.Tensor:
        gain = 1 - torch.sigmoid(self.weight) * (1 - self.theta_buff) - self.theta_buff
        return gain * x_last[..., self._internal_input_idx]

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_shapes=None,
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
        Grid type: ``"legendre-gauss"`` for regular lat-lon, ``"octahedral"`` for
        octahedral reduced grids.
    theta_init : float
        Initial value for theta.
    theta_buff : float
        Lower bound buffer for theta.
    zmean_term : bool
        Whether to include a zonal mean (mu) term.
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
        grid: str = "legendre-gauss",
        theta_init: float = 0.00,
        theta_buff: float = 0.00,
        zmean_term: bool = True,
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
        theta = np.sqrt(4 * np.pi) * np.log(theta / (1 - theta))

        weight = torch.zeros(len(regressors) + 2, len(self._internal_input_idx), lmax, lmax, 2)
        weight[0, :, 0, 0, 0] = torch.from_numpy(np.broadcast_to(theta, weight[0, :, 0, 0, 0].shape).copy())
        self.weight = Parameter(weight)

        if grid == "octahedral":
            self.isht = InverseOctahedralSHT(self.nlat, truncation=lmax - 1)
        else:
            self.isht = InverseRegularSHT(self.nlat, truncation=lmax - 1)

        muzero = torch.ones_like(weight)
        muzero[1, :, :, :, :] = 1.0 if zmean_term else 0.0
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
        else:
            nlon = 2 * self.nlat
            reg_lons = [nlon] * self.nlat
            trunc = self.nlat - 1
            self.x_fsht = SphericalHarmonicTransform(lons_per_lat=reg_lons, truncation=trunc)
            self.x_isht = InverseSphericalHarmonicTransform(lons_per_lat=reg_lons, truncation=trunc)

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
        grid_shard_shapes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        x_last = x[:, -1, ...]
        out = torch.zeros_like(x_last)
        out[..., self._internal_input_idx] = self._learnable(x_last)
        return self._expand_time(out, n_step_output)
