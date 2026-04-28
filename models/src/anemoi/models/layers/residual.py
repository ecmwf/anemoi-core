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
from anemoi.models.layers.sht import CartesianInverseRealSHT
from anemoi.models.layers.sht import CartesianRealSHT
from anemoi.models.layers.sht import OctahedralInverseRealSHT
from anemoi.models.layers.sht import OctahedralRealSHT
from anemoi.models.layers.sparse_projector import SparseProjector


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


class NoResidual(BaseResidualConnection):
    """No residual connection. Equivalent to ``x(t+1) = model(x(t))``."""

    def __init__(
        self,
        graph: HeteroData | None = None,
        data_indices=None,
        dataset_name: str | None = None,
        **_,
    ) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_shapes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        out = torch.zeros_like(x[:, -1, ...])
        return self._expand_time(out, n_step_output)


class SimpleOrnsteinResidual(BaseResidualConnection):
    """Mean-reverting residual with a single scalar theta per prognostic variable.

    ``residual(x) = (1 - theta) * x`` where theta is in (theta_buff, 1) and learned
    independently for each prognostic variable. No spatial or spectral structure.
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
        assert data_indices is not None, "SimpleOrnsteinResidual needs `data_indices`."
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


class BasicOrnsteinResidual(BaseResidualConnection):
    """Ornstein residual with theta and mu defined as smooth (bandwidth-limited) functions
    on the sphere via spherical harmonic coefficients.

    ``residual(x) = (1 - theta(s)) * x + mu(s) + sum_i beta_i(s) * f_i``

    where s is a node on the sphere, theta/mu/beta_i are stored as
    ``lmax x lmax`` complex SH coefficients (per prognostic variable), and the
    spatial fields are obtained via inverse SHT. ``f_i`` are forcing variables
    listed in ``regressors``.
    """

    def __init__(
        self,
        lmax: int = 2,
        grid: str = "legendre-gauss",
        node_order: str = "lat-lon",
        theta_init: float = 0.00,
        theta_buff: float = 0.00,
        zmean_term: bool = True,
        regressors: list[str] | None = None,
        graph: HeteroData | None = None,
        statistics: dict | None = None,
        data_indices=None,
        dataset_name: str | None = None,
        **_,
    ) -> None:
        super().__init__()
        regressors = regressors or []
        assert data_indices is not None, "BasicOrnsteinResidual needs `data_indices`."

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
            self.isht = OctahedralInverseRealSHT(self.nlat, lmax, lmax)
            self.values_reshape_for = "... values var -> ... var values"
            self.values_reshape_inv = "... var values -> ... values var"
            self.kwargs_reshape_for: dict = {}
        else:
            self.isht = CartesianInverseRealSHT(self.nlat, self.nlon, lmax, grid)
            order = node_order.replace("-", " ")
            self.values_reshape_for = f"... ({order}) var -> ... var lat lon"
            self.values_reshape_inv = f"... var lat lon -> ... ({order}) var"
            self.kwargs_reshape_for = {"lat": self.nlat, "lon": self.nlon}

        muzero = torch.ones_like(weight)
        muzero[1, :, :, :, :] = 1.0 if zmean_term else 0.0
        self.register_buffer("muzero", muzero)
        self.theta_buff = theta_buff

    def _learnable(self, x_last: torch.Tensor) -> torch.Tensor:
        # weight: (R+2, V_orn, lmax, lmax, 2) -> (R+2, V_orn, ...spatial...) -> (R+2, ...spatial..., V_orn)
        weight = self.isht(torch.view_as_complex(self.weight * self.muzero))
        weight = einops.rearrange(weight, self.values_reshape_inv)

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


class CompleteOrnsteinResidual(BasicOrnsteinResidual):
    """``BasicOrnsteinResidual`` preceded by a learnable spherical low-pass filter
    on the prognostic fields.

    The residual smooths each input field so that only gross features are
    inherited from the previous step, and the fine details are reconstructed
    from scratch by the model.
    """

    def __init__(
        self,
        lmax: int = 2,
        grid: str = "legendre-gauss",
        node_order: str = "lat-lon",
        theta_init: float = 0.05,
        theta_buff: float = 0.00,
        zmean_term: bool = True,
        regressors: list[str] | None = None,
        anti_aliasing: bool = True,
        skip_blur: list[str] | None = None,
        graph: HeteroData | None = None,
        statistics: dict | None = None,
        data_indices=None,
        dataset_name: str | None = None,
        **_,
    ) -> None:
        super().__init__(
            lmax=lmax,
            grid=grid,
            node_order=node_order,
            theta_init=theta_init,
            theta_buff=theta_buff,
            zmean_term=zmean_term,
            regressors=regressors,
            graph=graph,
            statistics=statistics,
            data_indices=data_indices,
            dataset_name=dataset_name,
        )

        skip_blur = skip_blur or []
        variables = data_indices.model.input.name_to_index

        if grid == "octahedral":
            self.x_fsht = OctahedralRealSHT(self.nlat)
            self.x_isht = OctahedralInverseRealSHT(self.nlat, self.x_fsht.lmax, self.x_fsht.mmax)
        else:
            self.x_fsht = CartesianRealSHT(self.nlat, self.nlon, grid)
            self.x_isht = CartesianInverseRealSHT(self.nlat, self.nlon, self.x_fsht.lmax, grid)

        skip_idx = {variables[v] for v in skip_blur if v in variables}
        self._blurring_input_idx = [int(idx) for idx in self._internal_input_idx if idx not in skip_idx]

        # Tail of slice(None)s along the spatial axes (1 for octahedral, 2 for cartesian).
        self._var_axis_tail = (slice(None),) * max(len(self.kwargs_reshape_for), 1)

        filt = torch.ones(len(self._blurring_input_idx), self.x_fsht.lmax)
        filt = filt * max(theta_init, 0.01) / (0.5 - max(theta_init, 0.01))
        filt = torch.sqrt(filt / self.x_fsht.lmax)

        walias = torch.zeros(len(self._blurring_input_idx), lmax, lmax, 2)

        self.filter = Parameter(filt)
        self.walias = Parameter(walias)

        self.lpass_filter = self.blur_with_anti_aliasing if anti_aliasing else self.blur_without_anti_aliasing

    def x_filter(self) -> torch.Tensor:
        f = torch.square(self.filter)
        f = torch.cumsum(f, -1)
        return f / (1 + f)

    def w_filter(self) -> torch.Tensor:
        walias = self.isht(torch.view_as_complex(self.walias))
        return torch.sigmoid(walias)

    def blur_without_anti_aliasing(self, x_blur: torch.Tensor) -> torch.Tensor:
        x_blur = self.x_fsht(x_blur)
        f = self.x_filter()
        x_blur = x_blur * (1 - f.unsqueeze(-1))
        return self.x_isht(x_blur)

    def blur_with_anti_aliasing(self, x_blur: torch.Tensor) -> torch.Tensor:
        x_skip = self.x_fsht(x_blur)
        f = self.x_filter()
        walias = self.w_filter()

        x_skip = x_skip * (1 - f.unsqueeze(-1))
        return walias * x_blur + (1 - walias) * self.x_isht(x_skip)

    def blurring(self, x_last: torch.Tensor) -> torch.Tensor:
        x_last = einops.rearrange(x_last, self.values_reshape_for, **self.kwargs_reshape_for)
        x_last[..., self._blurring_input_idx, *self._var_axis_tail] = self.lpass_filter(
            x_last[..., self._blurring_input_idx, *self._var_axis_tail]
        )
        return einops.rearrange(x_last, self.values_reshape_inv)

    def _learnable(self, x_last: torch.Tensor) -> torch.Tensor:
        return super()._learnable(self.blurring(x_last))
