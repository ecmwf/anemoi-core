# (C) Copyright 2024-2026 Anemoi contributors.
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
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
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


class NoSkipConnection(BaseResidualConnection):
    """No skip connection module.

    This layer returns zeros, effectively disabling the residual connection
    so the model output is not biased by the input.

    This is useful when input observations are sparse or contain missing values,
    as adding them back to the output can create artefacts.
    """

    def __init__(self, **_) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_sizes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        """Return zeros with the expected output shape."""
        x_skip = torch.zeros_like(x[:, -1, ...])  # x shape: (batch, time, ens, nodes, features)
        return self._expand_time(x_skip, n_step_output)


def _parse_normalize_methods(normalize_method) -> tuple[str, dict[str, str]]:
    """Parse normalize_method config into (default, {var_name: method}).

    Accepts the same dict structure as the preprocessor normalizer config:
    ``{"default": "mean-std", "min-max": ["var1", "var2"], ...}``.
    """
    if not normalize_method:
        return "mean-std", {}
    normalize_method = dict(normalize_method)
    default = normalize_method.get("default", "mean-std")
    methods: dict[str, str] = {}
    for key, value in normalize_method.items():
        if key == "default":
            continue
        if isinstance(value, (list, tuple)):
            for var in value:
                methods[str(var)] = key
        elif isinstance(value, str):
            methods[value] = key
    return default, methods


def _apply_climatology_normalization(
    climatology: np.ndarray,
    feature_names: list,
    statistics: dict,
    normalize_method,
    data_name_to_idx: dict,
) -> None:
    """Normalize climatology columns in-place using dataset statistics.

    Parameters
    ----------
    climatology : np.ndarray
        Shape ``(num_grid_points, num_features)``, modified in-place.
    feature_names : list
        ``feature_names[col]`` gives the variable name for column ``col``,
        or ``None`` to skip that column.
    statistics : dict
        Dataset statistics with keys ``"mean"``, ``"stdev"``, ``"minimum"``,
        ``"maximum"`` — each a 1-D array indexed by data input index.
    normalize_method : dict | None
        Method config with the same structure as the preprocessor normalizer:
        ``{"default": "mean-std", "min-max": ["var1"], ...}``.
    data_name_to_idx : dict
        Mapping from variable name to data input index (used to index statistics).
    """
    default_method, var_methods = _parse_normalize_methods(normalize_method)
    for col, var_name in enumerate(feature_names):
        if var_name is None or var_name not in data_name_to_idx:
            continue
        method = var_methods.get(var_name, default_method)
        if method == "none":
            continue
        d_idx = data_name_to_idx[var_name]
        col_data = climatology[:, col]
        if method == "mean-std":
            climatology[:, col] = (col_data - statistics["mean"][d_idx]) / statistics["stdev"][d_idx]
        elif method == "std":
            climatology[:, col] = col_data / statistics["stdev"][d_idx]
        elif method == "min-max":
            lo, hi = statistics["minimum"][d_idx], statistics["maximum"][d_idx]
            climatology[:, col] = (col_data - lo) / (hi - lo)
        elif method == "max":
            climatology[:, col] = col_data / statistics["maximum"][d_idx]
        else:
            raise ValueError(f"Unknown normalization method {method!r} for variable {var_name!r}.")


class ClimatologySkipConnection(BaseResidualConnection):
    """Skip connection that returns a fixed climatological field as the residual.

    Instead of returning zeros (NoSkipConnection) or the raw input (SkipConnection),
    this class returns pre-computed climatological values so the model predicts
    departures from climatology.

    The climatology is loaded from a NumPy ``.npz`` file whose keys are variable
    names and values are 1-D arrays of shape ``(num_grid_points,)``. Variables
    present in the file are slotted into the corresponding feature index; variables
    absent from the file default to zero (equivalent to NoSkipConnection for those
    channels).

    The loaded climatology is registered as a buffer so it is saved in the model
    checkpoint and the ``.npz`` file is **not** required at inference time.

    Parameters
    ----------
    climatology_path : str
        Path to a ``.npz`` file. Keys are variable names, values are 1-D numpy
        arrays of shape ``(num_grid_points,)`` in un-normalized or normalized space
        depending on ``normalize_climatology``.
    graph : HeteroData, optional
        The graph data (unused, accepted for the shared residual factory kwargs).
    data_indices : object, optional
        Index collection providing ``model.input.name_to_index``.
    missing_value : float, optional
        Sentinel value marking missing points when ``fill_missing_only=True``.
    fill_missing_only : bool, optional
        If ``True``, return the latest input with points equal to ``missing_value``
        replaced by climatology, instead of returning the climatology everywhere.
    group_variables : list, optional
        When instantiated inside :class:`PerVariableGroupResidual`, the group's
        variable names; the buffer is then built for the group's feature dimension.
    normalize_climatology : bool, optional
        If ``True``, normalize the loaded climatology using ``statistics`` before
        storing the buffer. If ``False`` (default), the npz values are used as-is
        and are assumed to already be in normalized space.
    normalize_method : dict | None, optional
        Normalization method config. Uses the same structure as the preprocessor
        normalizer config: ``{"default": "mean-std", "min-max": ["var1"], ...}``.
        Only used when ``normalize_climatology=True``. Defaults to ``mean-std``
        for all variables.
    statistics : dict | None, optional
        Dataset statistics dict with keys ``"mean"``, ``"stdev"``, ``"minimum"``,
        ``"maximum"``. Required when ``normalize_climatology=True``.
    """

    def __init__(
        self,
        climatology_path: str,
        graph: Optional[HeteroData] = None,
        data_indices=None,
        missing_value: float = 0.0,
        fill_missing_only: bool = False,
        group_variables: Optional[list] = None,
        normalize_climatology: bool = False,
        normalize_method=None,
        statistics: Optional[dict] = None,
        **_,
    ) -> None:
        super().__init__()
        assert (
            data_indices is not None
        ), "ClimatologySkipConnection requires data_indices to map variable names to feature indices."

        self._missing_value = missing_value
        self._fill_missing_only = fill_missing_only

        name_to_index = dict(data_indices.model.input.name_to_index)

        # If called from PerVariableGroupResidual, only build buffer for the group's variables
        if group_variables is not None:
            feature_names = list(group_variables)
        else:
            feature_names = list(name_to_index.keys())
        num_features = len(feature_names)

        clim_data = np.load(climatology_path)

        # Infer grid size from the first array in the file
        first_key = next(iter(clim_data.files))
        num_grid_points = clim_data[first_key].shape[0]

        # Build the climatology tensor: (grid, features)
        climatology = np.zeros((num_grid_points, num_features), dtype=np.float32)

        for feat_idx, var_name in enumerate(feature_names):
            if var_name in clim_data:
                arr = clim_data[var_name]
                assert arr.shape == (num_grid_points,), (
                    f"ClimatologySkipConnection: variable {var_name!r} has shape {arr.shape}, "
                    f"expected ({num_grid_points},)."
                )
                climatology[:, feat_idx] = arr.astype(np.float32)

        if normalize_climatology:
            assert statistics is not None, "ClimatologySkipConnection: normalize_climatology=True requires statistics."
            _apply_climatology_normalization(
                climatology,
                feature_names,
                statistics,
                normalize_method,
                dict(data_indices.data.input.name_to_index),
            )

        self.register_buffer("_climatology", torch.from_numpy(climatology))

    def _local_climatology(self, grid_shard_sizes, model_comm_group) -> torch.Tensor:
        """Return the climatology buffer, sliced to this rank's grid partition when sharded."""
        clim = self._climatology
        if grid_shard_sizes is None:
            return clim
        assert sum(grid_shard_sizes) == clim.shape[0], (
            f"ClimatologySkipConnection: shard sizes sum to {sum(grid_shard_sizes)} "
            f"but climatology covers {clim.shape[0]} grid points."
        )
        rank = dist.get_rank(group=model_comm_group)
        offset = sum(grid_shard_sizes[:rank])
        return clim.narrow(0, offset, grid_shard_sizes[rank])

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_sizes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        """Return climatology, or input with missing values filled by climatology if fill_missing_only=True."""
        batch, _time, ensemble, _grid, _features = x.shape
        clim = self._local_climatology(grid_shard_sizes, model_comm_group)
        # clim shape: (grid, features) -> broadcast to (batch, ensemble, grid, features)
        clim = clim.unsqueeze(0).unsqueeze(0).expand(batch, ensemble, -1, -1)
        if self._fill_missing_only:
            x_last = x[:, -1, ...]  # (batch, ensemble, grid, features)
            mask = x_last == self._missing_value
            x_skip = torch.where(mask, clim, x_last)
        else:
            x_skip = clim
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
    sparse_projector_num_chunks : int, default 1
        Number of chunks to use for sparse projection matmuls.
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
        sparse_projector_num_chunks: int = 1,
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

        self.projector = SparseProjector(autocast=autocast, num_chunks=sparse_projector_num_chunks)

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
    graph : HeteroData, optional
        Graph data (unused, accepted for the shared residual factory kwargs).
    statistics : dict, optional
        Dataset statistics used to auto-initialize theta.
    data_indices : object
        The dataset's data indices, used to resolve variable names. Required.
    dataset_name : str, optional
        Dataset name (unused, accepted for the shared residual factory kwargs).
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


def _instantiate_sub_residual(
    cfg,
    graph: Optional[HeteroData] = None,
    data_indices=None,
    dataset_name: str | None = None,
    statistics: dict | None = None,
    data_node_name: str | None = None,
    sparse_projector_num_chunks: int = 1,
    group_variables: Optional[list] = None,
) -> BaseResidualConnection:
    """Instantiate a sub-residual from either a config dict or a pre-built module.

    Composite residuals (e.g. PerVariableGroupResidual) are constructed with
    ``_recursive_=False`` so their nested sub-residual configs arrive here as
    ``DictConfig`` / ``dict``; we instantiate them explicitly, forwarding the
    shared kwargs that ``BaseGraphModel._build_residual`` provides.

    Parameters
    ----------
    cfg : dict | DictConfig | BaseResidualConnection
        Sub-residual config or pre-built module.
    graph : HeteroData, optional
        Graph data forwarded to the sub-residual.
    data_indices : object, optional
        Data indices forwarded to the sub-residual.
    dataset_name : str, optional
        Dataset name forwarded to the sub-residual.
    statistics : dict, optional
        Dataset statistics forwarded to the sub-residual.
    data_node_name : str, optional
        Data node name forwarded to the sub-residual.
    sparse_projector_num_chunks : int, optional
        Number of chunks for sparse projections, forwarded to the sub-residual.
    group_variables : list of str, optional
        If the sub-residual is being instantiated inside PerVariableGroupResidual,
        this is the list of variable names for the group. Forwarded to classes
        that need it (e.g. ClimatologySkipConnection) so they can build buffers
        matching the group's feature dimension rather than the full model dimension.
        Classes that don't accept this kwarg will ignore it via **_.

    Returns
    -------
    BaseResidualConnection
        The instantiated sub-residual module.
    """
    if isinstance(cfg, BaseResidualConnection):
        return cfg
    if isinstance(cfg, (dict, DictConfig)):
        return instantiate(
            cfg,
            _recursive_=False,
            graph=graph,
            data_indices=data_indices,
            dataset_name=dataset_name,
            statistics=statistics,
            data_node_name=data_node_name,
            sparse_projector_num_chunks=sparse_projector_num_chunks,
            group_variables=group_variables,
        )
    raise TypeError(
        f"Sub-residual must be a dict / DictConfig / BaseResidualConnection, got {type(cfg)!r}.",
    )


class PerVariableGroupResidual(BaseResidualConnection):
    """Residual that applies different sub-residuals to disjoint variable groups.

    Each group names a set of model-input variables and a residual module. The
    group's residual is applied to the sliced feature tensor, and results are
    scattered back into an output tensor whose feature dimension matches the
    model input. Variable positions not covered by any group are left at zero
    (they are unused downstream — ``_assemble_output`` only indexes the
    prognostic slice of ``x_skip``).

    Groups must be disjoint, and every prognostic variable must be covered by
    exactly one group (diagnostic/forcing variables may be omitted).

    Parameters
    ----------
    groups : list of dict
        Each entry has keys ``name`` (str, for error messages), ``variables``
        (list of model-input variable names), and ``residual`` (a sub-residual
        config or pre-built module).
    graph : HeteroData, optional
        Forwarded to sub-residuals.
    data_indices : object
        The dataset's data indices, used to resolve variable names. Required.
    dataset_name : str, optional
        Forwarded to sub-residuals.
    statistics : dict, optional
        Forwarded to sub-residuals.
    data_node_name : str, optional
        Forwarded to sub-residuals.
    sparse_projector_num_chunks : int, optional
        Forwarded to sub-residuals.

    Example
    -------
    >>> # yaml config:
    >>> # residual:
    >>> #   _target_: anemoi.models.layers.residual.PerVariableGroupResidual
    >>> #   groups:
    >>> #     - name: polar_sat
    >>> #       variables: [sat_ch1, sat_ch2]
    >>> #       residual:
    >>> #         _target_: anemoi.models.layers.residual.NoSkipConnection
    >>> #     - name: gridded_fields
    >>> #       variables: [z_500, t_850]
    >>> #       residual:
    >>> #         _target_: anemoi.models.layers.residual.SkipConnection
    """

    def __init__(
        self,
        groups,
        graph: Optional[HeteroData] = None,
        data_indices=None,
        dataset_name: str | None = None,
        statistics: dict | None = None,
        data_node_name: str | None = None,
        sparse_projector_num_chunks: int = 1,
        **_,
    ) -> None:
        super().__init__()
        assert data_indices is not None, (
            "PerVariableGroupResidual requires data_indices — check that _build_residual "
            "forwards data_indices to the residual instantiation."
        )

        name_to_index = dict(data_indices.model.input.name_to_index)
        prognostic_indices = [int(i) for i in data_indices.model.input.prognostic]
        prognostic_set = set(prognostic_indices)

        self.group_names: list[str] = []
        self.group_residuals = nn.ModuleList()
        seen: dict[int, str] = {}
        covered_prognostic: set[int] = set()

        for i, group in enumerate(groups):
            group = dict(group)  # tolerate DictConfig
            name = group.get("name", f"group_{i}")
            variables = list(group["variables"])
            sub_cfg = group["residual"]

            indices: list[int] = []
            for var_name in variables:
                if var_name not in name_to_index:
                    msg = (
                        f"PerVariableGroupResidual group {name!r}: variable {var_name!r} "
                        f"is not in model.input.name_to_index."
                    )
                    raise ValueError(msg)
                idx = int(name_to_index[var_name])
                if idx in seen:
                    msg = (
                        f"PerVariableGroupResidual: variable {var_name!r} (index {idx}) "
                        f"appears in groups {seen[idx]!r} and {name!r}; groups must be disjoint."
                    )
                    raise ValueError(msg)
                seen[idx] = name
                indices.append(idx)
                if idx in prognostic_set:
                    covered_prognostic.add(idx)

            sub_residual = _instantiate_sub_residual(
                sub_cfg,
                graph=graph,
                data_indices=data_indices,
                dataset_name=dataset_name,
                statistics=statistics,
                data_node_name=data_node_name,
                sparse_projector_num_chunks=sparse_projector_num_chunks,
                group_variables=variables,
            )
            self.group_names.append(name)
            self.group_residuals.append(sub_residual)
            # Persistent=False: indices are derived from config, not learned state.
            self.register_buffer(
                f"_group_indices_{i}",
                torch.as_tensor(indices, dtype=torch.long),
                persistent=False,
            )

        missing = prognostic_set - covered_prognostic
        if missing:
            # Reverse-lookup names for a helpful error message.
            index_to_name = {v: k for k, v in name_to_index.items()}
            missing_names = sorted(index_to_name.get(i, f"<idx {i}>") for i in missing)
            msg = (
                f"PerVariableGroupResidual: prognostic variables not covered by any group: "
                f"{missing_names}. Every prognostic variable must belong to exactly one group."
            )
            raise ValueError(msg)

    def _indices(self, i: int, device: torch.device) -> torch.Tensor:
        buf: torch.Tensor = getattr(self, f"_group_indices_{i}")
        return buf.to(device=device, non_blocking=True) if buf.device != device else buf

    def forward(
        self,
        x: torch.Tensor,
        grid_shard_sizes=None,
        model_comm_group=None,
        n_step_output: int | None = None,
    ) -> torch.Tensor:
        out: Optional[torch.Tensor] = None
        for i, sub_residual in enumerate(self.group_residuals):
            idx = self._indices(i, x.device)
            x_group = x.index_select(-1, idx)
            skip_group = sub_residual(
                x_group,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                n_step_output=n_step_output,
            )
            if out is None:
                out_shape = list(skip_group.shape)
                out_shape[-1] = x.shape[-1]
                out = torch.zeros(out_shape, dtype=skip_group.dtype, device=skip_group.device)
            out[..., idx] = skip_group
        assert out is not None, "PerVariableGroupResidual must have at least one group."
        return out


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
    graph : HeteroData, optional
        Graph data, used to derive the grid shape (nlat/nlon).
    statistics : dict, optional
        Dataset statistics used to auto-initialize theta.
    data_indices : object
        The dataset's data indices, used to resolve variable names. Required.
    dataset_name : str, optional
        Dataset name, used with ``graph`` to derive the grid shape.
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
