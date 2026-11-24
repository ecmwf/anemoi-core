# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Target-only spatial smoothing of selected variables.

The smoother applies one frozen sparse matrix ``A`` (rows = output grid points,
columns = input grid points) to selected variables of the raw training batch, before
any normalisation (the smoother and ``clamp_max`` work in physical units). The matrix is held by a
:class:`~anemoi.models.layers.graph_provider.ProjectionGraphProvider`, built either
from an ``.npz`` file (e.g. exported by ``anemoi-graphs export_to_sparse``) or from the
weighted edges of the training graph, and applied with a
:class:`~anemoi.models.layers.sparse_projector.SparseProjector`. The smoother is a
plain object owned by the training module: it is not an ``nn.Module`` and is not
registered on the model, so it never enters the checkpoint and inference is
unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003  (needed at runtime by pydantic)
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from scipy import sparse

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.distributed import ProcessGroup
    from torch_geometric.data import HeteroData

    from anemoi.models.data import Source
    from anemoi.models.data import TensorLayout
    from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)

Normalisation = Literal["none", "row"]

DIAGONAL_TOLERANCE = 1e-6


class TargetSmoothingSchema(BaseModel):
    """Configuration of the target smoother of one dataset (``data.datasets.<name>.target_smoothing``).

    The matrix is given either by ``matrix_path`` or by ``edges_name`` (exactly one).
    """

    model_config = ConfigDict(extra="forbid")

    variables: list[str] = Field(min_length=1)
    "Names of the (non-input) variables to smooth, e.g. a diagnostic mask."
    matrix_path: Path | None = None
    "Path to a square ``.npz`` sparse matrix, e.g. exported by ``anemoi-graphs export_to_sparse``."
    edges_name: tuple[str, str, str] | None = None
    "Edge set ``(src, relation, dst)`` of the training graph used as matrix, e.g. ``[data, to, data]``."
    edge_weight_attribute: str | None = None
    "Edge attribute holding the weights of ``edges_name``, e.g. ``gauss_weight``."
    src_node_weight_attribute: str | None = None
    "Optional source node attribute multiplied into the edge weights."
    normalisation: Normalisation = "none"
    (
        "``none``: the matrix is used as given. With a weight 1 on the diagonal a unit impulse keeps the value 1 at "
        "its own grid point. ``row``: rows are rescaled to sum to 1 (local weighted mean, constant fields are "
        "preserved)."
    )
    clamp_max: float | None = Field(default=None, gt=0)
    "Optional upper bound applied after smoothing, e.g. 1 to cap overlapping blobs of a mask."

    @field_validator("variables")
    @classmethod
    def _unique_variables(cls, variables: list[str]) -> list[str]:
        if len(set(variables)) != len(variables):
            msg = f"target_smoothing.variables contains duplicates: {variables}"
            raise ValueError(msg)
        return variables

    @model_validator(mode="after")
    def _one_matrix_source(self) -> TargetSmoothingSchema:
        if (self.matrix_path is None) == (self.edges_name is None):
            msg = "target_smoothing needs exactly one of 'matrix_path' or 'edges_name'."
            raise ValueError(msg)
        if self.matrix_path is not None and (self.edge_weight_attribute or self.src_node_weight_attribute):
            msg = "'edge_weight_attribute' and 'src_node_weight_attribute' only apply with 'edges_name'."
            raise ValueError(msg)
        if self.edges_name is not None and self.edge_weight_attribute is None:
            msg = "target_smoothing with 'edges_name' needs 'edge_weight_attribute'."
            raise ValueError(msg)
        return self


class TargetSmoothing:
    """Smooth selected variables of a training batch with a frozen sparse matrix.

    The smoother owns its projection provider, which is always asked for a float32 matrix on the batch device.

    Parameters
    ----------
    variables : list[str]
        Variables to smooth. They must exist in the dataset and must not be model inputs
        (prognostic or forcing), so that inputs are never altered.
    matrix_path : str | Path, optional
        ``.npz`` sparse matrix with rows = output points and columns = input points.
    normalisation : {"none", "row"}
        ``none`` uses the matrix as given (a warning is logged if its diagonal is not 1),
        ``row`` rescales the rows to sum to 1.
    clamp_max : float, optional
        Upper bound applied to the smoothed values.
    edges_name : tuple[str, str, str], optional
        Edge set of ``graph_data`` to use as matrix instead of ``matrix_path``.
    edge_weight_attribute : str, optional
        Edge attribute with the weights of ``edges_name``.
    src_node_weight_attribute : str, optional
        Source node attribute multiplied into the edge weights.
    graph_data : HeteroData, optional
        Training graph, needed with ``edges_name``.
    """

    def __init__(
        self,
        variables: list[str],
        matrix_path: str | Path | None = None,
        normalisation: Normalisation = "none",
        clamp_max: float | None = None,
        *,
        edges_name: tuple[str, str, str] | list[str] | None = None,
        edge_weight_attribute: str | None = None,
        src_node_weight_attribute: str | None = None,
        graph_data: HeteroData | None = None,
    ) -> None:
        if normalisation not in ("none", "row"):
            msg = f"Unknown normalisation {normalisation!r}, expected 'none' or 'row'."
            raise ValueError(msg)
        self.variables = list(variables)
        self.normalisation = normalisation
        self.clamp_max = clamp_max
        self._resolved = False

        if edges_name is not None:
            edges_name = _check_graph_edges(graph_data, edges_name, edge_weight_attribute, src_node_weight_attribute)
        self._provider = ProjectionGraphProvider(
            graph=graph_data if edges_name is not None else None,
            edges_name=edges_name,
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=matrix_path,
            row_normalize=normalisation == "row",
            warn_if_not_row_normalized=False,
        )
        self._projector = SparseProjector()

        matrix = self._provider.projection_matrix
        if matrix.shape[0] != matrix.shape[1]:
            msg = f"The smoothing matrix must be square (same input and output grid), got shape {tuple(matrix.shape)}."
            raise ValueError(msg)
        self.grid_size = matrix.shape[0]
        if normalisation == "none":
            _warn_if_diagonal_is_not_one(matrix)
        LOGGER.info(
            "Target smoothing of %s: %d x %d matrix, nnz=%d, normalisation=%s",
            self.variables,
            *matrix.shape,
            matrix._nnz(),
            normalisation,
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping | TargetSmoothingSchema,
        graph_data: HeteroData | None = None,
    ) -> TargetSmoothing:
        """Build from a ``target_smoothing`` config block (mapping, DictConfig or schema).

        ``graph_data`` is needed when the block uses ``edges_name``.
        """
        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config, TargetSmoothingSchema):
            config = TargetSmoothingSchema(**config)
        return cls(
            variables=config.variables,
            matrix_path=config.matrix_path,
            normalisation=config.normalisation,
            clamp_max=config.clamp_max,
            edges_name=config.edges_name,
            edge_weight_attribute=config.edge_weight_attribute,
            src_node_weight_attribute=config.src_node_weight_attribute,
            graph_data=graph_data,
        )

    def resolve_variables(self, data_indices: IndexCollection) -> None:
        """Validate the variable names against the dataset: they must exist and must not be model inputs."""
        name_to_index = data_indices.name_to_index
        unknown = [name for name in self.variables if name not in name_to_index]
        if unknown:
            msg = f"target_smoothing.variables {unknown} are not dataset variables. Available: {list(name_to_index)}."
            raise ValueError(msg)

        model_inputs = set(data_indices.prognostic) | set(data_indices.forcing)
        inputs = [name for name in self.variables if name in model_inputs]
        if inputs:
            msg = (
                f"target_smoothing.variables {inputs} are model inputs (prognostic or forcing). Only diagnostic or "
                "target variables can be smoothed, otherwise the model inputs would be altered too."
            )
            raise ValueError(msg)

        self._resolved = True

    def __call__(self, source: Source, model_comm_group: ProcessGroup | None = None) -> Source:
        """Return ``source`` with the selected variables smoothed.

        Parameters
        ----------
        source : Source
            Gridded dataset view holding raw (not normalised) values. The grid may be sharded over
            ``model_comm_group`` (see ``source.grid_shard_sizes``); it is gathered for the smoothing and
            sharded again. The input is not modified.
        model_comm_group : ProcessGroup, optional
            Process group over which the grid is sharded.

        Returns
        -------
        Source
            A new view where the selected variables are smoothed (and clamped), NaN points are left as NaN
            and no NaN is spread into finite points. All other variables are unchanged.
        """
        if not self._resolved:
            msg = "resolve_variables must be called before applying the smoother."
            raise RuntimeError(msg)
        if source.is_tabular:
            msg = f"target_smoothing needs a gridded dataset, but '{source.name}' is tabular."
            raise ValueError(msg)
        missing = [name for name in self.variables if name not in source.name_to_index]
        if missing:
            msg = f"target_smoothing.variables {missing} are not in the batch of '{source.name}': {source.variables}."
            raise ValueError(msg)

        return source.apply_func(
            self._smooth_values,
            layout=source.layout,
            grid_shard_sizes=source.grid_shard_sizes,
            model_comm_group=model_comm_group,
        )

    def _smooth_values(
        self,
        values: torch.Tensor,
        *,
        name_to_index: Mapping[str, int],
        layout: TensorLayout,
        grid_shard_sizes: list[int] | None,
        model_comm_group: ProcessGroup | None,
        **_: object,
    ) -> torch.Tensor:
        """Smooth the selected variables of the dense tensor ``values`` in place and return it."""
        var_dim = layout.axis("variables", ndim=values.ndim)
        grid_dim = layout.axis("grid", ndim=values.ndim)
        indices = torch.tensor([name_to_index[name] for name in self.variables], device=values.device)

        selected = values.index_select(var_dim, indices)
        if grid_shard_sizes is not None:
            selected = gather_tensor(selected, grid_dim, grid_shard_sizes, model_comm_group)
        # The projector takes (..., grid, variables).
        selected = selected.movedim((grid_dim, var_dim), (-2, -1))
        if selected.shape[-2] != self.grid_size:
            msg = (
                f"The smoothing matrix has {self.grid_size} grid points but the batch has {selected.shape[-2]}. "
                "Check that matrix_path was built for this dataset grid."
            )
            raise ValueError(msg)

        selected = selected.to(torch.float32)
        is_nan = torch.isnan(selected)
        matrix = self._provider.get_edges(device=selected.device, dtype=torch.float32)
        smoothed = self._projector(selected.masked_fill(is_nan, 0.0), matrix)
        if self.clamp_max is not None:
            smoothed = smoothed.clamp(max=self.clamp_max)
        smoothed = smoothed.masked_fill(is_nan, float("nan")).movedim((-2, -1), (grid_dim, var_dim))

        if grid_shard_sizes is not None:
            smoothed = shard_tensor(smoothed, grid_dim, grid_shard_sizes, model_comm_group)
        return values.index_copy_(var_dim, indices, smoothed.to(values.dtype))


def _check_graph_edges(
    graph_data: HeteroData | None,
    edges_name: tuple[str, str, str] | list[str],
    edge_weight_attribute: str | None,
    src_node_weight_attribute: str | None,
) -> tuple[str, str, str]:
    """Check that the edge set, its attributes and its node sets exist, and return the canonical edge key."""
    edges_name = tuple(edges_name)
    if graph_data is None:
        msg = f"target_smoothing with edges_name={list(edges_name)} needs the training graph, but none was given."
        raise ValueError(msg)
    if edges_name not in graph_data.edge_types:
        msg = f"target_smoothing.edges_name {list(edges_name)} is not in the graph. Available: {graph_data.edge_types}."
        raise ValueError(msg)

    src, _, dst = edges_name
    if graph_data[src].num_nodes != graph_data[dst].num_nodes:
        msg = (
            f"target_smoothing.edges_name {list(edges_name)} must connect nodes of the same grid, but '{src}' has "
            f"{graph_data[src].num_nodes} and '{dst}' has {graph_data[dst].num_nodes} nodes."
        )
        raise ValueError(msg)

    if edge_weight_attribute is not None and edge_weight_attribute not in graph_data[edges_name].keys():  # noqa: SIM118
        msg = (
            f"target_smoothing.edge_weight_attribute '{edge_weight_attribute}' is not an attribute of the edges "
            f"{list(edges_name)}. Available: {list(graph_data[edges_name].keys())}."
        )
        raise ValueError(msg)
    if src_node_weight_attribute is not None and src_node_weight_attribute not in graph_data[src].keys():  # noqa: SIM118
        msg = (
            f"target_smoothing.src_node_weight_attribute '{src_node_weight_attribute}' is not an attribute of the "
            f"nodes '{src}'. Available: {list(graph_data[src].keys())}."
        )
        raise ValueError(msg)
    return edges_name


def _warn_if_diagonal_is_not_one(matrix: torch.Tensor) -> None:
    """Warn when the diagonal of a CSR matrix is not 1, as then a unit impulse does not keep its peak value 1."""
    diagonal = sparse.csr_matrix(
        (matrix.values().numpy(), matrix.col_indices().numpy(), matrix.crow_indices().numpy()),
        shape=tuple(matrix.shape),
    ).diagonal()
    if not np.allclose(diagonal, 1.0, rtol=0.0, atol=DIAGONAL_TOLERANCE):
        LOGGER.warning(
            "Target smoothing with normalisation='none': the diagonal of the matrix is not 1 (min=%.6g, max=%.6g), "
            "so a unit impulse will not keep the peak value 1. Use a matrix with weight 1 on the diagonal "
            "(e.g. GaussianDistanceWeights with norm: null) or normalisation='row' for a local weighted mean.",
            diagonal.min(),
            diagonal.max(),
        )


def build_target_smoothers(
    data_config: Mapping,
    data_indices: Mapping[str, IndexCollection],
    grid_sizes: Mapping[str, int] | None = None,
    graph_data: HeteroData | None = None,
) -> dict[str, TargetSmoothing]:
    """Build the target smoothers of all datasets that configure ``target_smoothing``.

    Parameters
    ----------
    data_config : Mapping
        Per-dataset data configs (``data.datasets``).
    data_indices : Mapping[str, IndexCollection]
        Data indices per dataset, used to validate the variable names.
    grid_sizes : Mapping[str, int], optional
        Number of grid points of each dataset on which the smoother is applied, to validate the matrix.
    graph_data : HeteroData, optional
        Training graph, needed by configs that build the matrix from ``edges_name``.
    """
    smoothers = {}
    for dataset_name, dataset_config in data_config.items():
        if isinstance(dataset_config, dict):
            smoothing_config = dataset_config.get("target_smoothing")
        else:
            smoothing_config = getattr(dataset_config, "target_smoothing", None)
        if smoothing_config is None:
            continue
        smoother = TargetSmoothing.from_config(smoothing_config, graph_data=graph_data)
        smoother.resolve_variables(data_indices[dataset_name])
        if grid_sizes is not None and grid_sizes[dataset_name] != smoother.grid_size:
            msg = (
                f"datasets.{dataset_name}.target_smoothing matrix has {smoother.grid_size} grid points, "
                f"but the dataset grid has {grid_sizes[dataset_name]}."
            )
            raise ValueError(msg)
        smoothers[dataset_name] = smoother
    return smoothers
