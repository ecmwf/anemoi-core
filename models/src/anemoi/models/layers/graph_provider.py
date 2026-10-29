# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import copy
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse import load_npz
from scipy.sparse import spmatrix
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj

from anemoi.models.distributed.khop_edges import shard_edges_1hop
from anemoi.models.distributed.khop_edges import sort_edge_index_by_dst
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.projection_backend import PROJECTION_BACKENDS
from anemoi.models.layers.projection_backend import ProjectionBackend

LOGGER = logging.getLogger(__name__)

# Entries converted per step in csr_to_table; bounds the temporary memory.
_TABLE_BAND_ENTRIES = 2**24
# Largest padded table we build (16 GiB).
_MAX_TABLE_ENTRIES = 2**31


def csr_to_table(matrix: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Convert a CSR matrix into tables padded to the longest row.

    Returns the column indices (``int32``) and values (``float32``), both ``(rows, longest row)``,
    and the row lengths (``int32``). Padding is column 0 with weight 0.

    Parameters
    ----------
    matrix : Tensor
        Sparse CSR matrix of shape ``(rows, columns)``.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Columns, weights and row lengths, on the device of ``matrix``.

    Raises
    ------
    ValueError
        If the table would have more than ``_MAX_TABLE_ENTRIES`` entries.
    """
    row_starts = matrix.crow_indices().to(torch.int64)
    columns = matrix.col_indices()
    values = matrix.values()
    num_rows = matrix.shape[0]
    row_lengths = row_starts.diff()
    width = max(1, int(row_lengths.max())) if num_rows else 1

    if num_rows * width > _MAX_TABLE_ENTRIES:
        raise ValueError(
            f"The projection matrix is too large for the triton backend: its {num_rows} rows padded to the "
            f"longest row of {width} entries would make a table of {num_rows * width} entries, more than "
            f"{_MAX_TABLE_ENTRIES}. Use the 'torch' backend for this matrix."
        )

    neighbours = torch.zeros(num_rows, width, dtype=torch.int32, device=columns.device)
    weights = torch.zeros(num_rows, width, dtype=torch.float32, device=values.device)
    # Scatter the entries into their slots, a band of rows at a time.
    rows_per_band = max(1, _TABLE_BAND_ENTRIES // width)
    for start in range(0, num_rows, rows_per_band):
        stop = min(num_rows, start + rows_per_band)
        first, last = int(row_starts[start]), int(row_starts[stop])
        lengths = row_lengths[start:stop]
        rows = torch.repeat_interleave(torch.arange(start, stop, device=columns.device), lengths)
        slots = torch.arange(first, last, device=columns.device) - torch.repeat_interleave(
            row_starts[start:stop], lengths
        )
        neighbours[rows, slots] = columns[first:last].to(torch.int32)
        weights[rows, slots] = values[first:last].to(torch.float32)
    return neighbours, weights, row_lengths.to(torch.int32)


def create_graph_provider(
    graph: Optional[HeteroData] = None,
    edge_attributes: Optional[list[str]] = None,
    src_size: Optional[int] = None,
    dst_size: Optional[int] = None,
    trainable_size: int = 0,
) -> "BaseGraphProvider":
    """Factory function to create appropriate graph provider.

    Returns StaticGraphProvider if graph has edges,
    otherwise returns NoOpGraphProvider for edge-less architectures.

    Parameters
    ----------
    graph : HeteroData, optional
        Graph containing edges (for static mode)
    edge_attributes : list[str], optional
        Edge attributes to use (for static mode)
    src_size : int, optional
        Source grid size (for static mode)
    dst_size : int, optional
        Destination grid size (for static mode)
    trainable_size : int, optional
        Trainable tensor size, by default 0

    Returns
    -------
    BaseGraphProvider
        Appropriate graph provider instance
    """
    if graph:
        return StaticGraphProvider(
            graph=graph,
            edge_attributes=edge_attributes,
            src_size=src_size,
            dst_size=dst_size,
            trainable_size=trainable_size,
        )
    else:
        return NoOpGraphProvider()


def normalize_projection_edges_name(
    edges_name: tuple[str, str, str] | list[str] | None,
) -> tuple[str, str, str]:
    """Coerce a projection ``edges_name`` to the canonical PyG edge key ``(src, "to", dst)``.

    Only the explicit 3-element form is accepted; YAML yields a list, which is returned as a
    tuple (PyG's ``HeteroData`` requires a tuple key). Any other shape raises ``ValueError``.
    """
    if not (isinstance(edges_name, (list, tuple)) and len(edges_name) == 3):
        raise ValueError(f"edges_name must be a (src, 'to', dst) triple, got {edges_name!r}")
    return tuple(edges_name)


class BaseGraphProvider(nn.Module, ABC):
    """Base class for graph edge providers.

    Graph providers encapsulate the logic for supplying edge indices and attributes
    to mapper and processor layers. This allows for different strategies (static, dynamic, etc.).
    """

    @abstractmethod
    def get_edges(
        self,
        batch_size: Optional[int] = None,
        src_coords: Optional[Tensor] = None,
        dst_coords: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        shard_edges: bool = True,
    ) -> Union[tuple[Tensor, Adj, Optional[ShardSizes]], Tensor]:
        """Get edge information.

        Parameters
        ----------
        batch_size : int, optional
            Number of times to expand the edge index (used by static mode)
        src_coords : Tensor, optional
            Source node coordinates (used by dynamic mode for k-NN, radius graphs, etc.)
        dst_coords : Tensor, optional
            Destination node coordinates (used by dynamic mode for k-NN, radius graphs, etc.)
        model_comm_group : ProcessGroup, optional
            Model communication group
        shard_edges : bool, optional
            Whether to shard edges, by default True

        Returns
        -------
        Union[tuple[Tensor, Adj, Optional[ShardSizes]], Tensor]
            For standard providers: (edge_attr, edge_index, edge_shard_sizes) tuple
            For sparse providers: sparse projection matrix
        """
        pass

    @property
    @abstractmethod
    def edge_dim(self) -> int:
        """Return the edge dimension."""
        pass

    @property
    def is_sparse(self) -> bool:
        """Whether this provider returns sparse matrices."""
        return False


class StaticGraphProvider(BaseGraphProvider):
    """Provider for static graphs with fixed edge structure.

    This provider owns all graph-related state including edge attributes,
    edge indices, and trainable parameters.
    """

    # info on trainable layout versioning for migration:
    _TRAINABLE_LAYOUT_VERSION = 1
    _TRAINABLE_LAYOUT_VERSION_KEY = "trainable_layout_version"

    def __init__(
        self,
        graph: HeteroData,
        edge_attributes: list[str],
        src_size: int,
        dst_size: int,
        trainable_size: int,
    ) -> None:
        """Initialize StaticGraphProvider.

        Parameters
        ----------
        graph : HeteroData
            Graph containing edges
        edge_attributes : list[str]
            Edge attributes to use
        src_size : int
            Source grid size
        dst_size : int
            Destination grid size
        trainable_size : int
            Size of trainable edge parameters
        """
        super().__init__()

        assert graph, "StaticGraphProvider needs a valid graph to register edges."
        assert edge_attributes is not None, "Edge attributes must be provided"

        # sort all edge indices by dst at this stage to avoid expensive reordering operations later:
        edge_index, perm = sort_edge_index_by_dst(graph.edge_index, max_value=dst_size)
        edge_attr_tensor = torch.cat([graph[attr] for attr in edge_attributes], axis=1)
        edge_attr_tensor = edge_attr_tensor.index_select(0, perm)

        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("edge_attr", edge_attr_tensor, persistent=False)
        self.register_buffer("edge_index_base", edge_index, persistent=False)
        self.register_buffer(
            "edge_inc", torch.from_numpy(np.asarray([[src_size], [dst_size]], dtype=np.int64)), persistent=False
        )
        self.register_buffer(
            self._TRAINABLE_LAYOUT_VERSION_KEY,
            torch.tensor(self._TRAINABLE_LAYOUT_VERSION, dtype=torch.int64),
            persistent=True,
        )

        self.trainable = TrainableTensor(trainable_size=trainable_size, tensor_size=edge_attr_tensor.shape[0])

        self._edge_dim = edge_attr_tensor.shape[1] + trainable_size

    @property
    def edge_dim(self) -> int:
        """Return the edge dimension."""
        return self._edge_dim

    def _expand_edges(self, edge_index: Adj, edge_inc: Tensor, batch_size: int) -> Adj:
        """Expand edge index.

        Parameters
        ----------
        edge_index : Adj
            Edge index to start
        edge_inc : Tensor
            Edge increment to use
        batch_size : int
            Number of times to expand the edge index

        Returns
        -------
        Adj
            Expanded edge index
        """
        edge_index = torch.cat(
            [edge_index + i * edge_inc for i in range(batch_size)],
            dim=1,
        )
        return edge_index

    def _get_edges_impl(
        self,
        batch_size: int,
        shard_edges: bool,
        model_comm_group: Optional[ProcessGroup],
    ) -> tuple[Tensor, Adj, Optional[ShardSizes]]:
        """Implementation of get_edges."""
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)

        if shard_edges:
            src_size, dst_size = self.edge_inc[:, 0].tolist()
            edge_attr, edge_index, edge_shard_sizes = shard_edges_1hop(
                edge_attr,
                edge_index,
                src_size * batch_size,
                dst_size * batch_size,
                model_comm_group,
            )
            return edge_attr, edge_index, edge_shard_sizes

        return edge_attr, edge_index, None

    def get_edges(
        self,
        batch_size: int,
        src_coords: Optional[Tensor] = None,
        dst_coords: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        shard_edges: bool = True,
        act_checkpoint: bool = True,
    ) -> tuple[Tensor, Adj, Optional[ShardSizes]]:
        """Get edge attributes and expanded edge index for static graph.

        Parameters
        ----------
        batch_size : int
            Number of times to expand the edge index
        src_coords : Tensor, optional
            Source node coordinates (ignored for static graphs)
        dst_coords : Tensor, optional
            Destination node coordinates (ignored for static graphs)
        model_comm_group : ProcessGroup, optional
            Model communication group
        shard_edges : bool, optional
            Whether to shard edges, by default True.
        act_checkpoint : bool, optional
            Whether to use gradient checkpointing, by default True.

        Returns
        -------
        tuple[Tensor, Adj, Optional[ShardSizes]]
            Edge attributes, expanded edge index, and optional edge_shard_sizes.
            edge_shard_sizes is a list of per-rank partition sizes when shard_edges=True,
            otherwise None.
        """
        if act_checkpoint:
            return checkpoint(self._get_edges_impl, batch_size, shard_edges, model_comm_group, use_reentrant=False)
        return self._get_edges_impl(batch_size, shard_edges, model_comm_group)


class NoOpGraphProvider(BaseGraphProvider):
    """Provider for edge-less architectures (e.g., Transformers).

    Returns None for edges and has edge_dim=0. Used when the mapper/processor
    does not require graph structure (e.g., pure attention-based models).
    """

    def __init__(self) -> None:
        """Initialize NoOpGraphProvider."""
        super().__init__()

    @property
    def edge_dim(self) -> int:
        """Return the edge dimension (0 for no edges)."""
        return 0

    def get_edges(
        self,
        batch_size: Optional[int] = None,
        src_coords: Optional[Tensor] = None,
        dst_coords: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        shard_edges: bool = True,
    ) -> tuple[None, None, None]:
        """Return None for edge attributes, edge index, and edge_shard_sizes.

        Parameters
        ----------
        batch_size : int, optional
            Unused
        src_coords : Tensor, optional
            Unused
        dst_coords : Tensor, optional
            Unused
        model_comm_group : ProcessGroup, optional
            Unused
        shard_edges : bool, optional
            Unused

        Returns
        -------
        tuple[None, None, None]
            No edges
        """
        return None, None, None


class DynamicGraphProvider(BaseGraphProvider):
    """Provider for dynamic graphs where edges are supplied at runtime.

    Does not support trainable edge parameters.

    Future implementation will support on-the-fly graph construction via build_graph()
    (e.g., k-NN graphs, radius graphs, adaptive connectivity).
    """

    def __init__(self, edge_dim: int) -> None:
        """Initialize DynamicGraphProvider.

        Parameters
        ----------
        edge_dim : int
            Expected dimension of edge attributes
        """
        super().__init__()
        self._edge_dim = edge_dim

    @property
    def edge_dim(self) -> int:
        """Return the edge dimension."""
        return self._edge_dim

    def build_graph(self, src_nodes: Tensor, dst_nodes: Tensor, **kwargs) -> tuple[Tensor, Adj]:
        """Build graph dynamically from source and destination nodes.

        This method will be implemented in the future to support on-the-fly
        graph construction (e.g., k-NN graphs, radius graphs, etc.).

        Parameters
        ----------
        src_nodes : Tensor
            Source node features/positions
        dst_nodes : Tensor
            Destination node features/positions
        **kwargs
            Additional parameters for graph construction algorithm

        Returns
        -------
        tuple[Tensor, Adj]
            Edge attributes and edge index

        Raises
        ------
        NotImplementedError
            This functionality is not yet implemented
        """
        raise NotImplementedError("Dynamic graph construction is not yet implemented. ")

    def _get_edges_impl(
        self,
        src_coords: Tensor,
        dst_coords: Tensor,
        shard_edges: bool,
        model_comm_group: Optional[ProcessGroup],
    ) -> tuple[Tensor, Adj, Optional[ShardSizes]]:
        """Implementation of get_edges, separated for checkpointing."""
        # Build graph from coordinates
        edge_attr, edge_index = self.build_graph(src_coords, dst_coords)
        edge_index, perm = sort_edge_index_by_dst(edge_index, max_value=dst_coords.shape[0])
        edge_attr = edge_attr.index_select(0, perm)

        if shard_edges:
            edge_attr, edge_index, edge_shard_sizes = shard_edges_1hop(
                edge_attr, edge_index, src_coords.shape[0], dst_coords.shape[0], model_comm_group
            )
            return edge_attr, edge_index, edge_shard_sizes

        return edge_attr, edge_index, None

    def get_edges(
        self,
        batch_size: Optional[int] = None,
        src_coords: Optional[Tensor] = None,
        dst_coords: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        shard_edges: bool = True,
        act_checkpoint: bool = True,
    ) -> tuple[Tensor, Adj, Optional[ShardSizes]]:
        """Get dynamic edges constructed from node coordinates.

        Calls build_graph() to construct edges on-the-fly using k-NN, radius graphs, etc.

        Parameters
        ----------
        batch_size : int, optional
            Batch size (currently unused, reserved for future implementation)
        src_coords : Tensor, optional
            Source node coordinates
        dst_coords : Tensor, optional
            Destination node coordinates
        model_comm_group : ProcessGroup, optional
            Model communication group
        shard_edges : bool, optional
            Whether to shard edges, by default True
        act_checkpoint : bool, optional
            Whether to use gradient checkpointing, by default True.

        Returns
        -------
        tuple[Tensor, Adj, Optional[ShardSizes]]
            Edge attributes, edge index, and optional edge_shard_sizes.

        Raises
        ------
        ValueError
            If coordinates are not provided
        NotImplementedError
            If build_graph() is not yet implemented
        """
        if src_coords is None or dst_coords is None:
            raise ValueError("DynamicGraphProvider requires (src_coords, dst_coords) to construct edges.")

        if act_checkpoint:
            return checkpoint(
                self._get_edges_impl, src_coords, dst_coords, shard_edges, model_comm_group, use_reentrant=False
            )
        return self._get_edges_impl(src_coords, dst_coords, shard_edges, model_comm_group)


class ProjectionGraphProvider(BaseGraphProvider):
    """Provider for sparse projection matrices.

    Builds the matrix from a graph or a file and keeps only the form its backend needs:

    - ``"torch"``: CSR matrix (:meth:`get_edges`).
    - ``"triton"``: padded table (:meth:`get_table`), built once from the matrix.
    """

    def __init__(
        self,
        graph: Optional[HeteroData] = None,
        edges_name: Optional[tuple[str, str, str]] = None,
        edge_weight_attribute: Optional[str] = None,
        src_node_weight_attribute: Optional[str] = None,
        file_path: Optional[str | Path] = None,
        row_normalize: bool = False,
        edge_mask: Optional[Tensor] = None,
        backend: ProjectionBackend = "triton",
    ) -> None:
        """Initialize ProjectionGraphProvider.

        Parameters
        ----------
        graph : HeteroData, optional
            Graph containing edges for projection
        edges_name : tuple[str, str, str], optional
            Edge type identifier (src, relation, dst)
        edge_weight_attribute : str, optional
            Edge attribute name for weights
        src_node_weight_attribute : str, optional
            Source node attribute name for weights
        file_path : str | Path, optional
            Path to .npz file with projection matrix
        row_normalize : bool
            Whether to normalize weights per row (target node) so each row sums to 1
        edge_mask : Tensor, optional
            Boolean mask which selects graph edges before constructing the matrix
        backend : str
            ``"torch"`` keeps the CSR matrix, ``"triton"`` keeps the table made from it.
        """
        super().__init__()
        if backend not in PROJECTION_BACKENDS:
            raise ValueError(f"Projection backend must be one of {PROJECTION_BACKENDS}, got '{backend}'.")
        self.backend = backend
        self.projection_matrix: Optional[Tensor] = None
        self.projection_table: Optional[tuple[Tensor, Tensor, Tensor]] = None

        if (file_path is None and edges_name is None) or (file_path is not None and edges_name is not None):
            raise ValueError("Exactly one of file_path or edges_name must be provided.")

        if file_path is not None:
            if src_node_weight_attribute is not None:
                msg = f"Building ProjectionGraphProvider from file, so src_node_weight_attribute='{src_node_weight_attribute}' will be ignored."
                LOGGER.warning(msg)

            if edge_weight_attribute is not None:
                msg = f"Building ProjectionGraphProvider from file, so edge_weight_attribute='{edge_weight_attribute}' will be ignored."
                LOGGER.warning(msg)
            self._build_from_file(file_path, row_normalize)
        elif edges_name is not None:
            if graph is None:
                raise ValueError("graph must be provided when constructing a projection from edges_name.")
            self._build_from_graph(
                graph, edges_name, edge_weight_attribute, src_node_weight_attribute, edge_mask, row_normalize
            )

    def __deepcopy__(self, memo: dict) -> "ProjectionGraphProvider":
        """Deepcopy that shares the static projection matrix by reference.

        ``projection_matrix`` holds a sparse CSR tensor. Sparse CSR tensors cannot
        be deepcopied (``NotImplementedError: Cannot access storage of
        SparseCsrTensorImpl``), which breaks ``copy.deepcopy(pl_module.loss)``
        during validation/plotting. It is a constant lookup table, so sharing it
        by reference is safe and avoids the copy. ``projection_table`` is shared the same way.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        if self.projection_matrix is not None:
            memo[id(self.projection_matrix)] = self.projection_matrix
        if self.projection_table is not None:
            memo[id(self.projection_table)] = self.projection_table
        for key, value in self.__dict__.items():
            new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    def _build_from_file(self, file_path: str | Path, row_normalize: bool) -> None:
        """Load projection matrix from file."""
        self._create_csr_matrix_from_scipy(load_npz(file_path), row_normalize)

    def _build_from_graph(
        self,
        graph: HeteroData,
        edges_name: tuple[str, str, str],
        edge_weight_attribute: Optional[str],
        src_node_weight_attribute: Optional[str],
        edge_mask: Optional[Tensor],
        row_normalize: bool,
    ) -> None:
        """Build projection matrix from graph.

        The matrix is initially built in COO format
        and then converted to CSR format for efficient sparse operations.
        """
        sub_graph = graph[tuple(edges_name)]
        edge_index = sub_graph.edge_index

        if edge_weight_attribute:
            weights = sub_graph[edge_weight_attribute].reshape(-1)
        else:
            weights = torch.ones(edge_index.shape[1], device=edge_index.device)

        if src_node_weight_attribute:
            weights = weights * graph[edges_name[0]][src_node_weight_attribute].reshape(-1)[edge_index[0]]

        if edge_mask is not None:
            edge_index = edge_index[:, edge_mask]
            weights = weights[edge_mask]

        matrix = coo_matrix(
            (
                weights.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy(),
                (
                    edge_index[1].detach().cpu().contiguous().numpy(),
                    edge_index[0].detach().cpu().contiguous().numpy(),
                ),
            ),
            shape=(
                graph[edges_name[2]].num_nodes,  # dst_size (targets) = rows
                graph[edges_name[0]].num_nodes,  # src_size (sources) = cols
            ),
            dtype=np.float32,
        )
        self._create_csr_matrix_from_scipy(matrix, row_normalize)

    def _create_csr_matrix_from_scipy(self, matrix: spmatrix, row_normalize: bool) -> None:
        """Create sparse projection CSR matrix from a SciPy sparse matrix."""
        matrix = matrix.astype(np.float32, copy=False).tocsr()
        matrix.sum_duplicates()  # coalesce duplicate entries

        if row_normalize:
            matrix = self._row_normalize_matrix(matrix)

        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        if not np.allclose(row_sums, np.ones_like(row_sums), atol=1e-5):
            LOGGER.warning(
                "Projection matrix rows do not sum to 1 (min=%.4f, max=%.4f, mean=%.4f). "
                "This is unexpected; please check your matrix. "
                "Consider using pre-normalized weights or row_normalize=True.",
                row_sums.min().item(),
                row_sums.max().item(),
                row_sums.mean().item(),
            )

        csr = torch.sparse_csr_tensor(
            torch.from_numpy(matrix.indptr),
            torch.from_numpy(matrix.indices),
            torch.from_numpy(matrix.data),
            size=matrix.shape,
        )
        self.shape = tuple(matrix.shape)
        self._edge_dim = self.shape[1]
        if self.backend == "torch":
            self.projection_matrix = csr
            return

        self.projection_table = csr_to_table(csr)
        num_rows, width = self.projection_table[0].shape
        LOGGER.info(
            "Projection table: %d target nodes, %d entries per row (%.2f stored on average).",
            num_rows,
            width,
            matrix.nnz / max(1, num_rows),
        )

    @staticmethod
    def _row_normalize_matrix(matrix: spmatrix) -> spmatrix:
        """Normalize weights per row (target node) so each row sums to 1.

        Converts the input matrix to CSR format, computes the sum of each row, and divides each non-zero row by its sum. Rows that sum to zero remain unchanged.
        """
        matrix = matrix.tocsr(copy=True)
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        inv_row_sums = np.zeros_like(row_sums, dtype=np.float32)
        non_zero = row_sums != 0
        inv_row_sums[non_zero] = 1.0 / row_sums[non_zero]
        matrix = matrix.multiply(inv_row_sums[:, None])
        return matrix.tocsr()

    @property
    def edge_dim(self) -> int:
        """Return projection matrix shape."""
        return self._edge_dim

    @property
    def is_sparse(self) -> bool:
        """This provider returns sparse matrices."""
        return True

    def get_edges(
        self,
        batch_size: Optional[int] = None,
        src_coords: Optional[Tensor] = None,
        dst_coords: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        shard_edges: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Return the sparse projection matrix.

        Parameters
        ----------
        batch_size : int, optional
            Unused for sparse providers
        src_coords : Tensor, optional
            Unused for sparse providers
        dst_coords : Tensor, optional
            Unused for sparse providers
        model_comm_group : ProcessGroup, optional
            Unused for sparse providers
        shard_edges : bool, optional
            Unused for sparse providers
        device : torch.device, optional
            Target device for matrix
        dtype : torch.dtype, optional
            Target dtype for matrix

        Returns
        -------
        Tensor
            Sparse projection matrix
        """
        if self.projection_matrix is None:
            raise ValueError(
                "This projection was built for the triton backend and holds only its table, see get_table."
            )
        if device is not None or dtype is not None:
            # sparse tensors can't be registered as buffers with DDP, so materialize and retain them on demand
            # TODO(SL): Calling this with different dtypes at runtime can cause precision deterioration;
            # use a proper device/dtype cache that retains the canonical matrix.
            self.projection_matrix = self.projection_matrix.to(device=device, dtype=dtype)
        return self.projection_matrix

    def get_table(self, device: Optional[torch.device] = None) -> tuple[Tensor, Tensor, Tensor]:
        """Return the padded table (see :func:`csr_to_table`), moved to ``device`` if given.

        Parameters
        ----------
        device : torch.device, optional
            Device to move the tables to.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Source nodes, weights and row lengths.
        """
        if self.projection_table is None:
            raise ValueError(
                "This projection was built for the torch backend and holds only its matrix, see get_edges."
            )
        if device is not None and self.projection_table[0].device != torch.device(device):
            self.projection_table = tuple(table.to(device) for table in self.projection_table)
        return self.projection_table

    def use_torch_backend(self) -> None:
        """Switch to the torch backend, e.g. to run on a CPU.

        Rebuilds the CSR matrix from the padded table and drops the table.
        """
        if self.backend == "torch":
            return
        neighbours, weights, row_lengths = (table.cpu() for table in self.projection_table)
        stored = torch.arange(neighbours.shape[1])[None, :] < row_lengths[:, None]
        row_starts = torch.zeros(neighbours.shape[0] + 1, dtype=torch.int64)
        row_starts[1:] = row_lengths.to(torch.int64).cumsum(0)
        self.projection_matrix = torch.sparse_csr_tensor(
            row_starts, neighbours[stored].long(), weights[stored], size=self.shape
        )
        self.projection_table = None
        self.backend = "torch"

    @classmethod
    def from_config(
        cls,
        config: object,
        graph_data: Optional[HeteroData] = None,
        data_node_name: str = "data",
        backend: ProjectionBackend = "triton",
    ) -> Optional["ProjectionGraphProvider"]:
        """Create a provider from a config mapping, choosing the mode from the keys present.

        - ``matrix_path`` → file mode.
        - ``edges_name`` → edge mode (needs *graph_data*).
        - ``num_nearest_neighbours`` + ``grid``/``node_builder`` → target-grid mode,
          building a Gaussian-weighted KNN subgraph on the fly from ``sigma`` (needs
          *graph_data*).

        Returns ``None`` for an empty or ``None`` *config*, and raises ``ValueError`` on an
        ambiguous config or when *graph_data* is required but missing.
        """
        # --- normalise to plain dict ---
        if config is None:
            return None
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(config):
                config = OmegaConf.to_container(config, resolve=True)
        except ImportError:
            pass
        if not isinstance(config, dict):
            config = dict(config)
        if not config:
            return None

        has_matrix = "matrix_path" in config and config["matrix_path"] is not None
        has_edges = "edges_name" in config and config["edges_name"] is not None

        if has_matrix and has_edges:
            raise ValueError("projection config must specify at most one of 'matrix_path' or 'edges_name', not both")

        if has_matrix:
            return cls(
                file_path=config["matrix_path"],
                row_normalize=bool(config.get("row_normalize", False)),
                backend=backend,
            )

        if has_edges:
            if graph_data is None:
                raise ValueError("graph_data is required for projection mode 'edges'")
            return cls(
                graph=graph_data,
                edges_name=normalize_projection_edges_name(config["edges_name"]),
                edge_weight_attribute=config.get("edge_weight_attribute"),
                src_node_weight_attribute=config.get("src_node_weight_attribute"),
                row_normalize=bool(config.get("row_normalize", False)),
                backend=backend,
            )

        # target-grid mode: require its signal key here for a clear error, not a deep KeyError.
        if config.get("num_nearest_neighbours") is None:
            raise ValueError(
                "projection config must specify 'matrix_path', 'edges_name', or target-grid "
                "keys ('num_nearest_neighbours' with 'grid' or 'node_builder')"
            )
        if graph_data is None:
            raise ValueError("graph_data is required for projection mode 'target_grid'")

        from anemoi.graphs.builders import build_node_to_node_projection_subgraph
        from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE

        target_node_name = config.get("target_node_name", "target_grid")
        subgraph = build_node_to_node_projection_subgraph(graph_data, data_node_name, target_node_name, config)
        # The on-the-fly KNN subgraph carries Gaussian distance weights (derived from the
        # mandatory `sigma`) under DEFAULT_EDGE_WEIGHT_ATTRIBUTE. Consume them by default so
        # `sigma` actually takes effect; otherwise _build_from_graph falls back to uniform
        # weights and `sigma` is silently ignored. An explicit `edge_weight_attribute` wins.
        edge_weight_attribute = config.get("edge_weight_attribute")
        if edge_weight_attribute is None:
            edge_weight_attribute = DEFAULT_EDGE_WEIGHT_ATTRIBUTE
        return cls(
            graph=subgraph,
            edges_name=(data_node_name, "to", target_node_name),
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=config.get("src_node_weight_attribute"),
            row_normalize=bool(config.get("row_normalize", False)),
            backend=backend,
        )


def use_torch_projections(model: nn.Module) -> None:
    """Switch all projections in ``model`` to the torch backend, e.g. to run a checkpoint on a CPU."""
    for module in model.modules():
        if isinstance(module, ProjectionGraphProvider):
            module.use_torch_backend()


class NeighbourListProvider:
    """K nearest neighbours of every node of one node set, plus the nodes' unit vectors.

    ``neighbours`` is ``int32`` of shape ``(num_nodes, K)`` and includes the node itself;
    ``unit_vectors`` is ``float32`` of shape ``(num_nodes, 3)``. No weights are stored, so one
    provider serves any smoothing width. Deep copies share the tables.
    """

    def __init__(self, graph: HeteroData, nodes_name: str, num_neighbours: int) -> None:
        """Build the neighbour table of one node set.

        Parameters
        ----------
        graph : HeteroData
            Graph with the node coordinates.
        nodes_name : str
            Node set to build the neighbours for.
        num_neighbours : int
            Neighbours per node, the node itself included.
        """
        from anemoi.graphs.builders import build_neighbour_table

        num_nodes = graph[nodes_name].num_nodes
        if num_neighbours > num_nodes:
            raise ValueError(f"num_neighbours={num_neighbours} exceeds the {num_nodes} nodes of '{nodes_name}'.")

        self.neighbours = build_neighbour_table(graph, nodes_name, num_neighbours)

        latitude, longitude = graph[nodes_name].x.to(torch.float64).unbind(dim=1)
        self.unit_vectors = (
            torch.stack(
                [latitude.cos() * longitude.cos(), latitude.cos() * longitude.sin(), latitude.sin()],
                dim=1,
            )
            .to(torch.float32)
            .to(self.neighbours.device)
            .contiguous()
        )
        self.nodes_name = nodes_name
        self.num_neighbours = num_neighbours

    def __deepcopy__(self, memo: dict) -> "NeighbourListProvider":
        """Deep copy that shares the (constant) tables."""
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        memo[id(self.neighbours)] = self.neighbours
        memo[id(self.unit_vectors)] = self.unit_vectors
        for key, value in self.__dict__.items():
            new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    def get_neighbours(self, device: Optional[torch.device] = None) -> tuple[Tensor, Tensor]:
        """Neighbour table and unit vectors, moved to ``device`` if given."""
        if device is not None and self.neighbours.device != torch.device(device):
            self.neighbours = self.neighbours.to(device)
            self.unit_vectors = self.unit_vectors.to(device)
        return self.neighbours, self.unit_vectors

    def typical_reach(self, num_samples: int = 10_000) -> float:
        """Median distance (radians) to the farthest neighbour, over up to ``num_samples`` nodes."""
        from anemoi.models.layers.gaussian_smoothing import squared_neighbour_distances

        num_nodes = self.neighbours.shape[0]
        sample = torch.linspace(0, num_nodes - 1, min(num_samples, num_nodes), device=self.neighbours.device).long()
        farthest = squared_neighbour_distances(self.unit_vectors, self.neighbours, sample, torch.float64).amax(dim=1)
        return float(farthest.median().sqrt())
