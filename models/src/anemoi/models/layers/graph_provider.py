# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Iterator
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader.dataloader import Collater as PyGCollater
from torch_geometric.typing import Adj

from anemoi.models.distributed.khop_edges import shard_edges_1hop
from anemoi.models.distributed.khop_edges import sort_edge_index_by_dst
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.layers.graph import TrainableTensor

LOGGER = logging.getLogger(__name__)


def create_graph_provider(
    graph: Optional[HeteroData] = None,
    edge_attributes: Optional[list[str]] = None,
    src_size: Optional[int] = None,
    dst_size: Optional[int] = None,
    trainable_size: int = 0,
    dataset_name: Optional[str] = None,
) -> "BaseGraphProvider":
    """Factory function to create appropriate graph provider.

    Returns StaticGraphProvider if graph has edges,
    Returns FileGraphProvider if graph_dir is provided, and
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
        if isinstance(graph, Path) and graph.is_dir():
            return FileGraphProvider(graph_dir=graph, src_size=src_size, dst_size=dst_size, dataset_name=dataset_name, edge_attributes=edge_attributes, trainable_size=trainable_size)
        else:
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

    Builds and stores sparse projection matrix from graph or file.
    """

    def __init__(
        self,
        graph: Optional[HeteroData] = None,
        edges_name: Optional[tuple[str, str, str]] = None,
        edge_weight_attribute: Optional[str] = None,
        src_node_weight_attribute: Optional[str] = None,
        file_path: Optional[str | Path] = None,
        row_normalize: bool = False,
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
        """
        super().__init__()

        if file_path is not None:
            if src_node_weight_attribute is not None:
                msg = f"Building ProjectionGraphProvider from file, so src_node_weight_attribute='{src_node_weight_attribute}' will be ignored."
                LOGGER.warning(msg)

            if edge_weight_attribute is not None:
                msg = f"Building ProjectionGraphProvider from file, so edge_weight_attribute='{edge_weight_attribute}' will be ignored."
                LOGGER.warning(msg)
            self._build_from_file(file_path, row_normalize)
        else:
            assert (
                graph is not None and edges_name is not None
            ), "Must provide graph and edges_name if file_path not given"
            self._build_from_graph(graph, edges_name, edge_weight_attribute, src_node_weight_attribute, row_normalize)

    def _build_from_file(self, file_path: str | Path, row_normalize: bool) -> None:
        """Load projection matrix from file."""
        from scipy.sparse import load_npz

        truncation_data = load_npz(file_path)
        edge_index = torch.tensor(np.vstack(truncation_data.nonzero()), dtype=torch.long)
        weights = torch.tensor(truncation_data.data, dtype=torch.float32)
        src_size, dst_size = truncation_data.shape

        self._create_matrix(edge_index, weights, src_size, dst_size, row_normalize)

    def _build_from_graph(
        self,
        graph: HeteroData,
        edges_name: tuple[str, str, str],
        edge_weight_attribute: Optional[str],
        src_node_weight_attribute: Optional[str],
        row_normalize: bool,
    ) -> None:
        """Build projection matrix from graph."""
        sub_graph = graph[edges_name]

        if edge_weight_attribute:
            weights = sub_graph[edge_weight_attribute].squeeze()
        else:
            weights = torch.ones(sub_graph.edge_index.shape[1], device=sub_graph.edge_index.device)

        if src_node_weight_attribute:
            weights *= graph[edges_name[0]][src_node_weight_attribute][sub_graph.edge_index[0]]

        # PyG convention: edge_index[0]=source, edge_index[1]=target
        # For M @ x, we need matrix shape (targets, sources) with:
        #   - row indices = targets
        #   - col indices = sources
        # -> swap edge_index to [targets, sources] for COO tensor
        edge_index_for_coo = torch.stack([sub_graph.edge_index[1], sub_graph.edge_index[0]])

        self._create_matrix(
            edge_index_for_coo,
            weights,
            graph[edges_name[2]].num_nodes,  # dst_size (targets) = rows
            graph[edges_name[0]].num_nodes,  # src_size (sources) = cols
            row_normalize,
        )

    def _create_matrix(
        self,
        edge_index: Tensor,
        weights: Tensor,
        src_size: int,
        dst_size: int,
        row_normalize: bool,
    ) -> None:
        """Create sparse projection matrix."""
        row_index = edge_index[0].long()
        edge_index = torch.stack([row_index, edge_index[1].long()])

        if row_normalize:
            weights = self._row_normalize_weights(edge_index, weights, src_size)

        self.projection_matrix = torch.sparse_coo_tensor(
            edge_index,
            weights,
            (src_size, dst_size),
            device=edge_index.device,
        ).coalesce()

        self._edge_dim = self.projection_matrix.shape[1]

        row_sums = torch.zeros(src_size, device=weights.device).scatter_add_(0, row_index, weights)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            LOGGER.warning(
                "Projection matrix rows do not sum to 1 (min=%.4f, max=%.4f, mean=%.4f). "
                "This is unexpected; please check your matrix. "
                "Consider using row_normalize=True or pre-normalized weights.",
                row_sums.min().item(),
                row_sums.max().item(),
                row_sums.mean().item(),
            )

    @staticmethod
    def _row_normalize_weights(edge_index: Tensor, weights: Tensor, num_rows: int) -> Tensor:
        """Normalize weights per row (target node) so each row sums to 1."""
        total = torch.zeros(num_rows, device=weights.device)
        row_index = edge_index[0].long()
        # edge_index[0] contains row indices (targets) for COO tensor format
        norm = total.scatter_add_(0, row_index, weights)
        norm = norm[row_index]
        return weights / (norm + 1e-8)

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

        Returns
        -------
        Tensor
            Sparse projection matrix
        """
        if device is not None:
            # sparse tensors can't be registered as buffers with ddp, so move on demand
            self.projection_matrix = self.projection_matrix.to(device)
        return self.projection_matrix
        @classmethod
    def from_config(
        cls,
        config: object,
        graph_data: Optional[HeteroData] = None,
        data_node_name: str = "data",
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
        )


# ---------------------------------------------------------------------------
# File-based graph loading helpers
# ---------------------------------------------------------------------------


class _GraphFileDataset(Dataset):
    """Lazily loads graph files from a directory.

    Each call to ``__getitem__`` opens exactly one file from disk so the
    entire collection never has to reside in RAM simultaneously.

    Parameters
    ----------
    graph_dir : Path
        Directory that contains graph files.
    extension : str
        File suffix to glob for (default ``".pt"``).
    """

    def __init__(self, graph_dir: Path, extension: str = ".pt") -> None:
        self.graph_dir = graph_dir
        if not self.graph_dir.is_dir():
            raise FileNotFoundError(f"Graph directory not found: {self.graph_dir}")

        self.paths: list[Path] = sorted(self.graph_dir.glob(f"*{extension}"))
        if not self.paths:
            raise RuntimeError(f"No {extension} files found in {self.graph_dir}")
        
        self.paths = {path.parts[-1].split(".")[0]: path for path in self.paths}
        self.names = list(self.paths.keys())

        LOGGER.info("Found %d graph file(s) in %s", len(self.paths), self.graph_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, name: str) -> HeteroData:
        """Load and return the graph object at *names*."""
        path = self.paths[name]
        return torch.load(path, weights_only=False, map_location="cpu")

    def __repr__(self) -> str:
        return f"_GraphFileDataset(n={len(self)}, dir={self.graph_dir})"


class FileGraphProvider(BaseGraphProvider):
    """Provider that loads graphs from files in a directory using a DataLoader.

    Creates an internal graph DataLoader to lazily stream pre-computed graph
    files (e.g. ``*.pt``) from disk.  The provider exposes an iterable
    interface so callers can iterate over the loaded graphs.

    Graph metadata (``src_size``, ``dst_size``, ``edge_attributes``, ``trainable_size``)
    is inferred from the first graph file in the directory.  Each graph is expected to
    carry:

    * ``edge_index`` – [2, num_edges] tensor
    * One or more edge attribute tensors (names listed in ``edge_attribute_names``)
    * ``src_size`` (int attribute) – number of source nodes
    * ``dst_size`` (int attribute) – number of destination nodes
    * ``edge_attribute_names`` (list[str]) – names of edge attribute tensors to concatenate
    * ``trainable_size`` (int attribute, optional) – learnable edge param width (default 0)

    If ``src_size`` / ``dst_size`` are not stored on the graph they are inferred
    from ``edge_index``.

    Parameters
    ----------
    graph_dir : str | Path
        Directory containing graph files.
    extension : str
        File extension to search for (default ``".pt"``).
    batch_size : int
        Number of graphs per batch (default 1).
    num_workers : int
        Number of DataLoader worker processes (default 1).
    prefetch_factor : int
        Batches to prefetch per worker (default 1).
    shuffle : bool
        Whether to shuffle file order each epoch (default False).
    pin_memory : bool
        Whether to pin tensors into page-locked memory (default True).
    """

    def __init__(
        self,
        src_size: str,
        dst_size: str,
        graph_dir: Union[str, Path],
        dataset_name: str = None,
        edge_attributes: Optional[list[str]] = None,
        trainable_size: int = 0,
        extension: str = ".pt",
        batch_size: int = 1,
        num_workers: int = 1,
        prefetch_factor: int = 1,
        shuffle: bool = False,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.src_name = src_size
        self.dst_name = dst_size
        self.graph_name = dataset_name if dataset_name is not None else "graph"
        self.edge_attributes = edge_attributes
        self.trainable_size = trainable_size

        self.graph_dir = Path(graph_dir)

        # Build dataset and dataloader
        self._dataset = _GraphFileDataset(self.graph_dir, extension=extension)
        self.names = self._dataset.names

        collate_fn = PyGCollater(dataset=None, follow_batch=[], exclude_keys=[])

        loader_kwargs: dict = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
        )

        self._dataloader = DataLoader(self._dataset, **loader_kwargs)

        # Peek at the first graph to derive metadata
        first_graph = self._dataset[self.names[0]].cuda(non_blocking=True)
        self._init_from_graph(first_graph)

    def _init_from_graph(self, graph: HeteroData) -> None:
        """Derive src_size, dst_size, edge_attributes, trainable_size from a graph."""
        # --- src_size / dst_size ---
        self.src_size = graph[self.src_name].num_nodes
        self.dst_size = graph[self.dst_name].num_nodes
        assert (
            self.src_size is not None or self.dst_size is not None
        ), "Graph must have at least one of src_size or dst_size attributes"

        edge_attr_tensor = torch.cat([graph[(self.src_name, "to", self.dst_name)][attr] for attr in self.edge_attributes], axis=1)


        self._edge_dim = edge_attr_tensor.shape[1] + self.trainable_size

        self.register_buffer(
            "edge_inc",
            torch.from_numpy(np.asarray([[self.src_size], [self.dst_size]], dtype=np.int64)),
            persistent=False,
        )

        self.trainable = TrainableTensor(trainable_size=self.trainable_size, tensor_size=edge_attr_tensor.shape[0])

    @property
    def edge_dim(self) -> int:
        """Return the edge dimension."""
        return self._edge_dim

    @property
    def dataloader(self) -> DataLoader:
        """Return the underlying DataLoader."""
        return self._dataloader

    def __len__(self) -> int:
        """Return the number of graph files."""
        return len(self._dataset)

    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over graphs loaded by the DataLoader."""
        return iter(self._dataloader)

    def __getitem__(self, index: int) -> HeteroData:
        """Get a specific subgraph by index."""
        graph = self._dataset[self.names[index]][(self.src_name, "to", self.dst_name)]
        assert (
            graph.src_size == self.src_size
        ), f"Graph src_size {graph.src_size} does not match expected {self.src_size}"
        assert (
            graph.dst_size == self.dst_size
        ), f"Graph dst_size {graph.dst_size} does not match expected {self.dst_size}"
        assert (
            int(getattr(graph, "trainable_size", 0)) == self.trainable_size
        ), f"Graph trainable_size {getattr(graph, 'trainable_size', 0)} does not match expected {self.trainable_size}"
        return graph

    def _expand_edges(self, edge_index: Adj, batch_size: int) -> Adj:
        """Expand edge index for batched processing."""
        return torch.cat(
            [edge_index + i * self.edge_inc for i in range(batch_size)],
            dim=1,
        )

    def get_edges(
        self,
        batch_size: int = 1,
        src_coords: Optional[Tensor] = None,
        dst_coords: Optional[Tensor] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        shard_edges: bool = True,
        device: Optional[torch.device] = None,
    ) -> tuple[Tensor, Adj, Optional[ShardSizes]]:
        """Get edges from a specific loaded graph.

        Use ``iter(provider)`` or ``provider.dataloader`` to iterate over
        graphs, then pass the loaded graph to this method.

        Parameters
        ----------
        batch_size : int, optional
            Number of times to expand the edge index.
        src_coords : Tensor, optional
            Unused.
        dst_coords : Tensor, optional
            Unused.
        model_comm_group : ProcessGroup, optional
            Model communication group.
        shard_edges : bool, optional
            Whether to shard edges, by default True.
        graph : HeteroData, optional
            A graph loaded from the dataloader.  If None, the first graph
            in the dataset is loaded.

        Returns
        -------
        tuple[Tensor, Adj, Optional[ShardSizes]]
            Edge attributes, expanded edge index, and optional edge_shard_sizes.
        """

        full_graph = self._dataset[self.graph_name]
        src_size = full_graph[self.src_name].num_nodes
        dst_size = full_graph[self.dst_name].num_nodes
        graph = self._dataset[self.graph_name][(self.src_name, "to", self.dst_name)].to(device)

        edge_attr = torch.cat([graph[attr] for attr in self.edge_attributes], axis=1)
        edge_attr = self.trainable(edge_attr, batch_size).to(device)

        # Derive src/dst sizes from this specific graph (may differ across files)
        src_size = int(getattr(graph, "src_size", graph.edge_index[0].max().item() + 1))
        dst_size = int(getattr(graph, "dst_size", graph.edge_index[1].max().item() + 1))

        edge_index = graph.edge_index
        edge_inc = torch.tensor([[src_size], [dst_size]], dtype=torch.int64, device=device)
        edge_index = torch.cat(
            [edge_index + i * edge_inc for i in range(batch_size)],
            dim=1,
        )
        edge_index = edge_index.to(torch.int64)
        edge_attr = edge_attr.to(torch.float16)

        if shard_edges:
            # ensure correct dtypes for sharding
            
            return shard_edges_1hop(
                edge_attr, edge_index, src_size * batch_size, dst_size * batch_size, model_comm_group
            )
        # remove graph from memory after processing
        del graph
        del full_graph
        return edge_attr, edge_index, None

