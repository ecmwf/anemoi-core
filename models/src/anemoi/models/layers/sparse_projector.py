from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData


def _row_normalize_weights(edge_index: torch.Tensor, weights: torch.Tensor, num_target_nodes: int) -> torch.Tensor:
    total = torch.zeros(num_target_nodes, device=weights.device)
    norm = total.scatter_add_(0, edge_index[1].long(), weights)
    norm = norm[edge_index[1]]
    return weights / (norm + 1e-8)


class SparseProjector(torch.nn.Module):
    """Constructs and applies a sparse projection matrix for mapping features between grids.

    The projection matrix is constructed from edge indices and edge attributes (e.g., distances),
    with optional row normalisation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices (2, E) representing source and destination nodes.
    weights : torch.Tensor
        Raw edge attributes (e.g., distances) of shape (E,).
    src_size : int
        Number of nodes in the source grid.
    dst_size : int
        Number of nodes in the target grid.
    row_normalize : bool
        Whether to normalize weights per destination node.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        weights: torch.Tensor,
        src_size: int,
        dst_size: int,
        row_normalize: bool = True,
        transpose: bool = True,
        autocast: bool = False,
    ) -> None:
        super().__init__()
        self.autocast = autocast

        weights = _row_normalize_weights(edge_index, weights, dst_size) if row_normalize else weights

        self.projection_matrix = torch.sparse_coo_tensor(
            edge_index,
            weights,
            (src_size, dst_size),
            device=edge_index.device,
        ).coalesce()

        if transpose:
            self.projection_matrix = self.projection_matrix.T

    @classmethod
    def from_graph(
        cls,
        graph: HeteroData,
        edges_name: str,
        edge_weight_attribute: Optional[str] = None,
        src_node_weight_attribute: Optional[str] = None,
        row_normalize: bool = True,
        **kwargs,
    ) -> "SparseProjector":
        """Build a SparseProjection from a graph.

        Parameters
        ----------
        graph : HeteroData
            The input graph.
        edges_name : str
            The name/identifier for the edge set to use.
        edge_weight_attribute : str, optional
            Attribute name for edge weights.
        src_node_weight_attribute : str, optional
            Attribute name for source node weights.
        row_normalize : bool, optional
            Whether to normalize weights per destination node.
        **kwargs
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        SparseProjector
            A new SparseProjector instance built from the graph.
        """
        sub_graph = graph[edges_name]

        if edge_weight_attribute:
            weights = sub_graph[edge_weight_attribute].squeeze()
        else:
            # uniform weights
            weights = torch.ones(sub_graph.edge_index.shape[1], device=sub_graph.edge_index.device)

        if src_node_weight_attribute:
            weights *= graph[edges_name[0]][src_node_weight_attribute][sub_graph.edge_index[0]]

        return cls(
            edge_index=sub_graph.edge_index,
            weights=weights,
            src_size=graph[edges_name[0]].num_nodes,
            dst_size=graph[edges_name[2]].num_nodes,
            row_normalize=row_normalize,
            **kwargs,
        )

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        row_normalize: bool = True,
        **kwargs,
    ) -> "SparseProjector":
        """Load projection matrix from a file.

        Parameters
        ----------
        file_path : str or Path
            Path to the .npz file containing the sparse projection matrix.
        row_normalize : bool, optional
            Whether to normalize weights per destination node.
        **kwargs
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        SparseProjector
            A new SparseProjector instance loaded from the file.
        """
        from scipy.sparse import load_npz

        truncation_data = load_npz(file_path)
        edge_index = torch.tensor(np.vstack(truncation_data.nonzero()), dtype=torch.long)
        weights = torch.tensor(truncation_data.data, dtype=torch.float32)
        src_size, dst_size = truncation_data.shape
        return cls(
            edge_index=edge_index,
            weights=weights,
            src_size=src_size,
            dst_size=dst_size,
            row_normalize=row_normalize,
            **kwargs,
        )

    def forward(self, x, *args, **kwargs):
        # This has to be called in the forward because sparse tensors cannot be registered as buffers,
        # as they can't be broadcast correctly when using DDP.
        self.projection_matrix = self.projection_matrix.to(x.device)

        out = []
        with torch.amp.autocast(device_type=x.device.type, enabled=self.autocast):
            for i in range(x.shape[0]):
                out.append(torch.sparse.mm(self.projection_matrix, x[i, ...]))
        return torch.stack(out)


def build_sparse_projector(
    *,
    file_path: Optional[str | Path] = None,
    graph: Optional[HeteroData] = None,
    edges_name: Optional[tuple[str, str, str]] = None,
    edge_weight_attribute: Optional[str] = None,
    row_normalize: bool = True,
    transpose: bool = True,
    **kwargs,
) -> SparseProjector:
    """Factory method to build a SparseProjector.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to .npz file containing the projection matrix.
    graph : HeteroData, optional
        Graph data to build the projector from.
    edges_name : tuple[str, str, str], optional
        Name/identifier for the edge set to use from the graph.
    edge_weight_attribute : str, optional
        Attribute name for edge weights.
    row_normalize : bool, optional
        Whether to normalize weights per destination node.
    transpose : bool, optional
        Whether to transpose the projection matrix.
    **kwargs
        Additional keyword arguments passed to the constructor.

    Returns
    -------
    SparseProjector
        A new SparseProjector instance.
    """
    assert (file_path is not None) ^ (
        graph is not None and edges_name is not None
    ), "Either file_path or graph and edges_name must be provided."

    if file_path is not None:
        return SparseProjector.from_file(
            file_path=file_path,
            row_normalize=row_normalize,
            transpose=transpose,
            **kwargs,
        )
    else:
        assert edges_name in graph.edge_types, f"The specified edges_name, {edges_name}, is not present in the graph."
        return SparseProjector.from_graph(
            graph=graph,
            edges_name=edges_name,
            edge_weight_attribute=edge_weight_attribute,
            row_normalize=row_normalize,
            transpose=transpose,
            **kwargs,
        )
