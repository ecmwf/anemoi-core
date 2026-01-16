import torch


class SparseProjector(torch.nn.Module):
    """Applies sparse projection matrix to input tensors.

    Stateless projector that receives matrix as input to forward().
    """

    def __init__(self, autocast: bool = False) -> None:
        """Initialize SparseProjector.

        Parameters
        ----------
        autocast : bool
            Use automatic mixed precision
        """
        super().__init__()
        self.autocast = autocast

    def forward(self, x: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """Apply sparse projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        projection_matrix : torch.Tensor
            Sparse projection matrix (assumed to be on correct device)

        Returns
        -------
        torch.Tensor
            Projected tensor
        """
        out = []
        with torch.amp.autocast(device_type=x.device.type, enabled=self.autocast):
            for i in range(x.shape[0]):
                out.append(torch.sparse.mm(projection_matrix, x[i, ...]))
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
