from typing import Optional

import torch
from torch_geometric.data import HeteroData


def _row_normalize_weights(
    edge_index,
    weights,
    num_target_nodes,
):
    total = torch.zeros(num_target_nodes, device=weights.device)
    norm = total.scatter_add_(0, edge_index[1].long(), weights)
    norm = norm[edge_index[1]]
    return weights / (norm + 1e-8)


class SparseProjector(torch.nn.Module):
    """Constructs and applies a sparse projection matrix for mapping features between grids.

    The projection matrix is constructed from edge indices and edge attributes (e.g., distances),
    with optional row normalization.

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
        autocast: bool = False,
    ) -> None:
        super().__init__()
        self.autocast = autocast

        weights = _row_normalize_weights(edge_index, weights, dst_size) if row_normalize else weights

        self.projection_matrix = (
            torch.sparse_coo_tensor(
                edge_index,
                weights,
                (src_size, dst_size),
                device=edge_index.device,
            )
            .coalesce()
            .T
        )

    def forward(self, x, *args, **kwargs):
        # This has to be called in the forward because sparse tensors cannot be registered as buffers,
        # as they can't be broadcast correctly when using DDP.
        self.projection_matrix = self.projection_matrix.to(x.device)

        out = []
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.amp.autocast(device_type=device_type, enabled=self.autocast):
            for i in range(x.shape[0]):
                out.append(torch.sparse.mm(self.projection_matrix, x[i, ...]))
        return torch.stack(out)


def build_sparse_projector(
    graph: HeteroData,
    edges_name: tuple[str, str, str],
    edge_weight_attribute: Optional[str] = None,
    src_node_weight_attribute: Optional[str] = None,
    autocast: bool = False,
) -> SparseProjector:
    """Factory method to build a SparseProjector."""
    sub_graph = graph[edges_name]

    weights = torch.ones(sub_graph.edge_index.shape[1], device=sub_graph.edge_index.device)

    if edge_weight_attribute:
        weights = sub_graph[edge_weight_attribute].squeeze().to(sub_graph.edge_index.device)
    else:
        torch.ones(sub_graph.edge_index.shape[1], device=sub_graph.edge_index.device)

    if src_node_weight_attribute:
        weights *= graph[edges_name[0]][src_node_weight_attribute][sub_graph.edge_index[0]]

    return SparseProjector(
        edge_index=sub_graph.edge_index,
        weights=weights,
        src_size=sub_graph[edges_name[0]].num_nodes,
        dst_size=sub_graph[edges_name[2]].num_nodes,
        autocast=autocast,
    )
