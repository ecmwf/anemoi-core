# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import einops
import torch
from torch import Tensor
from torch import nn
from torch_geometric.data import HeteroData


class TrainableTensor(nn.Module):
    """Trainable Tensor Module."""

    def __init__(self, tensor_size: int, trainable_size: int) -> None:
        """Initialize TrainableTensor."""
        super().__init__()

        if trainable_size > 0:
            trainable = nn.Parameter(
                torch.empty(
                    tensor_size,
                    trainable_size,
                ),
            )
            nn.init.constant_(trainable, 0)
        else:
            trainable = None
        self.register_parameter("trainable", trainable)

    def forward(self, x: Tensor, batch_size: int) -> Tensor:
        latent = [einops.repeat(x, "e f -> (repeat e) f", repeat=batch_size)]
        if self.trainable is not None:
            latent.append(einops.repeat(self.trainable.to(x.device), "e f -> (repeat e) f", repeat=batch_size))
        return torch.cat(
            latent,
            dim=-1,  # feature dimension
        )


class NamedNodesAttributes(nn.Module):
    """Named Nodes Attributes information.

    Attributes
    ----------
    num_nodes : dict[str, int]
        Number of nodes for each group of nodes.
    attr_ndims : dict[str, int]
        Total dimension of node attributes (non-trainable + trainable) for each group of nodes.
    trainable_tensors : nn.ModuleDict
        Dictionary of trainable tensors for each group of nodes.

    Methods
    -------
    get_coordinates(self, name: str) -> Tensor
        Get the coordinates of a set of nodes.
    forward( self, name: str, batch_size: int) -> Tensor
        Get the node attributes to be passed trough the graph neural network.
    """

    num_nodes: dict[str, int]
    attr_ndims: dict[str, int]
    trainable_tensors: dict[str, TrainableTensor]

    def __init__(
        self,
        num_trainable_params: int,
        graph_data: HeteroData,
        static_attribute_names: list[str] | None = None,
    ) -> None:
        """Initialize NamedNodesAttributes.

        static_attribute_names (fine-scale epic, 2026-09-02): names of float node attributes stored in
        the graph (e.g. ``sdor``, ``slor``, ``fsr`` on the data nodes) to concatenate AFTER the
        coordinates and the trainable tensor, so they enter the model as static per-node inputs.
        Absent by default: with an empty list nothing changes and the class is bit-identical to
        the version without this argument. Attributes missing from a node set are skipped for it.
        """
        super().__init__()

        self.static_attribute_names = list(static_attribute_names or [])
        self.static_ndims: dict[str, int] = {}
        self.define_fixed_attributes(graph_data, num_trainable_params)

        self.trainable_tensors = nn.ModuleDict()
        for nodes_name, nodes in graph_data.node_items():
            self.register_coordinates(nodes_name, nodes.x)
            self.register_tensor(nodes_name, num_trainable_params)
            self.register_static_attributes(nodes_name, nodes)

    def define_fixed_attributes(self, graph_data: HeteroData, num_trainable_params: int) -> None:
        """Define fixed attributes."""
        nodes_names = list(graph_data.node_types)
        self.num_nodes = {nodes_name: graph_data[nodes_name].num_nodes for nodes_name in nodes_names}
        self.attr_ndims = {
            nodes_name: 2 * graph_data[nodes_name].x.shape[1] + num_trainable_params for nodes_name in nodes_names
        }
        for nodes_name in nodes_names:
            present = [a for a in self.static_attribute_names if a in graph_data[nodes_name]]
            self.static_ndims[nodes_name] = sum(int(graph_data[nodes_name][a].shape[-1]) for a in present)
            self.attr_ndims[nodes_name] += self.static_ndims[nodes_name]

    def register_coordinates(self, name: str, node_coords: Tensor) -> None:
        """Register coordinates."""
        sin_cos_coords = torch.cat([torch.sin(node_coords), torch.cos(node_coords)], dim=-1)
        self.register_buffer(f"latlons_{name}", sin_cos_coords, persistent=True)

    def get_coordinates(self, name: str) -> Tensor:
        """Return original coordinates."""
        sin_cos_coords = getattr(self, f"latlons_{name}")
        ndim = sin_cos_coords.shape[1] // 2
        sin_values = sin_cos_coords[:, :ndim]
        cos_values = sin_cos_coords[:, ndim:]
        return torch.atan2(sin_values, cos_values)

    def register_tensor(self, name: str, num_trainable_params: int) -> None:
        """Register a trainable tensor."""
        self.trainable_tensors[name] = TrainableTensor(self.num_nodes[name], num_trainable_params)

    def register_static_attributes(self, name: str, nodes) -> None:
        """Register the named static attributes of one node set as a persistent buffer (if any)."""
        present = [a for a in self.static_attribute_names if a in nodes]
        if not present:
            return
        cols = []
        for a in present:
            v = nodes[a]
            v = v.reshape(v.shape[0], -1) if v.ndim != 2 else v
            cols.append(v.to(torch.float32))
        self.register_buffer(f"static_{name}", torch.cat(cols, dim=-1), persistent=True)

    def forward(self, name: str, batch_size: int) -> Tensor:
        """Returns the node attributes to be passed trough the graph neural network.

        It includes both the coordinates and the trainable parameters.
        """
        latlons = getattr(self, f"latlons_{name}")
        out = self.trainable_tensors[name](latlons, batch_size)
        static = getattr(self, f"static_{name}", None)
        if static is not None:
            out = torch.cat([out, einops.repeat(static.to(out.device), "e f -> (repeat e) f", repeat=batch_size)], dim=-1)
        return out
