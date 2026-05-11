# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict
from types import NoneType

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

    def forward(self, batch_size: int) -> Tensor | None:
        if self.trainable is None:
            return None

        return einops.repeat(self.trainable, "e f -> (repeat e) f", repeat=batch_size)


class NamedNodesAttributes(nn.Module):
    """Named Nodes Attributes information.

    Attributes
    ----------
    num_nodes : dict[str, int]
        Number of nodes for each group of nodes. None if the number of nodes is not fixed over time 
        (e.g. for tabular datasets).
    num_trainable_parameters : dict[str, int]
        Total dimension of node attributes (non-trainable + trainable) for each group of nodes. If the dataset is
        tabular, trainable_parameter is set to 0.
    trainable_tensors : nn.ModuleDict
        Dictionary of trainable tensors for each group of nodes.

    Methods
    -------
    forward(self, name: str, batch_size: int, coords: Tensor | None = None) -> Tensor
        Get the node attributes to be passed trough the graph neural network.
        When ``coords`` is provided, sin/cos features are computed on the
        fly from the per-batch coordinates rather than read from the static
        ``latlons_{name}`` buffer.
    """

    num_nodes: dict[str, int]
    num_trainable_parameters: dict[str, int]
    trainable_tensors: dict[str, TrainableTensor]

    def __init__(self, trainable_parameters: dict[str, int], graph_data: HeteroData) -> None:
        """Initialize NamedNodesAttributes."""
        super().__init__()

        trainable_parameters = defaultdict(int, trainable_parameters)

        self.define_fixed_attributes(graph_data, trainable_parameters)

        self.trainable_tensors = nn.ModuleDict()
        for nodes_name, _ in graph_data.node_items():
            self.register_tensor(nodes_name, trainable_parameters[nodes_name])

    def define_fixed_attributes(self, graph_data: HeteroData, trainable_parameters: dict[str, int]) -> None:
        """Define fixed attributes."""
        nodes_names = list(graph_data.node_types)
        self.num_nodes = defaultdict(
            lambda: None,
            {nodes_name: graph_data[nodes_name].num_nodes for nodes_name in nodes_names}
        )
        self.num_trainable_parameters = defaultdict(
            int,
            {nodes_name: trainable_parameters[nodes_name] for nodes_name in nodes_names}
        )

    def register_tensor(self, name: str, num_trainable_params: int) -> None:
        """Register a trainable tensor."""
        self.trainable_tensors[name] = TrainableTensor(self.num_nodes[name], num_trainable_params)

    def forward(self, name: str, batch_size: int) -> Tensor | None:
        """Returns the node attributes to be passed trough the graph neural network.

        It includes both the coordinates and the trainable parameters.

        Parameters
        ----------
        name : str
            Name of the node group (graph node type).
        batch_size : int
            Batch size; the (per-node) coordinate features are repeated
            ``batch_size`` times along the leading axis to match the
            flattened ``(batch * grid)`` layout used by the encoder/decoder.
        """
        if name not in self.trainable_tensors:
            return None

        return self.trainable_tensors[name](batch_size)
