# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict

import einops
import torch
from torch import Tensor
from torch import nn
from torch_geometric.data import HeteroData

from rich.console import Console
from rich.tree import Tree


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

    def tree(self, prefix: str = "") -> str:
        if self.trainable is None:
            return prefix + "❌ No node trainable parameters"

        return prefix + self.__class__.__name__ + f" ({self.trainable.shape[0]} x {self.trainable.shape[1]})"


class NodeTrainableParameters(nn.Module):
    """Node Trainable Attributes information.

    Attributes
    ----------
    num_trainable_parameters : dict[str, int]
        Total dimension of node attributes (non-trainable + trainable) for each group of nodes. If the dataset is
        tabular, trainable_parameter is set to 0.
    trainable_tensors : nn.ModuleDict
        Dictionary of trainable tensors for each group of nodes.

    Methods
    -------
    forward(self, name: str, batch_size: int) -> Tensor
        Get the node attributes to be passed trough the graph neural network.
    """

    num_trainable_parameters: dict[str, int]
    trainable_tensors: dict[str, TrainableTensor]

    def __init__(self, trainable_parameters: dict[str, int], graph_data: HeteroData) -> None:
        """Initialize NodeTrainableParameters."""
        super().__init__()

        self.num_trainable_parameters = defaultdict(int, trainable_parameters) # default to 0 for missing nodes

        self.trainable_tensors = nn.ModuleDict()
        for nodes_name, nodes in graph_data.node_items():
            self.trainable_tensors[nodes_name] = TrainableTensor(nodes.num_nodes, self.num_trainable_parameters[nodes_name])

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

    def __contains__(self, name: str) -> bool:
        """Check if a node group exists in the named nodes attributes."""
        return name in self.trainable_tensors and self.num_trainable_parameters[name] > 0

    def tree(self, prefix: str = "") -> Tree:
        tree = Tree(prefix + " 💾 " + f"{self.__class__.__name__}")
        for dataset_name, trainable_tensor in self.trainable_tensors.items():
            tree.add(trainable_tensor.tree(f"{dataset_name}: "))
        return tree

    def __repr__(self) -> str:
        """Return a string representation of the NodeTrainableParameters."""
        console = Console(record=True, width=120)
        with console.capture() as capture:
            console.print(self.tree())
        return capture.get()
