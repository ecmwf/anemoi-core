# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict
from collections.abc import Iterable

import einops
import torch
from rich.console import Console
from rich.tree import Tree
from torch import Tensor
from torch import nn
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


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

    num_nodes: dict[str, int]
    num_trainable_parameters: dict[str, int]
    attr_ndims: dict[str, int]
    trainable_tensors: dict[str, TrainableTensor]

    def __init__(self, trainable_parameters: dict[str, int], graph_data: HeteroData) -> None:
        """Initialize NodeTrainableParameters."""
        super().__init__()

        # Only nodes present in the (static) graph get trainable tensors. Tabular/dynamic
        # datasets (e.g. observations) have a variable number of nodes per batch and are not
        # part of the static graph, so they carry no trainable node parameters. Keeping
        # ``num_trainable_parameters`` aligned with ``trainable_tensors`` ensures the input
        # dimension computed in the model matches what is actually concatenated at runtime.
        self.num_trainable_parameters = defaultdict(int)  # default to 0 for missing nodes

        self.trainable_tensors = nn.ModuleDict()
        for nodes_name, nodes in graph_data.node_items():
            self.num_trainable_parameters[nodes_name] = trainable_parameters.get(nodes_name, 0)
            self.trainable_tensors[nodes_name] = TrainableTensor(
                nodes.num_nodes, self.num_trainable_parameters[nodes_name]
            )

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

        Returns
        -------
        Tensor or None
            The trainable node parameters, repeated ``batch_size`` times, or ``None`` for a
            node group without them.
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


class StaticGraphGeometry:
    """What the forward pass reads from the static graph: node coordinates, node and edge counts.

    The model builds its layers from the static graph but does not keep it: the graph
    providers hold the edges, and this object holds the rest of what the forward pass reads:
    the node and edge counts, and the coordinates of the hidden nodes and of the gridded
    datasets' grids (which spatial pre-processors project onto). So the model runs
    without the graph, and inference checkpoints, which pickle the whole model, do not carry it.

    The coordinates are plain tensors, not buffers, as they were in the graph. They are
    pickled with the model but are not part of the ``state_dict``, so training checkpoints are
    unchanged. Casting the model (``model.to(torch.float64)``) leaves them alone: they are
    geometry, combined with the batch coordinates to build the dynamic graphs, not model
    features. They stay on the graph's device; the forward pass moves them to the batch's.

    Parameters
    ----------
    graph : HeteroData
        The static graph.
    coordinate_node_names : Iterable[str]
        Node sets whose coordinates are kept, e.g. the hidden meshes and the dataset grids.
    """

    def __init__(self, graph: HeteroData, coordinate_node_names: Iterable[str]) -> None:
        self.num_nodes: dict[str, int] = {name: int(graph[name].num_nodes) for name in graph.node_types}
        self.num_edges: dict[tuple[str, str, str], int] = {
            edge_type: int(graph[edge_type].num_edges) for edge_type in graph.edge_types
        }
        self._coordinates: dict[str, Tensor] = {
            name: graph[name].x.detach().clone() for name in dict.fromkeys(coordinate_node_names)
        }

    def coordinates(self, name: str) -> Tensor:
        """Return the ``(num_nodes, 2)`` coordinates of the node set ``name``."""
        if name not in self._coordinates:
            msg = f"No coordinates kept for node set {name!r}; kept: {list(self._coordinates)}."
            raise KeyError(msg)
        return self._coordinates[name]
