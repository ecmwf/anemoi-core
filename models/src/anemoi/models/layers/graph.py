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

import einops
import torch
from torch import Tensor
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import participant_node_name

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
    num_trainable_parameters: dict[str, int]
    attr_ndims: dict[str, int]
    trainable_tensors: dict[str, TrainableTensor]

    def __init__(
        self,
        trainable_parameters: dict[str, int],
        graph_data: HeteroData,
        participants: list[str] | None = None,
    ) -> None:
        """Initialize NamedNodesAttributes.

        ``participants`` names the participants whose node groups are suffixed in
        ``graph_data`` (``data_west``, ``hidden_west``, ...). ``trainable_parameters`` stays
        keyed by the unsuffixed names, as in the config.
        """
        super().__init__()

        self._participants = list(participants or [])
        self._active_participant: str | None = None
        self._assert_no_trainable_parameters_per_participant(trainable_parameters)

        self.num_trainable_parameters = defaultdict(int, trainable_parameters)
        self.define_fixed_attributes(graph_data, self.num_trainable_parameters)

        self.trainable_tensors = nn.ModuleDict()
        for nodes_name, nodes in graph_data.node_items():
            self.register_coordinates(nodes_name, nodes.x)
            self.register_tensor(nodes_name, self.num_trainable_parameters[self.base_name(nodes_name)])

    def _assert_no_trainable_parameters_per_participant(self, trainable_parameters: dict[str, int]) -> None:
        """Trainable node attributes are incompatible with multi-participant training.

        One tensor per participant means the set of parameters receiving a gradient changes
        from step to step, which DDP rejects under ``static_graph=True``.
        """
        if len(self._participants) <= 1:
            return

        for nodes_name, size in trainable_parameters.items():
            if size:
                msg = (
                    f"model.node_trainable_parameters.{nodes_name} must be 0 when training on the "
                    f"participants {self._participants} (got {size}): a trainable tensor per participant "
                    "changes the set of parameters receiving gradients between steps."
                )
                raise ValueError(msg)

    @property
    def active_participant(self) -> str | None:
        """Participant whose node groups are currently resolved, ``None`` if there are none."""
        return self._active_participant

    def set_active_participant(self, participant: str | None) -> None:
        """Select the participant that unsuffixed node names resolve to."""
        self._active_participant = participant if participant in self._participants else None

    def resolve(self, name: str) -> str:
        """Return the node-group name of ``name`` for the active participant."""
        return participant_node_name(name, self._active_participant)

    def base_name(self, name: str) -> str:
        """Return ``name`` without its participant suffix, as used in the config."""
        for participant in self._participants:
            suffix = f"_{participant}"
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def define_fixed_attributes(self, graph_data: HeteroData, trainable_parameters: dict[str, int]) -> None:
        """Define fixed attributes."""
        nodes_names = list(graph_data.node_types)

        self.num_nodes = {}
        self.attr_ndims = {}
        for nodes_name in nodes_names:
            base_name = self.base_name(nodes_name)
            if base_name not in trainable_parameters:
                LOGGER.warning(f"Nodes `{base_name}` not found in trainable parameters. Setting to 0.")

            self.num_nodes[nodes_name] = graph_data[nodes_name].num_nodes
            self.attr_ndims[nodes_name] = 2 * graph_data[nodes_name].x.shape[1] + trainable_parameters[base_name]
            LOGGER.info(
                f"{self.__class__.__name__} | Nodes `{nodes_name}` will have {trainable_parameters[base_name]} trainable parameters."
            )

    def register_coordinates(self, name: str, node_coords: Tensor) -> None:
        """Register coordinates."""
        sin_cos_coords = torch.cat([torch.sin(node_coords), torch.cos(node_coords)], dim=-1)
        self.register_buffer(f"latlons_{name}", sin_cos_coords, persistent=True)

    def sin_cos_coordinates(self, name: str) -> Tensor:
        """Return the registered sin/cos coordinates of the active participant's ``name`` nodes."""
        return getattr(self, f"latlons_{self.resolve(name)}")

    def get_coordinates(self, name: str) -> Tensor:
        """Return original coordinates."""
        sin_cos_coords = self.sin_cos_coordinates(name)
        ndim = sin_cos_coords.shape[1] // 2
        sin_values = sin_cos_coords[:, :ndim]
        cos_values = sin_cos_coords[:, ndim:]
        return torch.atan2(sin_values, cos_values)

    def register_tensor(self, name: str, num_trainable_params: int) -> None:
        """Register a trainable tensor."""
        self.trainable_tensors[name] = TrainableTensor(self.num_nodes[name], num_trainable_params)

    def get_tensor(self, name: str) -> TrainableTensor:
        """Return the trainable tensor of the active participant's ``name`` nodes."""
        return self.trainable_tensors[self.resolve(name)]

    def forward(self, name: str, batch_size: int) -> Tensor:
        """Returns the node attributes to be passed trough the graph neural network.

        It includes both the coordinates and the trainable parameters.
        """
        return self.get_tensor(name)(self.sin_cos_coordinates(name), batch_size)
