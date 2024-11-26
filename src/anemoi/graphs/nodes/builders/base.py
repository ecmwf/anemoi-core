# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData


class BaseNodeBuilder(ABC):
    """Base class for node builders.

    The node coordinates are stored in the `x` attribute of the nodes and they are stored in radians.

    Attributes
    ----------
    name : str
        name of the nodes, key for the nodes in the HeteroData graph object.
    area_mask_builder : KNNAreaMaskBuilder
        The area of interest mask builder, if any. Defaults to None.
    """

    hidden_attributes: set[str] = set()

    def __init__(self, name: str) -> None:
        self.name = name
        self.area_mask_builder = None

    def register_nodes(self, graph: HeteroData) -> HeteroData:
        """Register nodes in the graph.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the nodes.

        Returns
        -------
        HeteroData
            The graph with the registered nodes.
        """
        graph[self.name].x = self.get_coordinates()
        graph[self.name].node_type = type(self).__name__
        return graph

    def register_attributes(self, graph: HeteroData, config: DotDict | None = None) -> HeteroData:
        """Register attributes in the nodes of the graph specified.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the attributes.
        config : DotDict
            The configuration of the attributes.

        Returns
        -------
        HeteroData
            The graph with the registered attributes.
        """
        for hidden_attr in self.hidden_attributes:
            graph[self.name][f"_{hidden_attr}"] = getattr(self, hidden_attr)

        for attr_name, attr_config in config.items():
            graph[self.name][attr_name] = instantiate(attr_config).compute(graph, self.name)

        return graph

    @abstractmethod
    def get_coordinates(self) -> torch.Tensor: ...

    def reshape_coords(self, latitudes: np.ndarray, longitudes: np.ndarray) -> torch.Tensor:
        """Reshape latitude and longitude coordinates.

        Parameters
        ----------
        latitudes : np.ndarray of shape (num_nodes, )
            Latitude coordinates, in degrees.
        longitudes : np.ndarray of shape (num_nodes, )
            Longitude coordinates, in degrees.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        coords = np.stack([latitudes, longitudes], axis=-1).reshape((-1, 2))
        coords = np.deg2rad(coords)
        return torch.tensor(coords, dtype=torch.float32)

    def update_graph(self, graph: HeteroData, attrs_config: DotDict | None = None) -> HeteroData:
        """Update the graph with new nodes.

        Parameters
        ----------
        graph : HeteroData
            Input graph.
        attrs_config : DotDict
            The configuration of the attributes.

        Returns
        -------
        HeteroData
            The graph with new nodes included.
        """
        graph = self.register_nodes(graph)

        if attrs_config is None:
            return graph

        graph = self.register_attributes(graph, attrs_config)

        return graph
