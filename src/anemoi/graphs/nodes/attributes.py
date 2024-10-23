# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from scipy.spatial import SphericalVoronoi
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian
from anemoi.graphs.normalise import NormaliserMixin

LOGGER = logging.getLogger(__name__)


class BaseWeights(ABC, NormaliserMixin):
    """Base class for the weights of the nodes."""

    def __init__(self, norm: str | None = None) -> None:
        self.norm = norm

    @abstractmethod
    def get_raw_values(self, nodes: NodeStorage, *args, **kwargs): ...

    def post_process(self, values: np.ndarray) -> torch.Tensor:
        """Post-process the values."""
        if values.ndim == 1:
            values = values[:, np.newaxis]

        norm_values = self.normalise(values)

        return torch.tensor(norm_values, dtype=torch.float32)

    def compute(self, graph: HeteroData, nodes_name: str, *args, **kwargs) -> torch.Tensor:
        """Get the node weights.

        Parameters
        ----------
        graph : HeteroData
            Graph.
        nodes_name : str
            Name of the nodes.
        args : tuple
            Additional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Weights associated to the nodes.
        """
        nodes = graph[nodes_name]
        weights = self.get_raw_values(nodes, *args, **kwargs)
        return self.post_process(weights)


class UniformWeights(BaseWeights):
    """Implements a uniform weight for the nodes.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def get_raw_values(self, nodes: NodeStorage, *args, **kwargs) -> np.ndarray:
        """Compute the weights.

        Parameters
        ----------
        nodes : NodeStorage
            Nodes of the graph.
        args : tuple
            Additional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Weights.
        """
        return np.ones(nodes.num_nodes)


class AreaWeights(BaseWeights):
    """Implements the area of the nodes as the weights.

    Attributes
    ----------
    norm : str
        Normalisation of the weights.
    radius : float
        Radius of the sphere.
    centre : np.ndarray
        Centre of the sphere.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def __init__(self, norm: str | None = None, radius: float = 1.0, centre: np.ndarray = np.array([0, 0, 0])) -> None:
        super().__init__(norm)
        self.radius = radius
        self.centre = centre

    def get_raw_values(self, nodes: NodeStorage, *args, **kwargs) -> np.ndarray:
        """Compute the area associated to each node.

        It uses Voronoi diagrams to compute the area of each node.

        Parameters
        ----------
        nodes : NodeStorage
            Nodes of the graph.
        args : tuple
            Additional arguments.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Weights.
        """
        latitudes, longitudes = nodes.x[:, 0], nodes.x[:, 1]
        points = latlon_rad_to_cartesian((np.asarray(latitudes), np.asarray(longitudes)))
        sv = SphericalVoronoi(points, self.radius, self.centre)
        area_weights = sv.calculate_areas()
        LOGGER.debug(
            "There are %d of weights, which (unscaled) add up a total weight of %.2f.",
            len(area_weights),
            np.array(area_weights).sum(),
        )
        return area_weights
