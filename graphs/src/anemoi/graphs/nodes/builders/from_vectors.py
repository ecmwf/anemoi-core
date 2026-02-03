# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import torch

from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class LatLonNodes(BaseNodeBuilder):
    """Nodes from its latitude and longitude positions (in numpy arrays).

    Attributes
    ----------
    latitudes : list | np.ndarray
        The latitude of the nodes, in degrees.
    longitudes : list | np.ndarray
        The longitude of the nodes, in degrees.

    Methods
    -------
    get_coordinates()
        Get the lat-lon coordinates of the nodes.
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attrs_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(
        self,
        latitudes: list[float] | np.ndarray | torch.Tensor,
        longitudes: list[float] | np.ndarray | torch.Tensor,
        name: str,
    ) -> None:
        super().__init__(name)
        self.latitudes = latitudes if isinstance(latitudes, torch.Tensor) else torch.tensor(latitudes)
        self.longitudes = longitudes if isinstance(longitudes, torch.Tensor) else torch.tensor(longitudes)

        assert len(self.latitudes) == len(
            self.longitudes
        ), f"Lenght of latitudes and longitudes must match but {len(self.latitudes)}!={len(self.longitudes)}."
        assert self.latitudes.ndim == 1 or (
            self.latitudes.ndim == 2 and self.latitudes.shape[1] == 1
        ), "latitudes must have shape (N, ) or (N, 1)."
        assert self.longitudes.ndim == 1 or (
            self.longitudes.ndim == 2 and self.longitudes.shape[1] == 1
        ), "longitudes must have shape (N, ) or (N, 1)."

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        return self.reshape_coords(self.latitudes, self.longitudes)
