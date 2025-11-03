# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re

import torch

from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.utils.grids import grids

LOGGER = logging.getLogger(__name__)


class ReducedGaussianGridNodes(BaseNodeBuilder):
    """Nodes from a reduced gaussian grid.

    A gaussian grid is a latitude/longitude grid. The spacing of the latitudes is not regular. However, the spacing of
    the lines of latitude is symmetrical about the Equator. A grid is usually referred to by its 'number' N/O, which
    is the number of lines of latitude between a Pole and the Equator. The N code refers to the original ECMWF reduced
    Gaussian grid, whereas the code O refers to the octahedral ECMWF reduced Gaussian grid.

    Attributes
    ----------
    grid : str
        The reduced gaussian grid, of shape {n,N,o,O}XXX with XXX latitude lines between the pole and
        equator.

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

    def __init__(self, grid: int, name: str) -> None:
        """Initialize the ReducedGaussianGridNodes builder."""
        assert re.fullmatch(
            r"^[oOnN]\d+$", grid
        ), f"{self.__class__.__name__}.grid must match the format [n|N|o|O]XXX with XXX latitude lines between the pole and equator."
        self.grid = grid
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        grid_data = grids(self.grid)
        coords = self.reshape_coords(grid_data["latitudes"], grid_data["longitudes"])
        return coords
