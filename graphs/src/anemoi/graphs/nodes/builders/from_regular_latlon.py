# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch

from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class RegularLatLonNodes(BaseNodeBuilder):
    """Nodes at the cell centres of a regular latitude-longitude grid.

    With a resolution of ``r`` degrees the grid has ``180 / r`` latitudes,
    ``90 - (i + 0.5) * r`` for ``i = 0, 1, ...``, and ``360 / r`` longitudes,
    ``j * r`` for ``j = 0, 1, ...``. No node lies on a pole. Nodes are stored
    row by row, from north to south, with longitudes running east from zero
    within each row. Processors that work on the grid as an image, such as
    ``anemoi.models.layers.processor.ADRProcessor``, rely on this order.

    Attributes
    ----------
    resolution : float
        Grid spacing in degrees. 180 / resolution must be a whole number.

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

    def __init__(self, resolution: float, name: str) -> None:
        """Initialize the RegularLatLonNodes builder."""
        if resolution <= 0 or abs(180 / resolution - round(180 / resolution)) > 1e-9:
            raise ValueError(f"180 / resolution must be a whole number, got resolution={resolution}.")
        self.resolution = resolution
        self.nlat = round(180 / resolution)
        self.nlon = 2 * self.nlat
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        latitudes = 90 - (torch.arange(self.nlat, dtype=torch.float64) + 0.5) * self.resolution
        longitudes = torch.arange(self.nlon, dtype=torch.float64) * self.resolution
        latitudes, longitudes = torch.meshgrid(latitudes, longitudes, indexing="ij")
        return self.reshape_coords(latitudes.reshape(-1), longitudes.reshape(-1))
