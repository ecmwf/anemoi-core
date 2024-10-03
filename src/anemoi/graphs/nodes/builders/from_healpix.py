from __future__ import annotations

import logging

import numpy as np
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class HEALPixNodes(BaseNodeBuilder):
    """Nodes from HEALPix grid.

    HEALPix is an acronym for Hierarchical Equal Area isoLatitude Pixelization of a sphere.

    Attributes
    ----------
    resolution : int
        The resolution of the grid.

    Methods
    -------
    get_coordinates()
        Get the lat-lon coordinates of the nodes.
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attr_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(self, resolution: int, name: str) -> None:
        """Initialize the HEALPixNodes builder."""
        self.resolution = resolution
        super().__init__(name)

        assert isinstance(resolution, int), "Resolution must be an integer."
        assert resolution > 0, "Resolution must be positive."

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            Coordinates of the nodes, in radians.
        """
        import healpy as hp

        spatial_res_degrees = hp.nside2resol(2**self.resolution, arcmin=True) / 60
        LOGGER.info(f"Creating HEALPix nodes with resolution {spatial_res_degrees:.2} deg.")

        npix = hp.nside2npix(2**self.resolution)
        hpxlon, hpxlat = hp.pix2ang(2**self.resolution, range(npix), nest=True, lonlat=True)

        return self.reshape_coords(hpxlat, hpxlon)


class LimitedAreaHEALPixNodes(HEALPixNodes):
    """Nodes from HEALPix grid using an area of interest."""

    def __init__(
        self,
        resolution: str,
        reference_node_name: str,
        name: str,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:

        self.aoi_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

        super().__init__(resolution, name)

    def register_nodes(self, graph: HeteroData) -> None:
        self.aoi_mask_builder.fit(graph)
        return super().register_nodes(graph)

    def get_coordinates(self) -> np.ndarray:
        coords = super().get_coordinates()

        LOGGER.info(
            'Limiting the "%s" nodes to a radius of %.2f km from the nodes of interest.',
            self.name,
            self.aoi_mask_builder.margin_radius_km,
        )
        aoi_mask = self.aoi_mask_builder.get_mask(coords)

        LOGGER.info('Masking out %d nodes from "%s".', len(aoi_mask) - aoi_mask.sum(), self.name)
        coords = coords[aoi_mask]

        return coords
