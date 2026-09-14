# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import radius as pyg_radius

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.edges.builders.euclidean.base import BaseDistanceEdgeBuilders
from anemoi.graphs.utils import get_grid_reference_distance
from anemoi.graphs.utils import crop_to_max_num_neighbours

LOGGER = logging.getLogger(__name__)


class BaseCutOffEdges(BaseDistanceEdgeBuilders):
    """Base class for cut-off based edges."""

    def __init__(
        self,
        source_name: str,
        target_name: str,
        cutoff_factor: float | None = None,
        cutoff_distance_km: float | None = None,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
        max_num_neighbours: int = 64,
    ) -> None:
        super().__init__(source_name, target_name, source_mask_attr_name, target_mask_attr_name)

        # Validate that exactly one of cutoff_factor or cutoff_distance_km is provided
        if cutoff_factor is None and cutoff_distance_km is None:
            raise ValueError("Either cutoff_factor or cutoff_distance_km must be provided.")
        if cutoff_factor is not None and cutoff_distance_km is not None:
            raise ValueError("cutoff_factor and cutoff_distance_km are mutually exclusive. Provide only one.")

        if cutoff_factor is not None:
            assert isinstance(cutoff_factor, (int, float)), "Cutoff factor must be a float."
            assert cutoff_factor > 0, "Cutoff factor must be positive."

        if cutoff_distance_km is not None:
            assert isinstance(cutoff_distance_km, (int, float)), "Cutoff distance must be a float."
            assert cutoff_distance_km > 0, "Cutoff distance must be positive."

        assert isinstance(max_num_neighbours, int), "Number of nearest neighbours must be an integer."
        assert max_num_neighbours > 0, "Number of nearest neighbours must be positive."

        self.cutoff_factor = cutoff_factor
        self.cutoff_distance_km = cutoff_distance_km
        self.max_num_neighbours = max_num_neighbours

    def get_cutoff_radius(self, reference_coords: torch.Tensor) -> float:
        """Compute the cut-off radius.

        The cut-off radius is computed either as:
        - The product of the target nodes reference distance and the cut-off factor, or
        - Directly from the cutoff distance in kilometers.

        Parameters
        ----------
        reference_coords : torch.Tensor
            The reference coordinates. Assumed to be in Cartesian coordinates on the unit sphere.

        Returns
        -------
        float
            The cut-off radius in Cartesian coordinates on unit sphere.
        """
        if self.cutoff_distance_km is not None:
            LOGGER.info(
                "Using %s (with radius = %.1f km [direct]) between %s and %s.",
                self.__class__.__name__,
                self.cutoff_distance_km,
                self.source_name,
                self.target_name,
            )
            # Convert km to Cartesian distance on unit sphere
            # For small distances: Cartesian distance ≈ great circle distance (radians)
            # radians = km / EARTH_RADIUS
            radius = self.cutoff_distance_km / EARTH_RADIUS
        else:
            # Use factor-based approach
            reference_dist = get_grid_reference_distance(reference_coords, use_cartesian=False)
            radius = reference_dist * self.cutoff_factor
            LOGGER.info(
                "Using %s (with radius = %.1f km [factor=%.2f]) between %s and %s.",
                self.__class__.__name__,
                radius * EARTH_RADIUS,
                self.cutoff_factor,
                self.source_name,
                self.target_name,
            )
            
        return radius

    def prepare_method_kwargs(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> dict:
        """Prepare keyword arguments for computing edge index."""
        return {"max_num_neighbors": self.max_num_neighbours}

    def _compute_edge_index_pyg(
        self,
        source_coords: torch.Tensor,
        target_coords: torch.Tensor,
        radius: float,
        max_num_neighbors: int,
    ) -> torch.Tensor:
        edge_index = pyg_radius(source_coords, target_coords, r=radius, max_num_neighbors=max_num_neighbors)
        edge_index = torch.flip(edge_index, [0])

        return edge_index

    def _compute_adj_matrix_sklearn(
        self,
        source_coords: torch.Tensor,
        target_coords: torch.Tensor,
        radius: float,
        max_num_neighbours: int,
    ) -> torch.Tensor:
        nearest_neighbour = NearestNeighbors(metric="euclidean", n_jobs=4)
        nearest_neighbour.fit(source_coords.cpu())

        adj_matrix = nearest_neighbour.radius_neighbors_graph(
            target_coords.cpu(), radius=radius, mode="distance"
        ).tocoo()

        adj_matrix = crop_to_max_num_neighbours(adj_matrix, max_num_neighbours=max_num_neighbours)
        return adj_matrix


class CutOffEdges(BaseCutOffEdges):
    """Computes cut-off based edges and adds them to the graph.

    It uses as reference the target nodes.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    cutoff_factor : float | None
        Factor to multiply the grid reference distance to get the cut-off radius.
        Mutually exclusive with cutoff_distance_km.
    cutoff_distance_km : float | None
        Cutoff radius in kilometers. Mutually exclusive with cutoff_factor.
    source_mask_attr_name : str | None
        The name of the source mask attribute to filter edge connections.
    target_mask_attr_name : str | None
        The name of the target mask attribute to filter edge connections.
    max_num_neighbours : int
        The maximum number of nearest neighbours to consider when building edges.

    Methods
    -------
    register_edges(graph)
        Register the edges in the graph.
    register_attributes(graph, config)
        Register attributes in the edges of the graph.
    update_graph(graph, attrs_config)
        Update the graph with the edges.
    """
    def prepare_method_kwargs(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> dict:
        """Prepare keyword arguments for computing edge index."""
        radius = self.get_cutoff_radius(reference_coords=target_coords)
        return {"radius": radius} | super().prepare_method_kwargs(source_coords, target_coords)


class ReversedCutOffEdges(BaseCutOffEdges):
    """Computes cut-off based edges and adds them to the graph.

    It uses as reference the source nodes.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    cutoff_factor : float
        Factor to multiply the grid reference distance to get the cut-off radius.
    source_mask_attr_name : str | None
        The name of the source mask attribute to filter edge connections.
    target_mask_attr_name : str | None
        The name of the target mask attribute to filter edge connections.
    max_num_neighbours : int
        The maximum number of nearest neighbours to consider when building edges.

    Methods
    -------
    register_edges(graph)
        Register the edges in the graph.
    register_attributes(graph, config)
        Register attributes in the edges of the graph.
    update_graph(graph, attrs_config)
        Update the graph with the edges.
    """

    def prepare_method_kwargs(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> dict:
        """Prepare keyword arguments for computing edge index."""
        radius = self.get_cutoff_radius(reference_coords=source_coords)
        return {"radius": radius} | super().prepare_method_kwargs(source_coords, target_coords)


    def compute_edge_index_from_coords(
        self,
        source_coords: torch.Tensor,
        target_coords: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        edge_index = super().compute_edge_index_from_coords(target_coords, source_coords, **kwargs)
        edge_index = torch.flip(edge_index, [0])
        return edge_index
