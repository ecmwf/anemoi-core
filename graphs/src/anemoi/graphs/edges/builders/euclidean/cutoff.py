# (C) Copyright 2024-2026 Anemoi contributors.
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
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import radius as pyg_radius

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.edges.builders.euclidean.base import BaseDistanceEdgeBuilders
from anemoi.graphs.utils import crop_to_max_num_neighbours
from anemoi.graphs.utils import get_grid_reference_distance

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
        assert not (
            cutoff_factor is None and cutoff_distance_km is None
        ), "Either cutoff_factor or cutoff_distance_km must be provided."
        assert not (
            cutoff_factor is not None and cutoff_distance_km is not None
        ), "cutoff_factor and cutoff_distance_km are mutually exclusive. Provide only one."

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
        return {"max_num_neighbours": self.max_num_neighbours}

    def _compute_edge_index_pyg(
        self,
        source_coords: torch.Tensor,
        target_coords: torch.Tensor,
        radius: float,
        max_num_neighbours: int,
        skip_flip: bool = False,
    ) -> torch.Tensor:
        """Compute the edge index using PyG's radius-based implementation.

        If the number of actual neighbours is greater than :obj:`max_num_neighbors`,
        returned neighbours are picked randomly. (default: :obj:`32`)

        Parameters
        ----------
        source_coords : torch.Tensor
            The coordinates of the source nodes.
        target_coords : torch.Tensor
            The coordinates of the target nodes.
        radius : float
            The cut-off radius for connecting nodes.
        max_num_neighbours : int
            The maximum number of nearest neighbours to consider for each target node.
        skip_flip : bool, optional
            Whether to skip flipping the edge index. Defaults to False. This is
            useful to avoid duplicated operations in reversed edge builders.

        Returns
        -------
        torch.Tensor
            The computed edge index.
        """
        edge_index = pyg_radius(source_coords, target_coords, r=radius, max_num_neighbors=max_num_neighbours)

        if not skip_flip:
            edge_index = torch.flip(edge_index, [0])

        return edge_index

    def _compute_adj_matrix_sklearn(
        self,
        source_coords: torch.Tensor,
        target_coords: torch.Tensor,
        radius: float,
        max_num_neighbours: int,
    ) -> torch.Tensor:
        """Compute the adjacency matrix using sklearn's radius-based implementation.

        If the number of actual neighbors is greater than :obj:`max_num_neighbors`,
        only the nearest neighbours are returned.

        Parameters
        ----------
        source_coords : torch.Tensor
            The coordinates of the source nodes.
        target_coords : torch.Tensor
            The coordinates of the target nodes.
        radius : float
            The cut-off radius for connecting nodes.
        max_num_neighbours : int
            The maximum number of nearest neighbours to consider for each target node.

        Returns
        -------
        torch.Tensor
            The computed adjacency matrix.
        """
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
        return super().compute_edge_index_from_coords(target_coords, source_coords, skip_flip=True, **kwargs)


def random_subsample_per_group(groups: np.ndarray, max_per_group: int, rng: np.random.Generator) -> np.ndarray:
    """Pick at most ``max_per_group`` random elements of each group.

    Parameters
    ----------
    groups : np.ndarray
        Group label of each element.
    max_per_group : int
        Maximum number of elements kept per group.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Sorted indices of the kept elements.
    """
    if len(groups) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((rng.random(len(groups)), groups))  # by group, random within group
    sorted_groups = groups[order]
    starts = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    rank = np.arange(len(groups)) - np.repeat(starts, np.diff(np.r_[starts, len(groups)]))
    return np.sort(order[rank < max_per_group])


class RandomSampledCutOffEdges(CutOffEdges):
    """Cut-off edges keeping a random sample of the neighbours within the radius.

    For each target node, finds the source nodes within the cut-off radius and keeps
    a random sample of ``num_neighbours`` of them (all of them if there are fewer).
    Unlike :class:`CutOffEdges`, which keeps the nearest neighbours, each target node
    sees a uniform sample of its whole neighbourhood. This matters when the source
    nodes are much denser than the target nodes, e.g. a fine limited-area grid
    connected to a coarse mesh. The sample is fixed by ``random_seed``.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    num_neighbours : int
        Number of source nodes sampled per target node.
    cutoff_factor : float | None
        Factor to multiply the grid reference distance to get the cut-off radius.
        Mutually exclusive with cutoff_distance_km.
    cutoff_distance_km : float | None
        Cutoff radius in kilometers. Mutually exclusive with cutoff_factor.
    source_mask_attr_name : str | None
        The name of the source mask attribute to filter edge connections.
    target_mask_attr_name : str | None
        The name of the target mask attribute to filter edge connections.
    max_candidates : int
        Maximum number of neighbours found per target node before sampling. If more
        source nodes are within the radius, pyg-lib keeps a random subset and sklearn
        the nearest ones. Defaults to 1024.
    random_seed : int
        Seed of the sampling. Defaults to 42.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
        num_neighbours: int,
        cutoff_factor: float | None = None,
        cutoff_distance_km: float | None = None,
        source_mask_attr_name: str | None = None,
        target_mask_attr_name: str | None = None,
        max_candidates: int = 1024,
        random_seed: int = 42,
    ) -> None:
        assert isinstance(num_neighbours, int) and num_neighbours > 0, "num_neighbours must be a positive integer."
        assert (
            isinstance(max_candidates, int) and max_candidates >= num_neighbours
        ), "max_candidates must be an integer not smaller than num_neighbours."
        super().__init__(
            source_name=source_name,
            target_name=target_name,
            cutoff_factor=cutoff_factor,
            cutoff_distance_km=cutoff_distance_km,
            source_mask_attr_name=source_mask_attr_name,
            target_mask_attr_name=target_mask_attr_name,
            max_num_neighbours=max_candidates,
        )
        self.num_neighbours = num_neighbours
        self.random_seed = random_seed

    def compute_edge_index_from_coords(
        self,
        source_coords: torch.Tensor,
        target_coords: torch.Tensor,
        skip_flip: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        edge_index = super().compute_edge_index_from_coords(source_coords, target_coords, skip_flip=skip_flip, **kwargs)
        targets = edge_index[0 if skip_flip else 1].cpu().numpy()
        keep = random_subsample_per_group(targets, self.num_neighbours, np.random.default_rng(self.random_seed))
        LOGGER.info(
            "%s kept %d of %d edges (at most %d per target node).",
            self.__class__.__name__,
            len(keep),
            edge_index.shape[1],
            self.num_neighbours,
        )
        return edge_index[:, torch.from_numpy(keep).to(edge_index.device)]
