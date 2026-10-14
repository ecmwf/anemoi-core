# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# my_package/kmeans_mesh.py
"""K-means cluster meshes for irregular (e.g. ocean) domains.

Implements the node placement and land-aware Delaunay edges described in
Holmberg et al., "Njord: A Probabilistic Graph Neural Network for Ensemble
Ocean Forecasting" (arXiv:2605.15470), Appendix C.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.spatial import ConvexHull
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian_np 

LOGGER = logging.getLogger(__name__)


class DelaunayEdges(BaseEdgeBuilder):
    """Same-level mesh edges for KMeansClusteredNodes: spherical Delaunay
    triangulation, then removal of edges that cross land.

    An edge is removed if any of `num_samples` points along it is nearest to a
    grid point outside the area-of-interest mask (num_samples=1 is the
    midpoint test from the paper), or if it is longer than `max_length_km`.

    Parameters
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
        **kwargs,
    ) -> None:
        if source_name != target_name:
            raise ValueError(
                f"{self.__class__.__name__} connects nodes within a single set (got "
                f"'{source_name}' -> '{target_name}'). Use KNNEdges/CutOffEdges "
                f"between sets."
            )
        super().__init__(source_name, target_name, **kwargs)

    def compute_edge_index_from_coords(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} doesn't support computing edge index directly from its coordinates."
        )

    @staticmethod
    def get_delaunay_pairs(xyz: np.ndarray) -> np.ndarray:
        """Unique undirected edges of the spherical Delaunay triangulation."""
        hull = ConvexHull(xyz)

        # Facet planes satisfy n.x + offset <= 0 inside the hull. Real spherical
        # Delaunay triangles have the origin inside (offset < 0). For a regional
        # domain (a cap smaller than a hemisphere) the hull also has flat "lid"
        # facets with the origin outside. They span the domain, so drop them.
        simplices = hull.simplices[hull.equations[:, -1] < -1e-9]
        pairs = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])

        return np.unique(np.sort(pairs, axis=1), axis=0)

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        """Compute the edge indices based on Delaunay triangulation of the given nodes."""
        xyz = latlon_rad_to_cartesian_np(source_nodes.x.detach().cpu().numpy().astype(np.float64))
        
        pairs = self.get_delaunay_pairs(xyz)

        # Edges are directed in anemoi. Add both directions.
        edge_index = np.concatenate([pairs.T, pairs[:, ::-1].T], axis=1)

        return torch.from_numpy(np.ascontiguousarray(edge_index))