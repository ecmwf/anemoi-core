# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.spatial import Delaunay
from scipy.spatial import QhullError
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder

LOGGER = logging.getLogger(__name__)


class TriangulationEdges(BaseEdgeBuilder):
    """Build hidden-hidden edges from a 2D Delaunay triangulation.

    This edge builder is intended for regionally defined hidden meshes where
    nodes are represented by latitude/longitude coordinates in radians.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
        **kwargs,
    ) -> None:
        super().__init__(source_name, target_name)
        assert source_name == target_name, f"{self.__class__.__name__} requires source and target nodes to be the same."

    @staticmethod
    def _project_local_plane(coords_rad: np.ndarray) -> np.ndarray:
        """Project lat/lon radians onto a local plane for triangulation."""
        lat = coords_rad[:, 0].astype(np.float64)
        lon = np.unwrap(coords_rad[:, 1].astype(np.float64))

        lat0 = float(np.mean(lat))
        lon0 = float(np.mean(lon))

        x = (lon - lon0) * np.cos(lat0)
        y = lat - lat0

        return np.stack([x, y], axis=1)

    @staticmethod
    def _empty() -> torch.Tensor:
        return torch.empty((2, 0), dtype=torch.int64)

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        assert source_nodes.num_nodes == target_nodes.num_nodes, "TriangulationEdges expects identical source/target sets."

        coords_rad = source_nodes.x.detach().cpu().numpy()
        num_nodes = int(coords_rad.shape[0])

        if num_nodes < 3:
            LOGGER.warning("TriangulationEdges requires >=3 nodes. Received %d nodes; no edges will be created.", num_nodes)
            return self._empty()

        points_2d = self._project_local_plane(coords_rad)

        try:
            triangulation = Delaunay(points_2d, qhull_options="QJ")
        except QhullError as exc:
            LOGGER.warning("TriangulationEdges failed to triangulate nodes (%s). Returning an empty edge set.", exc)
            return self._empty()

        simplices = triangulation.simplices
        if simplices.size == 0:
            return self._empty()

        tri_edges = np.concatenate(
            [
                simplices[:, [0, 1]],
                simplices[:, [1, 2]],
                simplices[:, [2, 0]],
            ],
            axis=0,
        )

        # Add reverse directions and remove duplicates for directed message passing.
        directed_edges = np.concatenate([tri_edges, tri_edges[:, [1, 0]]], axis=0)
        directed_edges = np.unique(directed_edges, axis=0)

        edge_index = torch.from_numpy(directed_edges.T)

        LOGGER.info(
            "TriangulationEdges created %d directed edges from %d nodes.",
            edge_index.shape[1],
            num_nodes,
        )

        return edge_index.to(torch.int64)
