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
from typing import Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.concentric import create_concentric_mesh
from anemoi.graphs.generate.concentric import create_stretched_concentric
from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.graphs.nodes.builders.from_refined_icosahedron import IcosahedralNodes

LOGGER = logging.getLogger(__name__)


class ConcentricNodes(BaseNodeBuilder, ABC):
    """
    Nodes based on concentric circles around a specific location.

    Attributes
    ----------
    center_coords: tuple(int, int)
        Latitude and Longitude of center.

    n_circles: int
        Number of circles to generate around the center.

    base_dist: float = 0.1
        Distance of the first circle fromn the center in km.

    min_n_points: int = 64,
        Minimum number of points in the further away circle.

    max_n_points: int = 1024
        Maximum number of points in the innermost circle.

    """

    def __init__(
        self,
        name: str,
        center_coords: Tuple[int, int],
        n_circles: int = 200,
        base_dist: float = 0.1,
        min_n_points: int = 64,
        max_n_points: int = 1024,
    ) -> None:

        super().__init__(name)
        self.center_coords = center_coords
        self.n_circles = n_circles
        self.base_dist = base_dist
        self.min_n_points = min_n_points
        self.max_n_points = max_n_points

        self.hidden_attributes = BaseNodeBuilder.hidden_attributes | {
            "center_coords",
            "n_circles",
            "base_dist",
            "min_n_points",
            "max_n_points",
            "nx_graph",
            "node_ordering",
            "area_mask_builder",
        }

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        self.nx_graph, coords_rad, self.node_ordering = self.create_nodes()
        return torch.tensor(coords_rad[self.node_ordering], dtype=torch.float32)

    def create_nodes(self) -> tuple[nx.DiGraph, np.ndarray, list[int]]:
        return create_concentric_mesh(
            self.center_coords,
            self.n_circles,
            self.base_dist,
            self.min_n_points,
            self.max_n_points,
        )


class StretchedConcentricNodes(IcosahedralNodes):
    """
    Nodes based on iterative refinements of an icosahedron for the LAM area,
    and concentric mesh for the global.

    Attributes
    ----------
    area_mask_builder : KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    def __init__(
        self,
        lam_resolution: int,
        n_circles: int,
        base_dist: float,
        min_n_points: int,
        max_n_points: int,
        name: str,
        reference_node_name: str,
        mask_attr_name: str,
        margin_radius_km: float = 100.0,
    ) -> None:

        super().__init__(lam_resolution, name)
        self.n_circles = n_circles
        self.base_dist = base_dist
        self.min_n_points = min_n_points
        self.max_n_points = max_n_points

        self.hidden_attributes = self.hidden_attributes | {
            "n_circles",
            "base_dist",
            "min_n_points",
            "max_n_points",
        }

        self.area_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

    def register_nodes(self, graph: HeteroData) -> None:
        self.area_mask_builder.fit(graph)
        return super().register_nodes(graph)

    def create_nodes(self) -> tuple[nx.Graph, np.ndarray, list[int]]:
        return create_stretched_concentric(
            self.n_circles,
            self.base_dist,
            self.min_n_points,
            self.max_n_points,
            max(self.resolutions),
            self.area_mask_builder,
        )
