# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.kmeans import KMeansClusterBuilder
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class KMeansClusteredNodes(BaseNodeBuilder):
    """Nodes at the centroids of a weighted spherical K-means clustering.

    The clustering is run over a reference node set, optionally restricted to a
    mask (e.g. the sea points).

    Parameters
    ----------
    reference_node_name : str
        Nodes to cluster. Must be created earlier in the graph config. Can be
        another KMeansClusteredNodes set, which gives the next hierarchy level.
    ratio : float
        Refinement factor: number of reference points per cluster
        (num_nodes = round(num_reference_points / ratio)).
    name : str
        Name of the new node set (injected by the graph creator).
    mask_attr_name : str, optional
        Boolean node attribute on the reference nodes defining the area of
        interest (True = include, e.g. sea points). Points outside the mask are
        excluded from the clustering.
    node_weight_attribute_name : str, optional
        Name of the node attribute to use as weights for the clustering. If not
        specified, defaults to uniform weighting.
    kmeans_kwargs : dict, optional
        Keyword arguments forwarded to `sklearn.cluster.MiniBatchKMeans`, such as
        `max_iter` or `random_state` (set it to make the graph reproducible).
    """

    def __init__(
        self,
        reference_node_name: str,
        ratio: float,
        name: str,
        mask_attr_name: str | None = None,
        node_weight_attribute_name: str | None = None,
        kmeans_kwargs: dict | None = None,
    ) -> None:
        if ratio <= 1:
            raise ValueError(f"ratio must be > 1, got {ratio}.")
        super().__init__(name)
        self.reference_node_name = reference_node_name
        self.ratio = ratio
        self.mask_attr_name = mask_attr_name
        self.node_weight_attribute_name = node_weight_attribute_name

        self.kmeans_cluster = KMeansClusterBuilder(
            reference_node_name=reference_node_name,
            mask_attr_name=mask_attr_name,
            node_weight_attribute_name=node_weight_attribute_name,
        )
        self.kmeans_kwargs = kmeans_kwargs or {}

    def register_nodes(self, graph: HeteroData) -> HeteroData:
        self.kmeans_cluster.fit(graph)
        return super().register_nodes(graph)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the K-means cluster centers in latitude and longitude (radians)."""
        return torch.tensor(self.kmeans_cluster.get_clusters(self.ratio, **self.kmeans_kwargs), dtype=torch.float32)
