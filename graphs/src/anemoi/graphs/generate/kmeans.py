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

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian_np

LOGGER = logging.getLogger(__name__)


class KMeansClusterBuilder:
    """K-means cluster builder for reference nodes on a unit sphere.

    Methods
    -------
    fit(graph: HeteroData)
        Gather the reference node coordinates (and weights) from the graph.
    get_clusters(ratio: float, **kwargs) -> np.ndarray
        Cluster the reference nodes and return the cluster centres, in radians.
    """

    def __init__(
        self,
        reference_node_name: str,
        mask_attr_name: str | None = None,
        node_weight_attribute_name: str | None = None,
    ):
        """Initialisation of the KMeansClusterBuilder."""
        self.reference_node_name = reference_node_name
        self.mask_attr_name = mask_attr_name
        self.node_weight_attribute_name = node_weight_attribute_name

        self._backend = None
        self._coords_xyz = None
        self._weights = None

    def _get_reference_coords(self, graph: HeteroData) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Retrieve coordinates from the reference nodes.

        Parameters
        ----------
        graph : HeteroData
            Graph object containing the reference nodes.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Latitude and longitude of the reference nodes in radians, of shape
            (N_ref, 2), and their weights of shape (N_ref,), if requested.
        """
        assert (
            self.reference_node_name in graph.node_types
        ), f'Reference node "{self.reference_node_name}" not found in the graph.'

        coords_rad = graph[self.reference_node_name].x
        if self.mask_attr_name is not None:
            assert (
                self.mask_attr_name in graph[self.reference_node_name].node_attrs()
            ), f'Mask attribute "{self.mask_attr_name}" not found in the reference nodes.'
            mask = graph[self.reference_node_name][self.mask_attr_name].squeeze()
            coords_rad = coords_rad[mask]

        if self.node_weight_attribute_name is not None:
            assert (
                self.node_weight_attribute_name in graph[self.reference_node_name].node_attrs()
            ), f'Node weight attribute "{self.node_weight_attribute_name}" not found in the reference nodes.'
            weights = graph[self.reference_node_name][self.node_weight_attribute_name].squeeze()
            if self.mask_attr_name is not None:
                weights = weights[mask]
        else:
            weights = None

        return coords_rad, weights

    def fit(self, graph: HeteroData) -> None:
        """Gather the reference nodes to cluster from the graph."""
        reference_mask_str = self.reference_node_name
        if self.mask_attr_name is not None:
            reference_mask_str += f"[{self.mask_attr_name}]"

        coords_rad, weights = self._get_reference_coords(graph)
        LOGGER.info(
            "Fitting %s with %d reference nodes from %s.",
            self.__class__.__name__,
            len(coords_rad),
            reference_mask_str,
        )
        LOGGER.debug("Reference nodes live on %s", coords_rad.device)

        self._coords_xyz = latlon_rad_to_cartesian_np(coords_rad.detach().cpu().numpy().astype(np.float64))
        self._weights = None if weights is None else weights.detach().cpu().numpy().astype(np.float64)

    def get_clusters(self, ratio: float, **kwargs) -> np.ndarray:
        """Get the cluster centers in latitude and longitude (radians)."""
        if self._coords_xyz is None:
            raise RuntimeError("No coordinates available. Fit the model first.")

        num_points = len(self._coords_xyz)
        num_clusters = max(1, round(num_points / ratio))
        self._backend = MiniBatchKMeans(n_clusters=num_clusters, **kwargs)
        LOGGER.info(
            "Fitting %s with %d clusters for %d reference points.",
            self._backend.__class__.__name__,
            num_clusters,
            num_points,
        )

        self._backend.fit(self._coords_xyz, sample_weight=self._weights)
        return cartesian_to_latlon_rad(self._backend.cluster_centers_)
