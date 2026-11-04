# (C) Copyright 2026 Anemoi contributors.
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
from sklearn.neighbors import BallTree
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.masks import AreaMaskBuilder
from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad
from anemoi.graphs.generate.tri_icosahedron import create_stretched_tri_nodes
from anemoi.graphs.generate.utils import compute_orography_gradient
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class BaseOrographyNodeBuilder(BaseNodeBuilder):
    """Base class for node builders driven by an orography field and a LAM domain.

    Subclasses must implement :meth:`get_coordinates`.

    Parameters
    ----------
    name : str
        Key for the node set in the HeteroData graph.
    orography_path : str
        Path to a NetCDF file with a 2-D orography field.
    reference_node_name : str
        Name of the reference node set used to define the LAM boundary.
    mask_attr_name : str or None, optional
        Node attribute to use as the LAM boundary mask.
    margin_radius_km : float, optional
        Buffer (km) around the reference nodes when building the LAM mask.
    """

    def __init__(
        self,
        name: str,
        orography_path: str,
        reference_node_name: str,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:
        super().__init__(name)
        self.orography_path = orography_path
        self.area_mask_builder = AreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)
        self.hidden_attributes = BaseNodeBuilder.hidden_attributes | {"area_mask_builder"}

    def register_nodes(self, graph: HeteroData) -> HeteroData:
        self.area_mask_builder.fit(graph)
        return super().register_nodes(graph)


class AdaptiveOrographyTriNodes(BaseOrographyNodeBuilder):
    """Hidden mesh with orography-adaptive node density.

    Builds a StretchedTriNodes base mesh (``global_resolution`` outside LAM,
    ``lam_resolution`` inside LAM) and inserts extra nodes at the sphere-surface
    midpoints of all edges whose both endpoints are LAM nodes with orographic
    gradient above ``gradient_threshold_percentile``.  The midpoint strategy
    effectively halves the local mesh spacing in steep-terrain areas without
    building a full next-resolution icosphere (which would have 4× as many
    global nodes).

    Encoder/decoder edges (CutOffEdges / KNNEdges) adapt automatically to the
    denser mesh.

    Compatible with ``MultiScaleEdges(x_hops=1, scale_resolutions=lam_resolution+1)``.
    Adaptive midpoints sit exactly at icosphere(lam_resolution+1) vertex positions,
    so all nodes can be assigned icosphere-vertex indices and
    ``StretchedTriNodesEdgeBuilder`` builds full multi-scale connectivity.

    Parameters
    ----------
    global_resolution : int
        Icosphere subdivision level for nodes outside the LAM domain.
    lam_resolution : int
        Icosphere subdivision level for the base LAM mesh.
    reference_node_name : str
        Name of the reference node set used to define the LAM boundary.
    orography_path : str
        Path to a NetCDF file with a 2-D orography field.  The variable must
        be named one of: orog, z, orography, hsurf, topography, fi
        (case-insensitive).  Geopotential values (>10 000 J/kg) are
        automatically converted to metres using g = 9.806 65 m/s².
    name : str
        Key for the node set in the HeteroData graph.
    gradient_threshold_percentile : float, optional
        Percentile of the orographic gradient (0–100) above which adaptive
        refinement is applied.  Default 75 — refines the steepest 25 % of the
        LAM domain.
    mask_attr_name : str or None, optional
        Node attribute to use as the LAM boundary mask.
    margin_radius_km : float, optional
        Buffer (km) around the reference nodes when building the LAM mask.
    """

    # Required by MultiScaleEdges: StretchedTriNodesEdgeBuilder handles 1-hop
    # multi-scale edges for any node set whose coordinates lie on the trimesh
    # icosphere grid.  Adaptive midpoints are midpoints of lam_resolution edges,
    # which ARE the vertex positions of icosphere(lam_resolution+1).
    multi_scale_edge_cls: str = "anemoi.graphs.generate.multi_scale_edges.StretchedTriNodesEdgeBuilder"

    def __init__(
        self,
        global_resolution: int,
        lam_resolution: int,
        reference_node_name: str,
        orography_path: str,
        name: str,
        gradient_threshold_percentile: float = 75.0,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:
        super().__init__(name, orography_path, reference_node_name, mask_attr_name, margin_radius_km)
        self.global_resolution = global_resolution
        self.lam_resolution = lam_resolution
        self.gradient_threshold_percentile = gradient_threshold_percentile
        # Register for MultiScaleEdges: node_ordering maps each node to its index in
        # icosphere(lam_resolution+1); resolutions lists all resolution levels present.
        self.hidden_attributes = self.hidden_attributes | {"resolutions", "node_ordering"}

    def _snap_and_record_ordering(self, combined: np.ndarray) -> torch.Tensor:
        """Map nodes to icosphere(lam_resolution+1), snap coords, store ordering metadata.

        Sets ``self.resolutions`` and ``self.node_ordering`` so that
        ``MultiScaleEdges(x_hops=1, scale_resolutions=lam_resolution+1)`` can
        build full multi-scale processor edges over this node set.

        Returns the node coordinates snapped to exact icosphere vertex positions
        (required by the exact-equality assertion in ``add_1_hop_edges``).
        """
        import trimesh

        adaptive_resolution = self.lam_resolution + 1
        sphere_hires = trimesh.creation.icosphere(subdivisions=adaptive_resolution, radius=1.0)
        coords_hires = cartesian_to_latlon_rad(sphere_hires.vertices)  # float32, (V, 2)

        hires_tree = BallTree(coords_hires, metric="haversine")
        _, nn_idx = hires_tree.query(combined, k=1)
        self.node_ordering = nn_idx[:, 0].tolist()

        # Snap to exact icosphere float32 values so add_1_hop_edges assertion passes.
        combined_snapped = coords_hires[self.node_ordering]
        self.resolutions = list(range(adaptive_resolution + 1))  # [0, ..., lam_resolution+1]

        LOGGER.info(
            "AdaptiveOrographyTriNodes: mapped %d nodes to icosphere(res=%d); "
            "MultiScaleEdges(scale_resolutions=%d) supported.",
            len(combined_snapped),
            adaptive_resolution,
            adaptive_resolution,
        )
        return torch.tensor(combined_snapped, dtype=torch.float32)

    def _get_midpoint_coords(
        self,
        lam_base_coords: np.ndarray,
        high_gradient_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute sphere-surface midpoints for edges of high-gradient LAM nodes.

        For each edge of the ``lam_resolution`` icosphere whose both endpoints
        fall within the LAM domain AND are above the gradient threshold, compute
        the mid-point on the sphere surface.  This effectively halves the mesh
        spacing in the target region without building a full resolution+1 icosphere
        (which would have 4× as many global nodes and be prohibitively expensive).

        Parameters
        ----------
        lam_base_coords : np.ndarray, shape (N_lam, 2)
            Lat/lon coords (radians) of LAM nodes from the base mesh.
        high_gradient_mask : np.ndarray, shape (N_lam,), dtype bool
            True for LAM nodes that exceed the gradient threshold.

        Returns
        -------
        np.ndarray, shape (M, 2)
            Midpoint coordinates in radians.
        """
        import trimesh

        sphere = trimesh.creation.icosphere(subdivisions=self.lam_resolution, radius=1.0)
        all_sphere_coords = cartesian_to_latlon_rad(sphere.vertices)

        sphere_tree = BallTree(all_sphere_coords, metric="haversine")
        _, nn_idx = sphere_tree.query(lam_base_coords, k=1)
        lam_sphere_indices = nn_idx[:, 0]

        hg_sphere_arr = np.zeros(len(sphere.vertices), dtype=bool)
        hg_sphere_arr[lam_sphere_indices[high_gradient_mask]] = True

        edges = sphere.edges_unique
        both_hg = hg_sphere_arr[edges[:, 0]] & hg_sphere_arr[edges[:, 1]]
        hg_edges = edges[both_hg]

        if len(hg_edges) == 0:
            LOGGER.warning("No high-gradient edges found; adaptive layer is empty.")
            return np.empty((0, 2), dtype=np.float32)

        verts = sphere.vertices
        mid_cart = (verts[hg_edges[:, 0]] + verts[hg_edges[:, 1]]) / 2.0
        mid_cart /= np.linalg.norm(mid_cart, axis=1, keepdims=True)
        return cartesian_to_latlon_rad(mid_cart)

    def get_coordinates(self) -> torch.Tensor:
        """Build orography-adaptive hidden mesh coordinates.

        Returns
        -------
        torch.Tensor, shape (num_nodes, 2)
            Node coordinates in radians [lat, lon].
        """
        _, all_coords, ordering = create_stretched_tri_nodes(
            base_resolution=self.global_resolution,
            lam_resolution=self.lam_resolution,
            area_mask_builder=self.area_mask_builder,
        )
        base_coords = all_coords[ordering]

        base_lam_mask = self.area_mask_builder.get_mask(base_coords).cpu().numpy()
        lam_base_coords = base_coords[base_lam_mask]

        LOGGER.info(
            "Computing orographic gradient from %s for %d LAM nodes",
            self.orography_path,
            len(lam_base_coords),
        )
        gradients = compute_orography_gradient(self.orography_path, lam_base_coords)
        threshold = np.percentile(gradients, self.gradient_threshold_percentile)
        # Flat nodes are never refined, even when most of the LAM is flat and the threshold is 0.
        high_grad_mask = (gradients >= threshold) & (gradients > 0)
        LOGGER.info(
            "Gradient threshold (p%.0f): %.5f — refining steepest %.0f%% (%d/%d LAM nodes)",
            self.gradient_threshold_percentile,
            threshold,
            100.0 - self.gradient_threshold_percentile,
            int(high_grad_mask.sum()),
            len(gradients),
        )

        mid_coords = self._get_midpoint_coords(lam_base_coords, high_grad_mask)
        LOGGER.info("Edge-midpoint adaptive candidates: %d", len(mid_coords))

        if len(mid_coords) == 0:
            LOGGER.info("AdaptiveOrographyTriNodes: %d total nodes (base=%d, adaptive=0)", *[len(base_coords)] * 2)
            return self._snap_and_record_ordering(base_coords)

        mid_lam_mask = self.area_mask_builder.get_mask(mid_coords).cpu().numpy()
        mid_coords = mid_coords[mid_lam_mask]

        min_spacing_rad = 0.5 * np.sqrt(4 * np.pi / (10 * (4**self.lam_resolution)))
        base_tree = BallTree(base_coords, metric="haversine")
        dist_to_base, _ = base_tree.query(mid_coords, k=1)
        unique_mask = dist_to_base[:, 0] > min_spacing_rad
        adaptive_coords = mid_coords[unique_mask]
        LOGGER.info(
            "Unique adaptive nodes after dedup: %d (removed %d duplicates)",
            len(adaptive_coords),
            int((~unique_mask).sum()),
        )

        combined = np.concatenate([base_coords, adaptive_coords], axis=0)
        LOGGER.info(
            "AdaptiveOrographyTriNodes: %d total nodes (base=%d, adaptive=%d)",
            len(combined),
            len(base_coords),
            len(adaptive_coords),
        )
        return self._snap_and_record_ordering(combined)
