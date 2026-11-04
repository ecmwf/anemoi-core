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

_EARTH_RADIUS_M = 6_371_000.0
# Accepted nodes are added to the BallTree in batches of this size; until then they are
# checked by brute force.
_TREE_REBUILD_INTERVAL = 1000


def _haversine_rad(coord: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Great-circle distance (radians) from one [lat, lon] point to an array of points."""
    dlat = others[:, 0] - coord[0]
    dlon = others[:, 1] - coord[1]
    a = np.sin(dlat / 2) ** 2 + np.cos(coord[0]) * np.cos(others[:, 0]) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _poisson_disk_sample(
    candidates: np.ndarray,
    reference_coords: np.ndarray,
    n_target: int,
    min_dist_rad: float,
) -> np.ndarray:
    """Accept candidates, in order, that keep a minimum distance to all other nodes.

    A candidate is accepted only if it is farther than ``min_dist_rad`` from every
    node in ``reference_coords`` and from every candidate accepted before it.

    Parameters
    ----------
    candidates : np.ndarray, shape (M, 2)
        Candidate coordinates [lat, lon] in radians, in order of priority.
    reference_coords : np.ndarray, shape (N, 2)
        Existing node coordinates [lat, lon] in radians. N can be 0.
    n_target : int
        Stop after accepting this many nodes.
    min_dist_rad : float
        Minimum allowed distance in radians.

    Returns
    -------
    np.ndarray, shape (K, 2)
        Accepted coordinates, K <= n_target.
    """
    reference_coords = np.asarray(reference_coords, dtype=float).reshape(-1, 2)
    tree_points = [reference_coords]
    tree = BallTree(reference_coords, metric="haversine") if len(reference_coords) else None
    accepted: list[np.ndarray] = []
    pending: list[np.ndarray] = []  # accepted but not yet in the tree

    for coord in candidates:
        if len(accepted) >= n_target:
            break
        if tree is not None and tree.query(coord.reshape(1, -1), k=1)[0][0, 0] <= min_dist_rad:
            continue
        if pending and _haversine_rad(coord, np.asarray(pending)).min() <= min_dist_rad:
            continue
        accepted.append(coord)
        pending.append(coord)
        if len(pending) >= _TREE_REBUILD_INTERVAL:
            tree_points.append(np.asarray(pending))
            pending = []
            tree = BallTree(np.concatenate(tree_points), metric="haversine")

    return np.array(accepted, dtype=np.float32) if accepted else np.empty((0, 2), dtype=np.float32)


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


class ProbabilisticOrographyIcosahedralNodes(BaseOrographyNodeBuilder):
    """Stretched icosahedral mesh with extra nodes over the steepest terrain.

    Starts from the same mesh as :class:`StretchedTriNodes` and adds
    ``extra_nodes_percent`` % more nodes inside the limited area. Candidates come from
    a 0.02 deg lat/lon grid over the limited area and are taken from the steepest to
    the flattest, each accepted only if it is at least ``min_dist_factor`` times the
    ``lam_resolution`` node spacing away from all other nodes.

    Unlike :class:`AdaptiveOrographyTriNodes`, the extra nodes are not placed at edge
    midpoints, so they follow the terrain more closely but do not lie on an
    icosahedron and do not support ``MultiScaleEdges``. The result is deterministic.

    Parameters
    ----------
    global_resolution : int
        Icosphere subdivision level for nodes outside the LAM domain.
    lam_resolution : int
        Icosphere subdivision level for the base LAM mesh.
    reference_node_name : str
        Name of the reference node set used to define the LAM boundary.
    orography_path : str
        Path to a NetCDF file with a 2-D orography field.
    name : str
        Key for the node set in the HeteroData graph.
    extra_nodes_percent : float, optional
        Extra nodes as a percentage of the LAM base nodes. Default 10.
    min_dist_factor : float, optional
        Minimum distance between nodes as a fraction of the ``lam_resolution`` node
        spacing. Default 0.5.
    mask_attr_name : str or None, optional
        Node attribute to use as the LAM boundary mask.
    margin_radius_km : float, optional
        Buffer (km) around the reference nodes when building the LAM mask.
    """

    candidate_resolution_deg: float = 0.02

    def __init__(
        self,
        global_resolution: int,
        lam_resolution: int,
        reference_node_name: str,
        orography_path: str,
        name: str,
        extra_nodes_percent: float = 10.0,
        min_dist_factor: float = 0.5,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:
        super().__init__(name, orography_path, reference_node_name, mask_attr_name, margin_radius_km)
        assert extra_nodes_percent >= 0, f"extra_nodes_percent must be non-negative, got {extra_nodes_percent}"
        assert min_dist_factor > 0, f"min_dist_factor must be positive, got {min_dist_factor}"
        self.global_resolution = global_resolution
        self.lam_resolution = lam_resolution
        self.extra_nodes_percent = extra_nodes_percent / 100.0
        self.min_dist_factor = min_dist_factor

    def get_coordinates(self) -> torch.Tensor:
        """Build the stretched mesh plus the extra terrain-following nodes.

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
        lam_base_coords = base_coords[self.area_mask_builder.get_mask(base_coords).cpu().numpy()]

        n_extra = int(len(lam_base_coords) * self.extra_nodes_percent)
        LOGGER.info(
            "%s: %d LAM base nodes, adding up to %d extra nodes",
            self.__class__.__name__,
            len(lam_base_coords),
            n_extra,
        )
        if n_extra == 0 or len(lam_base_coords) == 0:
            return torch.tensor(base_coords, dtype=torch.float32)

        pad = np.deg2rad(0.5)
        candidates = _regular_grid(
            (lam_base_coords[:, 0].min() - pad, lam_base_coords[:, 0].max() + pad),
            (lam_base_coords[:, 1].min() - pad, lam_base_coords[:, 1].max() + pad),
            self.candidate_resolution_deg,
        )
        candidates = candidates[self.area_mask_builder.get_mask(candidates).cpu().numpy()]

        gradients = compute_orography_gradient(self.orography_path, candidates)
        steep_first = np.argsort(-gradients, kind="stable")
        candidates = candidates[steep_first][gradients[steep_first] > 0]

        min_dist = self.min_dist_factor * np.sqrt(4 * np.pi / (10 * 4**self.lam_resolution))
        extra = _poisson_disk_sample(candidates, base_coords, n_extra, min_dist)
        LOGGER.info("%s: placed %d/%d extra nodes", self.__class__.__name__, len(extra), n_extra)

        return torch.tensor(np.concatenate([base_coords, extra], axis=0), dtype=torch.float32)


class ProbabilisticOrographyNodes(BaseOrographyNodeBuilder):
    """Free-form mesh whose node density follows the orography.

    Places ``num_nodes_lam`` nodes inside the limited area and ``num_nodes_global``
    outside it, with no icosahedron. Candidates come from a 0.02 deg grid over the
    limited area and a 0.25 deg global grid. Each region's candidates are visited in a
    random order weighted by ``(slope + 1e-6) ** (1 / compression)``, and each is
    accepted if it is at least ``min_dist_m`` from all nodes placed so far. The
    limited area is filled first.

    A small ``compression`` concentrates nodes on steep terrain (e.g. 0.1 gives
    weights proportional to slope**10); a large one (e.g. 10) gives a nearly uniform
    distribution. Weights above the 99th percentile are clipped, so the steepest
    candidates do not take the whole budget. Outside the orography file the slope is
    0, so the global nodes are spread uniformly.

    Parameters
    ----------
    name : str
        Key for the node set in the HeteroData graph.
    orography_path : str
        Path to a NetCDF file with a 2-D orography field.
    reference_node_name : str
        Name of the reference node set used to define the LAM boundary.
    num_nodes_global : int
        Number of nodes to place outside the LAM domain.
    num_nodes_lam : int
        Number of nodes to place inside the LAM domain.
    compression_global : float
        Compression for the global nodes. Large values give a uniform distribution.
    compression_lam : float
        Compression for the LAM nodes. Small values follow the terrain more closely.
    min_dist_m : float, optional
        Minimum distance between any two nodes, in metres. Default 5 000 m.
    random_seed : int, optional
        Seed of the random ordering, so the same configuration gives the same mesh.
        Default 42.
    mask_attr_name : str or None, optional
        Node attribute to use as the LAM boundary mask.
    margin_radius_km : float, optional
        Buffer (km) around the reference nodes when building the LAM mask.
    """

    global_candidate_resolution_deg: float = 0.25
    lam_candidate_resolution_deg: float = 0.02

    def __init__(
        self,
        name: str,
        orography_path: str,
        reference_node_name: str,
        num_nodes_global: int,
        num_nodes_lam: int,
        compression_global: float,
        compression_lam: float,
        min_dist_m: float = 5_000.0,
        random_seed: int = 42,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:
        super().__init__(name, orography_path, reference_node_name, mask_attr_name, margin_radius_km)
        assert num_nodes_global >= 0 and num_nodes_lam >= 0, "Number of nodes must be non-negative."
        assert compression_global > 0 and compression_lam > 0, "Compression must be positive."
        assert min_dist_m > 0, f"min_dist_m must be positive, got {min_dist_m}"
        self.num_nodes_global = num_nodes_global
        self.num_nodes_lam = num_nodes_lam
        self.compression_global = compression_global
        self.compression_lam = compression_lam
        self.min_dist_rad = min_dist_m / _EARTH_RADIUS_M
        self.random_seed = random_seed

    @staticmethod
    def _weighted_random_order(log_weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Random permutation where higher weights tend to come first (Gumbel top-k)."""
        return np.argsort(-(log_weights + rng.gumbel(size=len(log_weights))))

    @staticmethod
    def _priority_log_weights(gradients: np.ndarray, compression: float) -> np.ndarray:
        """Log of ``(gradient + 1e-6) ** (1 / compression)``, clipped at its 99th percentile.

        Working in log space keeps small slopes distinguishable: with compression 0.1
        the weights themselves underflow to the same tiny value for most terrain.
        """
        log_weights = np.log(gradients + 1e-6) / compression
        return np.minimum(log_weights, np.percentile(log_weights, 99))

    def get_coordinates(self) -> torch.Tensor:
        """Build the free-form mesh.

        Returns
        -------
        torch.Tensor, shape (num_nodes, 2)
            Node coordinates in radians [lat, lon].
        """
        rng = np.random.default_rng(self.random_seed)

        global_grid = _regular_grid((-np.pi / 2, np.pi / 2), (-np.pi, np.pi), self.global_candidate_resolution_deg)
        in_lam = self.area_mask_builder.get_mask(global_grid).cpu().numpy()
        global_candidates = global_grid[~in_lam]

        lam_coarse = global_grid[in_lam]
        if len(lam_coarse) > 0:
            pad = np.deg2rad(0.5)
            lam_grid = _regular_grid(
                (lam_coarse[:, 0].min() - pad, lam_coarse[:, 0].max() + pad),
                (lam_coarse[:, 1].min() - pad, lam_coarse[:, 1].max() + pad),
                self.lam_candidate_resolution_deg,
            )
            lam_candidates = lam_grid[self.area_mask_builder.get_mask(lam_grid).cpu().numpy()]
        else:
            LOGGER.warning("No LAM candidates found, check reference_node_name and mask_attr_name.")
            lam_candidates = np.empty((0, 2))

        nodes = np.empty((0, 2), dtype=np.float32)
        for candidates, target_count, compression, label in [
            (lam_candidates, self.num_nodes_lam, self.compression_lam, "LAM"),
            (global_candidates, self.num_nodes_global, self.compression_global, "global"),
        ]:
            if len(candidates) == 0 or target_count == 0:
                continue
            gradients = compute_orography_gradient(self.orography_path, candidates)
            order = self._weighted_random_order(self._priority_log_weights(gradients, compression), rng)
            accepted = _poisson_disk_sample(candidates[order], nodes, target_count, self.min_dist_rad)
            LOGGER.info("%s %s: placed %d/%d nodes", self.__class__.__name__, label, len(accepted), target_count)
            nodes = np.concatenate([nodes, accepted], axis=0)

        return torch.tensor(nodes, dtype=torch.float32)


def _regular_grid(lat_range: tuple[float, float], lon_range: tuple[float, float], res_deg: float) -> np.ndarray:
    """Regular [lat, lon] grid in radians over the given ranges, spaced ``res_deg`` degrees."""
    res = np.deg2rad(res_deg)
    lats = np.arange(lat_range[0], lat_range[1] + res * 0.5, res)
    lons = np.arange(lon_range[0], lon_range[1] + res * 0.5, res)
    lon_g, lat_g = np.meshgrid(lons, lats)
    return np.stack([lat_g.ravel(), lon_g.ravel()], axis=-1)
