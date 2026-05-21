# (C) Copyright 2024 Anemoi contributors.
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
from torch_geometric.data import HeteroData
from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.generate.utils import compute_orography_gradient
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from sklearn.neighbors import BallTree
from anemoi.graphs.generate.tri_icosahedron import create_stretched_tri_nodes

LOGGER = logging.getLogger(__name__)

_EARTH_RADIUS_M = 6_371_000.0
# Rebuild BallTree every this many accepted nodes to keep insertion O(n log n) overall
_TREE_REBUILD_INTERVAL = 1000


def _poisson_disk_sample(
    candidates: np.ndarray,
    reference_coords: np.ndarray,
    n_target: int,
    min_dist_rad: float,
) -> np.ndarray:
    """Accept candidate coordinates with minimum Poisson-disk separation.

    Iterates over ``candidates`` in order and accepts each one only if it is
    farther than ``min_dist_rad`` (radians, haversine) from all already-accepted
    nodes and all ``reference_coords``.  The BallTree is rebuilt every
    ``_TREE_REBUILD_INTERVAL`` acceptances to keep cost manageable.

    Parameters
    ----------
    candidates : np.ndarray, shape (M, 2)
        Candidate coordinates [lat, lon] in radians, pre-sorted as desired.
    reference_coords : np.ndarray, shape (N, 2)
        Existing node coordinates to respect during repulsion (e.g. base mesh).
    n_target : int
        Stop after accepting this many nodes.
    min_dist_rad : float
        Minimum allowed distance in radians.

    Returns
    -------
    np.ndarray, shape (K, 2)  where K <= n_target
    """
    accepted: list[np.ndarray] = []
    all_ref = list(reference_coords)
    tree = BallTree(reference_coords, metric="haversine")

    for coord in candidates:
        if len(accepted) >= n_target:
            break
        dist, _ = tree.query(coord.reshape(1, -1), k=1)
        if dist[0, 0] > min_dist_rad:
            accepted.append(coord)
            all_ref.append(coord)
            if len(accepted) % _TREE_REBUILD_INTERVAL == 0:
                tree = BallTree(np.array(all_ref), metric="haversine")

    return np.array(accepted, dtype=np.float32) if accepted else np.empty((0, 2), dtype=np.float32)


class BaseOrographyNodeBuilder(BaseNodeBuilder):
    """Shared base providing LAM masking and orographic gradient computation.

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
        self.area_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)
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
        from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad

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
        from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad

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

        base_lam_mask = self.area_mask_builder.get_mask(base_coords)
        lam_base_coords = base_coords[base_lam_mask]

        LOGGER.info(
            "Computing orographic gradient from %s for %d LAM nodes",
            self.orography_path,
            len(lam_base_coords),
        )
        gradients = compute_orography_gradient(self.orography_path, lam_base_coords)
        threshold = np.percentile(gradients, self.gradient_threshold_percentile)
        high_grad_mask = gradients >= threshold
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
            LOGGER.info("AdaptiveOrographyTriNodes: %d total nodes (base=%d, adaptive=0)", len(base_coords), len(base_coords))
            return self._snap_and_record_ordering(base_coords)

        mid_lam_mask = self.area_mask_builder.get_mask(mid_coords)
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
    """Icosahedron-base mesh augmented with gradient-probability-sampled extra nodes.

    Starts from a StretchedTriNodes base mesh (same as :class:`AdaptiveOrographyTriNodes`)
    and adds ``extra_nodes_percent`` % more nodes inside the LAM domain.  Candidate
    positions are drawn from a dense lat/lon grid, weighted by the normalised
    orographic gradient distribution (steeper terrain → higher probability).
    Accepted candidates must be separated by at least ``min_dist_factor`` × typical
    icosphere spacing from all existing nodes (Poisson-disk repulsion).

    Unlike :class:`AdaptiveOrographyTriNodes`, the extra nodes are not constrained to
    edge midpoints — they are placed stochastically wherever the gradient is high,
    which gives a smoother density transition at the cost of mesh irregularity.

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
        Target extra nodes as a percentage of LAM base nodes.  Default 10.
    min_dist_factor : float, optional
        Minimum distance for Poisson-disk repulsion as a fraction of the typical
        icosphere spacing at ``lam_resolution``.  Default 0.5.
    mask_attr_name : str or None, optional
        Node attribute to use as the LAM boundary mask.
    margin_radius_km : float, optional
        Buffer (km) around the reference nodes when building the LAM mask.
    """

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
        self.global_resolution = global_resolution
        self.lam_resolution = lam_resolution
        self.extra_nodes_percent = extra_nodes_percent / 100.0
        self.min_dist_factor = min_dist_factor

    def get_coordinates(self) -> torch.Tensor:
        """Build icosahedron-base + gradient-sorted adaptive mesh.

        Returns
        -------
        torch.Tensor, shape (num_nodes, 2)
            Node coordinates in radians [lat, lon].
        """
        # ── 1. Base StretchedTriNodes mesh ────────────────────────────────
        _, all_coords, ordering = create_stretched_tri_nodes(
            base_resolution=self.global_resolution,
            lam_resolution=self.lam_resolution,
            area_mask_builder=self.area_mask_builder,
        )
        base_coords = all_coords[ordering]

        base_lam_mask = self.area_mask_builder.get_mask(base_coords)
        lam_base_coords = base_coords[base_lam_mask]

        n_extra = int(len(lam_base_coords) * self.extra_nodes_percent)
        LOGGER.info(
            "ProbabilisticOrographyIcosahedralNodes: base LAM=%d, target extra=%d (%.0f%%)",
            len(lam_base_coords),
            n_extra,
            self.extra_nodes_percent * 100,
        )
        if n_extra == 0:
            return torch.tensor(base_coords, dtype=torch.float32)

        # ── 2. Fine LAM candidate grid (0.02° ≈ 2.2 km) ──────────────────
        # Build bbox from base LAM nodes; 0.25° is far too coarse to reach
        # the target extra-node budget.
        pad = np.deg2rad(0.5)
        lat_r = (lam_base_coords[:, 0].min() - pad, lam_base_coords[:, 0].max() + pad)
        lon_r = (lam_base_coords[:, 1].min() - pad, lam_base_coords[:, 1].max() + pad)
        res = np.deg2rad(0.02)
        lats = np.arange(lat_r[0], lat_r[1] + res * 0.5, res)
        lons = np.arange(lon_r[0], lon_r[1] + res * 0.5, res)
        lon_g, lat_g = np.meshgrid(lons, lats)
        fine_grid = np.stack([lat_g.ravel(), lon_g.ravel()], axis=-1)
        lam_fine_mask = self.area_mask_builder.get_mask(fine_grid)
        lam_candidates = fine_grid[lam_fine_mask]
        LOGGER.info("Fine LAM candidates: %d (0.02° grid, %.1f°×%.1f° bbox)",
                    len(lam_candidates),
                    np.rad2deg(lat_r[1] - lat_r[0]),
                    np.rad2deg(lon_r[1] - lon_r[0]))

        # ── 3. Sort candidates by gradient descending ─────────────────────
        # Steep terrain gets first pick in the Poisson-disk pass, ensuring
        # the fixed budget concentrates on Alpine ridges rather than flat areas.
        LOGGER.info("Computing gradient for %d fine LAM candidates", len(lam_candidates))
        gradients = compute_orography_gradient(self.orography_path, lam_candidates)
        sort_order = np.argsort(gradients)[::-1]
        sorted_candidates = lam_candidates[sort_order]

        # ── 4. Poisson-disk repulsion against base mesh ───────────────────
        typical_spacing = np.sqrt(4 * np.pi / (10 * (4**self.lam_resolution)))
        min_dist = typical_spacing * self.min_dist_factor

        adaptive_coords = _poisson_disk_sample(sorted_candidates, base_coords, n_extra, min_dist)
        LOGGER.info(
            "ProbabilisticOrographyIcosahedralNodes: %d/%d extra nodes placed",
            len(adaptive_coords), n_extra,
        )

        if len(adaptive_coords) == 0:
            return torch.tensor(base_coords, dtype=torch.float32)

        combined = np.concatenate([base_coords, adaptive_coords], axis=0)
        LOGGER.info(
            "ProbabilisticOrographyIcosahedralNodes: %d total nodes (base=%d, extra=%d)",
            len(combined), len(base_coords), len(adaptive_coords),
        )
        return torch.tensor(combined, dtype=torch.float32)


class ProbabilisticOrographyNodes(BaseOrographyNodeBuilder):
    """Free-form mesh with node density following orography.

    Generates a completely free-form hidden mesh — no icosphere topology — by
    sampling node positions from a dense lat/lon candidate grid with probability
    proportional to orographic gradient raised to a compression power.  Separate
    budgets and compression factors can be set for the global and LAM regions,
    allowing the LAM to be much denser and more terrain-following than the global
    background.

    The compression exponent ``c`` controls how sharply density follows terrain:
    ``p ∝ (gradient + ε)^(1/c)``.  At ``c → 0`` the distribution is nearly
    uniform; at large ``c`` almost all nodes concentrate on the steepest slopes.
    A small baseline (``ε = 1e-6``) prevents zero-probability zones in flat
    regions.

    Accepted candidates must be separated by at least ``min_dist_m`` metres from
    all already-placed nodes (Poisson-disk repulsion).

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
        Compression exponent for global node placement.  0 = uniform.
    compression_lam : float
        Compression exponent for LAM node placement.  Higher = steeper bias.
    min_dist_m : float, optional
        Minimum distance between any two nodes, in metres.  Default 5 000 m.
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
        num_nodes_global: int,
        num_nodes_lam: int,
        compression_global: float,
        compression_lam: float,
        min_dist_m: float = 5_000.0,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:
        super().__init__(name, orography_path, reference_node_name, mask_attr_name, margin_radius_km)
        self.num_nodes_global = num_nodes_global
        self.num_nodes_lam = num_nodes_lam
        self.compression_global = compression_global
        self.compression_lam = compression_lam
        self.min_dist_rad = min_dist_m / _EARTH_RADIUS_M

    @staticmethod
    def _make_grid(lat_rad_range: tuple, lon_rad_range: tuple, res_deg: float) -> np.ndarray:
        """Uniform grid in radians over the given range at ``res_deg`` degree spacing."""
        res = np.deg2rad(res_deg)
        lats = np.arange(lat_rad_range[0], lat_rad_range[1] + res * 0.5, res)
        lons = np.arange(lon_rad_range[0], lon_rad_range[1] + res * 0.5, res)
        lon_g, lat_g = np.meshgrid(lons, lats)
        return np.stack([lat_g.ravel(), lon_g.ravel()], axis=-1)

    @staticmethod
    def _gradient_probs(gradients: np.ndarray, compression: float) -> np.ndarray:
        """Convert gradient magnitudes to a probability distribution."""
        p = np.power(gradients + 1e-6, 1.0 / max(compression, 1e-9))
        return p / p.sum()

    def get_coordinates(self) -> torch.Tensor:
        """Build free-form orography-following mesh coordinates.

        Returns
        -------
        torch.Tensor, shape (num_nodes, 2)
            Node coordinates in radians [lat, lon].
        """
        # ── 1. Coarse global grid to locate LAM bounding box ──────────────
        # 0.25° grid: fine enough to find the LAM extent, cheap to compute
        global_grid = self._make_grid((-np.pi / 2, np.pi / 2), (-np.pi, np.pi), 0.25)
        is_lam_coarse = self.area_mask_builder.get_mask(global_grid)
        global_candidates = global_grid[~is_lam_coarse]
        global_grads = compute_orography_gradient(self.orography_path, global_candidates)
        LOGGER.info("Global candidates: %d (0.25° grid)", len(global_candidates))

        # ── 2. Fine LAM grid using LAM bounding box ───────────────────────
        # Target spacing for num_nodes_lam across the LAM area requires a
        # candidate grid significantly finer than the desired node spacing.
        # We use 0.02° (~2.2 km) to give dense coverage.
        lam_approx = global_grid[is_lam_coarse]
        if len(lam_approx) > 0:
            pad = np.deg2rad(0.5)
            lat_r = (lam_approx[:, 0].min() - pad, lam_approx[:, 0].max() + pad)
            lon_r = (lam_approx[:, 1].min() - pad, lam_approx[:, 1].max() + pad)
            lam_fine_grid = self._make_grid(lat_r, lon_r, 0.02)
            is_lam_fine = self.area_mask_builder.get_mask(lam_fine_grid)
            lam_candidates = lam_fine_grid[is_lam_fine]
            lam_grads = compute_orography_gradient(self.orography_path, lam_candidates)
            LOGGER.info("LAM candidates: %d (0.02° fine grid over %.2f°×%.2f° bbox)",
                        len(lam_candidates),
                        np.rad2deg(lat_r[1] - lat_r[0]),
                        np.rad2deg(lon_r[1] - lon_r[0]))
        else:
            lam_candidates = np.empty((0, 2), dtype=np.float32)
            lam_grads = np.empty(0, dtype=np.float32)
            LOGGER.warning("No LAM candidates found — check reference_node_name / mask_attr_name.")

        # ── 3. Order candidates and run Poisson-disk for each region ─────
        # LAM: sort by gradient descending so steep terrain is filled first.
        # Global: sort by gradient descending too, but with near-uniform
        #   compression (large value) the gradient signal is mild and we get
        #   coverage-driven placement.
        final_nodes: list[np.ndarray] = []

        for cand_pool, grad_pool, target_count, comp, label in [
            (lam_candidates, lam_grads, self.num_nodes_lam, self.compression_lam, "LAM"),
            (global_candidates, global_grads, self.num_nodes_global, self.compression_global, "global"),
        ]:
            if len(cand_pool) == 0 or target_count == 0:
                continue

            # Sort by effective priority: (grad+ε)^(1/comp) descending.
            # For LAM (comp≈0.1): p = grad^10 → strongly terrain-sorted.
            # For global (comp≈10): p = grad^0.1 → near-uniform ordering.
            priority = np.power(
                grad_pool + 1e-6,
                1.0 / max(comp, 1e-9),
            )

            priority = np.clip(
                priority,
                None,
                np.percentile(priority, 99),
            )

            priority = np.maximum(priority, 1e-12)

            # Weighted random permutation
            gumbel = -np.log(-np.log(np.random.rand(len(priority))))

            sample_order = np.argsort(
                -(np.log(priority) + gumbel)
            )

            sampled = cand_pool[sample_order]

            # Reference includes all nodes accepted so far (LAM placed first)
            ref = np.array(final_nodes) if final_nodes else np.array([[0.0, 0.0]])

            accepted = _poisson_disk_sample(sampled, ref, target_count, self.min_dist_rad)
            LOGGER.info(
                "ProbabilisticOrographyNodes %s: %d/%d nodes placed",
                label, len(accepted), target_count,
            )
            final_nodes.extend(accepted.tolist())

        all_nodes = np.array(final_nodes, dtype=np.float32)
        LOGGER.info("ProbabilisticOrographyNodes: %d total nodes", len(all_nodes))
        return torch.tensor(all_nodes, dtype=torch.float32)
