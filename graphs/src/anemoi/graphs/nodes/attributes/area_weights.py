# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError
from scipy.spatial import SphericalVoronoi
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree
from torch_geometric.data.storage import NodeStorage

from anemoi.datasets import open_dataset
from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian_np
from anemoi.graphs.nodes.attributes.base_attributes import BaseNodeAttribute

LOGGER = logging.getLogger(__name__)

# A Voronoi cell is trusted when every vertex lies within this factor of the node's median
# neighbour distance. On any rectangular lattice the furthest vertex sits at
# sqrt(dx^2 + dy^2) / 2, below the median neighbour distance (dx + dy) / 2 at every aspect
# ratio, so interior cells always pass; cells deformed by the edge of the node set have
# vertices several node spacings out and fail by a wide margin.
TRUST_FACTOR = 1.2


class BaseAreaWeights(BaseNodeAttribute, ABC):
    """Base class for area weights of the nodes."""

    def get_latlon_coordinates(self, nodes: NodeStorage) -> torch.Tensor:
        return nodes.x.to(torch.float64)

    @abstractmethod
    def compute_area_weights(self, latlons: np.ndarray) -> np.ndarray: ...

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        """Compute the weights.

        Parameters
        ----------
        nodes : NodeStorage
            Nodes of the graph.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Area weights for the nodes.
        """
        latlons = self.get_latlon_coordinates(nodes).cpu().numpy()
        area_weights = self.compute_area_weights(latlons)
        return torch.from_numpy(area_weights)


class UniformWeights(BaseAreaWeights):
    """Implements a uniform weight for the nodes.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def compute_area_weights(self, latlons: np.ndarray) -> np.ndarray:
        """Compute area weights.

        Parameters
        ----------
        latlons : np.ndarray
            2D array of shape (N, 2) with latitude and longitude coordinates of the
            nodes of the graph.

        Returns
        -------
        np.ndarray
            Ones.
        """
        return np.ones(latlons.shape[0])


class PlanarAreaWeights(BaseAreaWeights):
    """Planar area weights

    It computes the area in a 2D plane asociated to each node.

    Attributes
    ----------
    norm : str
        Normalisation of the weights.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    @staticmethod
    def _median_neighbour_distance(v: Voronoi, points: np.ndarray) -> np.ndarray:
        """Median distance from each point to its natural (Delaunay) neighbours."""
        ridge = v.ridge_points
        ridge_length = np.linalg.norm(points[ridge[:, 0]] - points[ridge[:, 1]], axis=1)
        owner = np.concatenate([ridge[:, 0], ridge[:, 1]])
        distance = np.tile(ridge_length, 2)
        order = np.lexsort((distance, owner))
        owner, distance = owner[order], distance[order]

        counts = np.bincount(owner, minlength=len(points))
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        median = np.full(len(points), np.inf)  # a point without neighbours trusts nothing
        occupied = counts > 0
        lower = distance[starts[occupied] + (counts[occupied] - 1) // 2]
        upper = distance[starts[occupied] + counts[occupied] // 2]
        median[occupied] = (lower + upper) / 2
        return median

    def _trusted_cells(self, v: Voronoi, points: np.ndarray) -> np.ndarray:
        """Which Voronoi cells the tessellation got right.

        A cell is trusted when its region is bounded and every vertex lies within
        ``TRUST_FACTOR`` times the node's median neighbour distance. Cells on the edge of
        the node set are unbounded, or stretch out to vertices several node spacings away,
        so they fail; interior cells of any quasi-uniform node set pass (see the note on
        ``TRUST_FACTOR``).
        """
        n_points = len(points)
        median = self._median_neighbour_distance(v, points)
        regions = [v.regions[idx] if idx >= 0 else [] for idx in v.point_region[:n_points]]
        lengths = np.array([len(region) for region in regions], dtype=np.intp)
        flat = np.fromiter(
            (vertex_idx for region in regions for vertex_idx in region),
            dtype=np.intp,
            count=int(lengths.sum()),
        )
        owner = np.repeat(np.arange(n_points), lengths)

        # -1 in a region marks a vertex at infinity; np.maximum only keeps the index valid,
        # such regions are rejected by the first term.
        reach = np.linalg.norm(v.vertices[np.maximum(flat, 0)] - points[owner], axis=1)
        far = (flat < 0) | (reach > TRUST_FACTOR * median[owner])
        return (lengths >= 3) & (np.bincount(owner[far], minlength=n_points) == 0)

    @staticmethod
    def _voronoi_region_areas(v: Voronoi, point_indices: np.ndarray) -> np.ndarray:
        """Polygon area of the Voronoi regions of ``point_indices``, vectorised.

        Uses the shoelace formula instead of a per-node :class:`scipy.spatial.ConvexHull`.
        SciPy returns each 2-D region's vertices in boundary order, so for a convex cell
        this matches ``ConvexHull(...).volume`` to float-rounding level. A region whose
        stored vertex order is not convex would make shoelace under-count, so those are
        detected and recomputed exactly with ``ConvexHull``.

        Parameters
        ----------
        v : scipy.spatial.Voronoi
            Voronoi tessellation of the nodes.
        point_indices : np.ndarray
            Indices of the points whose cell area is returned.

        Returns
        -------
        np.ndarray
            Area of the Voronoi cell of each of ``point_indices``.
        """
        n_points = len(point_indices)
        regions = [v.regions[v.point_region[idx]] for idx in point_indices]
        lengths = np.array([len(region) for region in regions], dtype=np.intp)

        # Regions must be bounded polygons (>= 3 vertices); trusted cells all are.
        assert (
            n_points == 0 or lengths.min() >= 3
        ), "Voronoi regions must be bounded polygons with >= 3 vertices (were untrusted cells passed?)."

        # Flatten the ragged regions; per-region offsets and a wrap-around "next vertex" index.
        flat = np.fromiter(
            (vertex_idx for region in regions for vertex_idx in region),
            dtype=np.intp,
            count=int(lengths.sum()),
        )
        starts = np.zeros(n_points, dtype=np.intp)
        np.cumsum(lengths[:-1], out=starts[1:])
        ends = starts + lengths
        nxt = np.arange(1, flat.shape[0] + 1, dtype=np.intp)
        nxt[ends - 1] = starts  # wrap each region's last vertex back to its first

        # Shoelace formula: area = 1/2 |sum_i (x_i * y_{i+1} - x_{i+1} * y_i)|, per region.
        pts = v.vertices[flat]
        x, y = pts[:, 0], pts[:, 1]
        cross = x * y[nxt] - x[nxt] * y
        areas = 0.5 * np.abs(np.add.reduceat(cross, starts))

        # Flag non-convex regions: a convex polygon's consecutive edge-turn cross products
        # all share one sign, so a region with both signs has an interior/degenerate vertex.
        edge_x, edge_y = x[nxt] - x, y[nxt] - y
        turn = edge_x * edge_y[nxt] - edge_y * edge_x[nxt]
        has_pos = np.add.reduceat((turn > 0).astype(np.intp), starts) > 0
        has_neg = np.add.reduceat((turn < 0).astype(np.intp), starts) > 0
        flagged = np.flatnonzero(has_pos & has_neg)

        # Exact ConvexHull fallback for the rare degenerate regions (matches the old code).
        for pos in flagged:
            region = v.regions[v.point_region[point_indices[pos]]]
            areas[pos] = ConvexHull(v.vertices[region]).volume

        return areas

    def compute_area_weights(self, latlons: np.ndarray) -> np.ndarray:
        """Compute area weights.

        The tessellation only defines the cells of interior nodes; a cell on the edge of
        the node set is unbounded or stretches far beyond the nodes. Rather than closing
        those cells geometrically, every untrusted cell takes the area of the nearest
        trusted one, which on a quasi-uniform grid is the exact answer. A node set with no
        trusted cell at all (fewer than 3 nodes, collinear, or too irregular) gets uniform
        weights.

        Parameters
        ----------
        latlons : np.ndarray
            2D array of shape (N, 2) with latitude and longitude coordinates of the
            nodes of the graph.

        Returns
        -------
        np.ndarray
            Planar area weights.
        """
        try:
            # Merged facets, not the joggle: joggle scales with the bounding box rather
            # than the node spacing, so it fails on stretched grids (#690).
            v = Voronoi(latlons, qhull_options="Qbb Qc Qz Pp")
        except (QhullError, ValueError) as error:
            LOGGER.warning(
                "%s: the nodes cannot be tessellated (%s). Using uniform weights.",
                type(self).__name__,
                str(error).strip().splitlines()[0],
            )
            return np.ones(len(latlons))

        trusted = self._trusted_cells(v, latlons)
        if not trusted.any():
            LOGGER.warning(
                "%s: none of the %d Voronoi cells is bounded and local. Using uniform weights.",
                type(self).__name__,
                len(latlons),
            )
            return np.ones(len(latlons))

        areas = np.empty(len(latlons))
        trusted_idx = np.flatnonzero(trusted)
        untrusted_idx = np.flatnonzero(~trusted)
        areas[trusted_idx] = self._voronoi_region_areas(v, trusted_idx)
        if len(untrusted_idx):
            _, nearest = cKDTree(latlons[trusted_idx]).query(latlons[untrusted_idx])
            areas[untrusted_idx] = areas[trusted_idx[nearest]]
        return areas


class MaskedPlanarAreaWeights(PlanarAreaWeights):
    """Masked planar area weights

    It computes the area in a 2D plane asociated to each node.

    Attributes
    ----------
    mask_node_attr_name : str
        Name of a node attribute to use as a mask for the computing the area weights.
        It sets to 0 values outside this masked region.
    norm : str, optional
        Normalisation of the weights.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def __init__(
        self,
        mask_node_attr_name: str,
        norm: str | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__(norm, dtype)
        assert isinstance(
            mask_node_attr_name, str
        ), f"{self.__class__.__name__} requires a string for 'mask_node_attr_name' variable."
        self.mask_node_attr_name = mask_node_attr_name

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        """Compute the weights over the masked nodes alone, and zero elsewhere.

        The masked nodes are tessellated on their own. Tessellating every node and
        applying the mask afterwards instead leaves the cells on the edge of the mask
        bounded by whatever lies outside it, which inflates them.
        """
        assert self.mask_node_attr_name in nodes, f"Node attribute '{self.mask_node_attr_name}' not found in nodes."
        mask = nodes[self.mask_node_attr_name].squeeze()
        selected = mask.cpu().numpy() != 0
        latlons = self.get_latlon_coordinates(nodes).cpu().numpy()

        area_weights = np.zeros(len(latlons))
        if (n_selected := selected.sum()) < 3:
            LOGGER.warning(
                "%s: '%s' selects %d nodes, fewer than 3. Using uniform weights over them.",
                type(self).__name__,
                self.mask_node_attr_name,
                n_selected,
            )
            area_weights[selected] = 1.0
        else:
            area_weights[selected] = self.compute_area_weights(latlons[selected])

        # multiplying rather than indexing keeps a fractional mask scaling the weights
        return torch.from_numpy(area_weights).to(mask.device) * mask


class SphericalAreaWeights(BaseAreaWeights):
    """Spherical area weights

    It computes the area of a unit radius sphere asociated to each node.

    Attributes
    ----------
    norm : str
        Normalisation of the weights.
    radius : float
        Radius of the sphere.
    centre : np.ndarray
        Centre of the sphere.
    fill_value : float
        Value to fill the empty regions.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def __init__(
        self,
        norm: str | None = None,
        radius: float = 1.0,
        centre: np.ndarray = np.array([0, 0, 0]),
        fill_value: float = 0.0,
        dtype: str = "float32",
    ) -> None:
        assert isinstance(fill_value, float) or isinstance(
            fill_value, int
        ), f"fill_value must be float or nan but it is {type(fill_value)}"
        assert (
            isinstance(radius, float) or isinstance(radius, int)
        ) and radius > 0, f"radius must be a positive value, but radius={radius}"
        super().__init__(norm, dtype)
        self.radius = radius
        self.centre = centre
        self.fill_value = fill_value

    def compute_area_weights(self, latlons: np.ndarray) -> np.ndarray:
        """Compute the area associated to each node.

        It uses Voronoi diagrams to compute the area of each node.

        Parameters
        ----------
        latlons : np.ndarray
            2D array of shape (N, 2) with latitude and longitude coordinates of the
            nodes of the graph.

        Returns
        -------
        np.ndarray
            Spherical area weights.
        """
        points = latlon_rad_to_cartesian_np(latlons)
        sv = SphericalVoronoi(points, self.radius, self.centre)
        mask = np.array([bool(i) for i in sv.regions])
        sv.regions = [region for region in sv.regions if region]
        # compute the area weight without empty regions
        area_weights = sv.calculate_areas()
        if (null_nodes := (~mask).sum()) > 0:
            LOGGER.warning(
                "%s is filling %d (%.2f%%) nodes with value %f",
                self.__class__.__name__,
                null_nodes,
                100 * null_nodes / len(mask),
                self.fill_value,
            )
        result = np.ones(points.shape[0]) * self.fill_value
        result[mask] = area_weights
        LOGGER.debug(
            "There are %d of weights, which (unscaled) add up a total weight of %.2f.",
            len(result),
            result.sum(),
        )
        return result


class BaseLatWeightedAttribute(BaseAreaWeights, ABC):
    """Base class for latitude-weigthed area weights."""

    @abstractmethod
    def compute_latitude_weight(self, latitudes: np.ndarray) -> np.ndarray: ...

    def compute_area_weights(self, latlons: np.ndarray) -> np.ndarray:
        return self.compute_latitude_weight(latlons[:, 0])


class CosineLatWeightedAttribute(BaseLatWeightedAttribute):
    """Latitude-weighting of the node attributes for rectilinear grids.

    Attributes
    ----------
    min_value : float
        Minimum value of the weights when the latitude is -pi/2 or pi/2 radians.
    max_value : float
        Maximum value of the weights when the latitude is 0 radians.
    norm : str
        Normalisation of the weights.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def __init__(
        self,
        min_value: float = 1e-3,
        max_value: float = 1,
        norm: str | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__(norm, dtype)
        self.min_value = min_value
        self.max_value = max_value

    def compute_latitude_weight(self, latitudes: np.ndarray) -> np.ndarray:
        return (self.max_value - self.min_value) * np.cos(latitudes) + self.min_value


class IsolatitudeAreaWeights(BaseLatWeightedAttribute):
    r"""Latitude-weighted area weights for rectilinear grids.

    Attributes
    ----------
    norm : str
        Normalisation of the weights.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.

    Notes
    ------
    The area of a latitude band is

    .. math::
        A = 2\pi R(\sin(lat_2) - \sin(lat_1))

    where R is the earth radius and lat_1, lat_2 are in radians.
    """

    def compute_latitude_weight(self, latitudes: np.ndarray) -> np.ndarray:
        # Get the latitudes defining the bands
        unique_lats = np.sort(np.unique(latitudes))
        divisory_lats = (unique_lats[1:] + unique_lats[:-1]) / 2
        divisory_lats = np.concatenate([[-np.pi / 2], divisory_lats, [np.pi / 2]])

        # Compute the latitude band area
        lat_1 = divisory_lats[1:]
        lat_2 = divisory_lats[:-1]
        assert np.all(lat_1 >= lat_2), "Nodes should be sorted by latitude."
        ring_area_km = 2 * np.pi * EARTH_RADIUS * (np.sin(lat_1) - np.sin(lat_2))

        # Compute the number of points/nodes at each latitude band
        lat_to_ring = {lat: idx for idx, lat in enumerate(unique_lats)}
        lat_rings = np.array([lat_to_ring[lat] for lat in latitudes])
        lat_counts = np.bincount(lat_rings, minlength=len(unique_lats))

        # Compute the area of each node
        area_km = dict(zip(unique_lats, ring_area_km / lat_counts))
        return np.array([area_km[lat] for lat in latitudes])


class AnemoiDatasetVariableWeights(BaseNodeAttribute):
    """Load area weights from a variable in the dataset.

    Attributes
    ----------
    variable : str
        Name of the variable to use as weights.
    norm : str, optional
        Method to use to normalise the weights.

    Methods
    -------
    get_raw_values(self, nodes)
        Extract the data and convert it to a Torch tensor object.
    """

    def __init__(self, variable: str, norm: str | None = None, dtype: str = "float32") -> None:
        super().__init__(norm, dtype)
        self.variable = variable

    def _read_data(self, nodes: NodeStorage, **kwargs) -> np.ndarray:
        """Read the weighting variable from the dataset."""
        return open_dataset(nodes["_dataset"], select=self.variable)[0].squeeze()

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        """Extract the data and convert it to a Torch tensor object.

        Parameters
        ----------
        nodes : NodeStorage
            Nodes of the graph.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Area weights for the nodes.
        """
        assert nodes["node_type"] in [
            "ZarrDatasetNodes",
            "AnemoiDatasetNodes",
        ], f"{self.__class__.__name__} can only be used with AnemoiDatasetNodes."
        data = torch.from_numpy(self._read_data(nodes))
        return self.post_process(data)
