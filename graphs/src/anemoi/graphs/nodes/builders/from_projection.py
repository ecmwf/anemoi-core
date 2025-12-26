"""Build nodes from projected coordinates (e.g., Lambert Conformal, Stereographic)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from pyproj import CRS, Transformer

from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class ProjectedGridNodes(BaseNodeBuilder):
    """Build nodes on a regular grid in a projected coordinate system.
    
    This builder creates nodes at specified intervals in a projected coordinate
    system (e.g., Lambert Conformal Conic, Polar Stereographic), then transforms
    them back to geographic coordinates (lat/lon) for use in the graph.
    
    Parameters
    ----------
    projection : str or dict
        Projection specification. Can be:
        - PROJ string (e.g., "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96")
        - EPSG code (e.g., "EPSG:3857")
        - Dictionary with projection parameters
    x_min : float
        Minimum x coordinate in projected system (meters or projection units)
    x_max : float
        Maximum x coordinate in projected system
    y_min : float
        Minimum y coordinate in projected system
    y_max : float
        Maximum y coordinate in projected system
    resolution : float
        Grid spacing in projection units (e.g., meters, km)
    resolution_unit : str, optional
        Unit of resolution ('m' or 'km'), default 'km'
    reference_lat : float, optional
        Reference latitude for the projection (if not in projection spec)
    reference_lon : float, optional
        Reference longitude for the projection (if not in projection spec)
    name : str, optional
        Name of the nodes, default "projected_grid"
        
    Examples
    --------
    >>> # Lambert Conformal over CONUS
    >>> builder = ProjectedGridNodes(
    ...     projection="+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +datum=WGS84",
    ...     x_min=-2400,
    ...     x_max=2400,
    ...     y_min=-1800,
    ...     y_max=1800,
    ...     resolution=25,
    ...     resolution_unit='km'
    ... )
    
    >>> # Polar Stereographic over Arctic
    >>> builder = ProjectedGridNodes(
    ...     projection="EPSG:3413",  # NSIDC Sea Ice Polar Stereographic North
    ...     x_min=-3000000,
    ...     x_max=3000000,
    ...     y_min=-3000000,
    ...     y_max=3000000,
    ...     resolution=25000,  # 25 km
    ...     resolution_unit='m'
    ... )
    """

    def __init__(
        self,
        projection: str | dict,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: float,
        resolution_unit: str = "km",
        reference_lat: Optional[float] = None,
        reference_lon: Optional[float] = None,
        name: str = "projected_grid",
    ):
        super().__init__(name)
        
        self.projection = projection
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution
        self.resolution_unit = resolution_unit
        self.reference_lat = reference_lat
        self.reference_lon = reference_lon
        
        # Convert resolution to meters if needed
        if resolution_unit.lower() == "km":
            self.resolution_m = resolution * 1000
        elif resolution_unit.lower() == "m":
            self.resolution_m = resolution
        else:
            raise ValueError(f"Unknown resolution unit: {resolution_unit}")
        
        # Setup projection
        self._setup_projection()
        
        # Build the nodes
        self._build_grid()

    def _setup_projection(self):
        """Setup the projection and transformer."""
        # Create CRS from projection specification
        if isinstance(self.projection, dict):
            # Build PROJ string from dict
            proj_params = [f"+{k}={v}" for k, v in self.projection.items()]
            proj_string = " ".join(proj_params)
            self.crs = CRS.from_proj4(proj_string)
        elif isinstance(self.projection, str):
            if self.projection.startswith("+proj"):
                self.crs = CRS.from_proj4(self.projection)
            elif self.projection.startswith("EPSG"):
                self.crs = CRS.from_string(self.projection)
            else:
                self.crs = CRS.from_string(self.projection)
        else:
            raise ValueError(f"Unknown projection type: {type(self.projection)}")
        
        # Create transformer from projected coordinates to lat/lon
        # CRS.from_epsg(4326) is WGS84 (lat/lon)
        self.transformer = Transformer.from_crs(
            self.crs,
            CRS.from_epsg(4326),
            always_xy=True  # Force x, y order (lon, lat)
        )
        
        LOGGER.info(f"Using projection: {self.crs.name}")

    def _build_grid(self):
        """Build the regular grid in projected coordinates."""
        # Create grid in projected coordinates
        x_coords = np.arange(self.x_min, self.x_max + self.resolution_m, self.resolution_m)
        y_coords = np.arange(self.y_min, self.y_max + self.resolution_m, self.resolution_m)
        
        LOGGER.info(
            f"Creating grid with {len(x_coords)} x {len(y_coords)} = "
            f"{len(x_coords) * len(y_coords)} nodes"
        )
        
        # Create meshgrid
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Flatten to 1D arrays
        x_proj = x_grid.flatten()
        y_proj = y_grid.flatten()
        
        # Transform to lat/lon
        lons, lats = self.transformer.transform(x_proj, y_proj)
        
        # Store projected coordinates as well (might be useful)
        self.x_projected = x_proj
        self.y_projected = y_proj
        
        # Filter out any invalid transformations (e.g., points outside valid projection area)
        valid_mask = np.isfinite(lons) & np.isfinite(lats)
        if not np.all(valid_mask):
            LOGGER.warning(
                f"Filtered {np.sum(~valid_mask)} invalid nodes outside projection domain"
            )
            lons = lons[valid_mask]
            lats = lats[valid_mask]
            self.x_projected = x_proj[valid_mask]
            self.y_projected = y_proj[valid_mask]
        
        self.lons = lons
        self.lats = lats
        
        LOGGER.info(f"Built {len(self.lons)} valid nodes")

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.
        
        Returns
        -------
        torch.Tensor
            Node coordinates in (lon, lat) format, shape (num_nodes, 2)
        """
        coords = np.stack([self.lons, self.lats], axis=1)
        return torch.tensor(coords, dtype=torch.float32)

    def register_nodes(self, graph):
        """Register nodes in the graph.
        
        Parameters
        ----------
        graph : HeteroData
            Graph to register the nodes.
        """
        graph[self.name].x = self.get_coordinates()
        graph[self.name].node_type = self.name
        
        # Store projected coordinates as additional features if needed
        proj_coords = np.stack([self.x_projected, self.y_projected], axis=1)
        graph[self.name].projected_coords = torch.tensor(proj_coords, dtype=torch.float32)
        
        return graph

    def register_attributes(self, graph, config):
        """Register additional attributes."""
        # Store projection information
        graph[self.name].projection_info = {
            "crs": self.crs.to_proj4(),
            "resolution": self.resolution,
            "resolution_unit": self.resolution_unit,
            "bounds": {
                "x_min": self.x_min,
                "x_max": self.x_max,
                "y_min": self.y_min,
                "y_max": self.y_max,
            }
        }
        return graph

    def update_graph(self, graph, attrs_config=None):
        """Update the graph with new nodes."""
        graph = self.register_nodes(graph)
        if attrs_config is not None:
            graph = self.register_attributes(graph, attrs_config)
        return graph