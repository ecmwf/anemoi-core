# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import math
from abc import ABC
from abc import abstractmethod

import torch
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate.masks import AreaMaskBuilder
from anemoi.graphs.generate.rotated_grid import geographic_to_rotated
from anemoi.graphs.generate.rotated_grid import rotated_to_geographic
from anemoi.graphs.generate.rotated_grid import rotation_matrix
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


def _parse_resolution(resolution: float | list[float]) -> tuple[float, float]:
    """Get the (latitude, longitude) spacing, in degrees, from one or two numbers."""
    dlat, dlon = (resolution, resolution) if isinstance(resolution, (int, float)) else tuple(resolution)
    if dlat <= 0 or dlon <= 0:
        raise ValueError(f"The resolution must be positive, got resolution={resolution}.")
    return float(dlat), float(dlon)


class BaseRectilinearNodes(BaseNodeBuilder, ABC):
    """Base class for nodes on a rectilinear grid.

    The nodes are stored row by row. Each node keeps its position in the grid in the hidden
    attribute ``_layout_index``, a tensor of shape (num_nodes, 2) with its row and column, and
    the grid size in ``_layout_shape``, a list [num_rows, num_cols].
    """

    hidden_attributes: set[str] = {"layout_index", "layout_shape"}

    @abstractmethod
    def get_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the latitudes and longitudes of the full grid.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Latitudes and longitudes, in degrees, each of shape (num_rows, num_cols).
        """

    def get_grid_mask(self, coords: torch.Tensor) -> torch.Tensor:
        """Get the grid points to keep as nodes. All of them by default.

        Parameters
        ----------
        coords : torch.Tensor of shape (num_rows * num_cols, 2)
            Coordinates of the grid points, row by row, in radians.

        Returns
        -------
        torch.Tensor of shape (num_rows * num_cols,)
            True for the grid points to keep.
        """
        return torch.ones(coords.shape[0], dtype=torch.bool)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        latitudes, longitudes = self.get_grid()
        self.layout_shape = list(latitudes.shape)
        coords = self.reshape_coords(latitudes.reshape(-1), longitudes.reshape(-1))

        mask = self.get_grid_mask(coords)
        flat_index = torch.nonzero(mask).squeeze(-1)
        self.layout_index = torch.stack(torch.unravel_index(flat_index, self.layout_shape), dim=-1).to(self.device)

        LOGGER.info(
            'Creating %d "%s" nodes on a %d x %d rectilinear grid.', len(flat_index), self.name, *self.layout_shape
        )
        return coords[mask]


class RectilinearNodes(BaseRectilinearNodes):
    """Nodes on a global regular latitude-longitude grid.

    Nodes are stored row by row, from north to south, with longitudes running east
    from zero within each row.

    Attributes
    ----------
    resolution : float | list[float]
        Grid spacing in degrees, the same for latitude and longitude or [dlat, dlon].
        180 / dlat and 360 / dlon must be whole numbers.
    include_poles : bool
        Whether the first and last rows sit on the poles. Defaults to False.

    Methods
    -------
    get_coordinates()
        Get the lat-lon coordinates of the nodes.
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attrs_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(self, resolution: float | list[float], name: str, include_poles: bool = False) -> None:
        """Initialize the RectilinearNodes builder."""
        self.dlat, self.dlon = _parse_resolution(resolution)
        for span, step in ((180, self.dlat), (360, self.dlon)):
            if abs(span / step - round(span / step)) > 1e-9:
                raise ValueError(f"{span} / resolution must be a whole number, got resolution={resolution}.")
        self.include_poles = include_poles
        self.nlat = round(180 / self.dlat) + int(include_poles)
        self.nlon = round(360 / self.dlon)
        super().__init__(name)

    def get_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        offset = 0.0 if self.include_poles else 0.5
        latitudes = 90 - (torch.arange(self.nlat, dtype=torch.float64) + offset) * self.dlat
        longitudes = torch.arange(self.nlon, dtype=torch.float64) * self.dlon
        return torch.meshgrid(latitudes, longitudes, indexing="ij")


class LimitedAreaRectilinearNodes(BaseRectilinearNodes):
    """Node builder for a regular grid around an area of interest.

    The rotated frame moves the centre of the reference nodes to (0, 0), and turns the grid axes
    anticlockwise about it by ``rotation`` degrees. Nodes are stored row by row,
    from rotated north to south, with rotated longitudes running east within each row.

    With ``crop_to_margin``, the grid points farther than ``margin_radius_km`` from every
    reference node are left out. The remaining nodes keep their row and column in the full grid.

    Arguments
    ----------
    resolution : float | list[float]
        Grid spacing in degrees in the rotated frame, the same for latitude and longitude or
        [dlat, dlon].
    reference_node_name : str
        The name of the reference nodes used to define the area of interest.
    mask_attr_name : str | None
        The name of the attribute storing the area mask. If None, no mask is used.
    crop_to_margin : bool
        Whether to leave out the grid points beyond the margin. Defaults to False.
    margin_radius_km : float
        The radius of the margin around the reference nodes, in kilometers.
    rotation : float
        Anticlockwise angle of the grid axes about the centre, in degrees. With 0, the rows run
        along true east at the centre. Defaults to 0.
    """

    def __init__(
        self,
        resolution: float | list[float],
        reference_node_name: str,
        name: str,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
        crop_to_margin: bool = False,
        rotation: float = 0.0,
    ) -> None:
        """Initialize the LimitedAreaRectilinearNodes builder."""
        self.dlat, self.dlon = _parse_resolution(resolution)
        self.crop_to_margin = crop_to_margin
        self.rotation = float(rotation)
        super().__init__(name)
        self.area_mask_builder = AreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

    def register_nodes(self, graph: HeteroData) -> HeteroData:
        self.area_mask_builder.fit(graph)
        self.reference_coords = self.area_mask_builder.get_reference_coords(graph).cpu().to(torch.float64)
        return super().register_nodes(graph)

    def get_centre(self) -> tuple[float, float]:
        """Get the centre of the rotated frame, the centre of the reference nodes, in degrees."""
        mean_vector = latlon_rad_to_cartesian(self.reference_coords).mean(dim=0)
        norm = torch.linalg.norm(mean_vector)
        if norm < 1e-6:
            raise ValueError(
                f'The reference nodes "{self.area_mask_builder.reference_node_name}" have no centre. '
                "Use RectilinearNodes for a global grid."
            )
        x, y, z = (mean_vector / norm).tolist()
        return math.degrees(math.asin(z)), math.degrees(math.atan2(y, x))

    def get_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        centre_lat, centre_lon = self.get_centre()
        rotation = rotation_matrix(centre_lat, centre_lon, self.rotation)
        rotated = torch.rad2deg(geographic_to_rotated(self.reference_coords, rotation))

        margin = math.degrees(self.area_mask_builder.margin_radius_km / EARTH_RADIUS)
        lat_min, lat_max = rotated[:, 0].min().item() - margin, rotated[:, 0].max().item() + margin
        if lat_max >= 90 or lat_min <= -90:
            raise ValueError(
                f'The area around "{self.area_mask_builder.reference_node_name}" reaches a pole of the rotated '
                "frame. Use RectilinearNodes for such a large area."
            )
        lon_margin = margin / math.cos(math.radians(max(abs(lat_min), abs(lat_max))))
        lon_min, lon_max = rotated[:, 1].min().item() - lon_margin, rotated[:, 1].max().item() + lon_margin
        if lon_max >= 180 or lon_min <= -180:
            raise ValueError(
                f'The area around "{self.area_mask_builder.reference_node_name}" goes all the way around the '
                "rotated frame. Use RectilinearNodes for such a large area."
            )

        rows = torch.arange(math.ceil(lat_max / self.dlat - 1e-9), math.floor(lat_min / self.dlat + 1e-9) - 1, -1)
        cols = torch.arange(math.floor(lon_min / self.dlon + 1e-9), math.ceil(lon_max / self.dlon - 1e-9) + 1)
        rotated_lat, rotated_lon = torch.meshgrid(
            rows.to(torch.float64) * self.dlat, cols.to(torch.float64) * self.dlon, indexing="ij"
        )

        LOGGER.info(
            'Rotated grid for "%s" centred at (%.2f, %.2f) deg and turned by %.2f deg, covering rotated '
            "latitudes [%.2f, %.2f] and longitudes [%.2f, %.2f] deg.",
            self.name,
            centre_lat,
            centre_lon,
            self.rotation,
            rotated_lat.min(),
            rotated_lat.max(),
            rotated_lon.min(),
            rotated_lon.max(),
        )
        rotated_coords = torch.deg2rad(torch.stack([rotated_lat, rotated_lon], dim=-1))
        geographic = torch.rad2deg(rotated_to_geographic(rotated_coords, rotation))
        return geographic[..., 0], geographic[..., 1]

    def get_grid_mask(self, coords: torch.Tensor) -> torch.Tensor:
        if not self.crop_to_margin:
            return super().get_grid_mask(coords)

        # The reference nodes are in float32, as stored in the graph.
        mask = self.area_mask_builder.get_mask(coords.to(torch.float32))
        LOGGER.info('Masking out %d nodes from "%s".', len(mask) - mask.sum(), self.name)
        return mask
