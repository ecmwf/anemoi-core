import copy
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from anemoi.training import diagnostics

LOGGER = logging.getLogger(__name__)


class EquirectangularProjection:
    """Class to convert lat/lon coordinates to Equirectangular coordinates."""

    def __init__(self) -> None:
        """Initialize the projection with default zero offsets."""
        self.x_offset = 0.0
        self.y_offset = 0.0

    def __call__(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project longitude and latitude coordinates to equirectangular x, y.

        Args:
            lon (np.ndarray): Array of longitude values in degrees.
            lat (np.ndarray): Array of latitude values in degrees.

        Returns
        -------
            tuple[np.ndarray, np.ndarray]: Radians-based (x, y) coordinates,
                with x shifted to the range [-pi, pi].
        """
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        x = np.array([v - 2 * np.pi if v > np.pi else v for v in lon_rad], dtype=lon_rad.dtype)
        y = lat_rad
        return x, y

    @staticmethod
    def inverse(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert equirectangular x, y coordinates back to degrees.

        Args:
            x (np.ndarray): Radians-based x coordinate.
            y (np.ndarray): Radians-based y coordinate.

        Returns
        -------
            tuple[np.ndarray, np.ndarray]: (longitude, latitude) in degrees.
        """
        return np.degrees(x), np.degrees(y)


class Coastlines:
    """Class to plot coastlines from a GeoJSON file."""

    def __init__(self, projection: EquirectangularProjection | None = None) -> None:
        """Initialize coastline data and projection.

        Args:
            projection (EquirectangularProjection | None): Projection to use.
                Defaults to a new EquirectangularProjection instance.

        Raises
        ------
            ModuleNotFoundError: If importlib_resources is missing on Python <= 3.8.
        """
        try:
            # this requires python 3.9 or newer
            from importlib.resources import files
        except ImportError:
            try:
                from importlib_resources import files
            except ModuleNotFoundError as e:
                msg = "Please install importlib_resources on Python <=3.8."
                raise ModuleNotFoundError(msg) from e

        self.continents_file = files(diagnostics) / "continents.json"
        with self.continents_file.open("rt") as file:
            self.data = json.load(file)

        self.projection = projection or EquirectangularProjection()
        self.process_data()

    def process_data(self) -> None:
        """Transform GeoJSON geometry into a Matplotlib LineCollection."""
        lines = []
        for feature in self.data["features"]:
            coords = feature["geometry"]["coordinates"]
            x, y = zip(*coords, strict=False)
            lines.append(list(zip(*self.projection(x, y), strict=False)))
        self.lines = LineCollection(lines, linewidth=0.5, color="black")

    def plot_continents(self, ax: plt.Axes) -> None:
        """Add the coastline LineCollection to the provided axis.

        Args:
            ax (plt.Axes): The Matplotlib axes to plot onto.
        """
        ax.add_collection(copy.copy(self.lines))


class Borders:
    """Class to plot country borders from a local GeoJSON file, following Coastlines logic."""

    def __init__(self, projection: EquirectangularProjection | None = None) -> None:
        """Initialize border data and projection.

        Args:
            projection (EquirectangularProjection | None): Projection to use.
                Defaults to a new EquirectangularProjection instance.
        """
        try:
            # this requires python 3.9 or newer
            from importlib.resources import files
        except ImportError:
            try:
                from importlib_resources import files
            except ModuleNotFoundError as e:
                msg = "Please install importlib_resources on Python <=3.8."
                raise ModuleNotFoundError(msg) from e

        # Assuming the file is placed in the same directory as 'continents.json'
        self.borders_file = files(diagnostics) / "countries.geo.json"
        with self.borders_file.open("rt") as file:
            self.data = json.load(file)

        self.projection = projection or EquirectangularProjection()
        self.process_data()

    def process_data(self) -> None:
        """Transform GeoJSON geometry into a Matplotlib LineCollection."""
        lines = []
        for feature in self.data["features"]:
            geometry = feature["geometry"]
            geom_type = geometry["type"]
            coords = geometry["coordinates"]

            # Handle both Polygon and MultiPolygon structures
            if geom_type == "Polygon":
                for ring in coords:
                    lon, lat = zip(*ring, strict=False)
                    lines.append(list(zip(*self.projection(lon, lat), strict=False)))
            elif geom_type == "MultiPolygon":
                for polygon in coords:
                    for ring in polygon:
                        lon, lat = zip(*ring, strict=False)
                        lines.append(list(zip(*self.projection(lon, lat), strict=False)))

        # Using a dashed linestyle to distinguish borders from coastlines
        self.lines = LineCollection(lines, linewidth=0.5, color="black", linestyle=":")

    def plot_borders(self, ax: plt.Axes) -> None:
        """Add the border LineCollection to the provided axis.

        Args:
            ax (plt.Axes): The Matplotlib axes to plot onto.
        """
        ax.add_collection(copy.copy(self.lines))


class MapFeatures:
    """Container class for optional map features (coastlines, borders, etc.)."""

    def __init__(
        self,
        continents: Coastlines | None = None,
        borders: Borders | None = None,
    ) -> None:
        """Initialize the map features container.

        Args:
            continents (Coastlines | None): Coastline plotting object.
            borders (Borders | None): Border plotting object.
        """
        self.continents = continents
        self.borders = borders

    def plot(self, ax: plt.Axes) -> None:
        """Plot all enabled map features on the given axis.

        Args:
            ax (plt.Axes): The Matplotlib or GeoAxes to plot onto.
        """
        if self.continents:
            try:
                self.continents.plot_continents(ax)
            except (AttributeError, RuntimeError) as exc:
                LOGGER.warning("Failed to plot continents: %s", exc)
        if self.borders:
            try:
                self.borders.plot_borders(ax)
            except (AttributeError, RuntimeError) as exc:
                LOGGER.warning("Failed to plot borders: %s", exc)


def _build_map_features() -> MapFeatures:
    """Factory function to create a MapFeatures instance with available components.

    Returns
    -------
        MapFeatures: An object containing initialized Coastlines and (if available) Borders.
    """
    continents = Coastlines()
    borders = Borders()

    return MapFeatures(continents=continents, borders=borders)


# Construct once at import time
map_features = _build_map_features()
