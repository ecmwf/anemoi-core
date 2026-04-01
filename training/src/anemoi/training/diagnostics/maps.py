# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
import json
import logging

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from anemoi.training import diagnostics
from anemoi.training.diagnostics.projections import Projection

LOGGER = logging.getLogger(__name__)


class Coastlines:
    """Class to plot coastlines from a GeoJSON file."""

    def __init__(self, projection: Projection | None = None) -> None:
        """Initialise the Coastlines object.

        Parameters
        ----------
        projection : Projection | None, optional
            Projection (e.g. Projection.equirectangular() or Projection.lambert_conformal(latlon)).
            By default None (uses equirectangular).

        Raises
        ------
        ModuleNotFoundError
            Whether the importlib_resources or importlib.resources module is not found.

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

        self.projection = projection or Projection.equirectangular()
        self.process_data()

    def process_data(self) -> None:
        lines_proj = []
        lines_deg = []
        for feature in self.data["features"]:
            coords = feature["geometry"]["coordinates"]
            x, y = zip(*coords, strict=False)
            lines_proj.append(list(zip(*self.projection(x, y), strict=False)))
            lines_deg.append(list(zip(x, y, strict=False)))
        self.lines = LineCollection(lines_proj, linewidth=0.5, color="black")
        self.lines_degrees = LineCollection(lines_deg, linewidth=0.5, color="black")

    def plot_continents(self, ax: plt.Axes, crs: object | None = None) -> None:
        # Add the lines to the axis as a collection
        # Note that we have to provide a copy of the lines, because of Matplotlib
        if crs is not None:
            import cartopy.crs as ccrs

            lc = copy.copy(self.lines_degrees)
            lc.set_transform(ccrs.PlateCarree())
            ax.add_collection(lc)
        else:
            ax.add_collection(copy.copy(self.lines))


class Borders:
    """Class to plot country borders from a local GeoJSON file."""

    def __init__(self, projection: Projection | None = None) -> None:
        """Initialize border data and projection.

        Parameters
        ----------
        projection : Projection | None
            Projection (e.g. Projection.equirectangular() or Projection.lambert_conformal(latlon)).
            Defaults to equirectangular.
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

        self.projection = projection or Projection.equirectangular()
        self.process_data()

    def process_data(self) -> None:
        lines_proj = []
        lines_deg = []
        for feature in self.data["features"]:
            geometry = feature["geometry"]
            geom_type = geometry["type"]
            coords = geometry["coordinates"]

            # Handle both Polygon and MultiPolygon structures
            if geom_type == "Polygon":
                for ring in coords:
                    lon, lat = zip(*ring, strict=False)
                    lines_proj.append(list(zip(*self.projection(lon, lat), strict=False)))
                    lines_deg.append(list(zip(lon, lat, strict=False)))
            elif geom_type == "MultiPolygon":
                for polygon in coords:
                    for ring in polygon:
                        lon, lat = zip(*ring, strict=False)
                        lines_proj.append(list(zip(*self.projection(lon, lat), strict=False)))
                        lines_deg.append(list(zip(lon, lat, strict=False)))

        # Using a dashed linestyle to distinguish borders from coastlines
        self.lines = LineCollection(lines_proj, linewidth=0.5, color="black", linestyle=":")
        self.lines_degrees = LineCollection(lines_deg, linewidth=0.5, color="black", linestyle=":")

    def plot_borders(self, ax: plt.Axes, crs: object | None = None) -> None:
        # Add the lines to the axis as a collection
        # Note that we have to provide a copy of the lines, because of Matplotlib
        if crs is not None:
            import cartopy.crs as ccrs

            lc = copy.copy(self.lines_degrees)
            lc.set_transform(ccrs.PlateCarree())
            ax.add_collection(lc)
        else:
            ax.add_collection(copy.copy(self.lines))


class MapFeatures:
    """Container class for optional map features (coastlines, borders, etc.)."""

    def __init__(
        self,
        continents: Coastlines | None = None,
        borders: Borders | None = None,
    ) -> None:
        """Initialize the map features container.

        Parameters
        ----------
        continents: Coastlines | None
            Coastline plotting object.
        borders: Borders | None
            Border plotting object.
        """
        self.continents = continents
        self.borders = borders

    def plot(self, ax: plt.Axes, crs: object | None = None) -> None:

        if self.continents:
            try:
                self.continents.plot_continents(ax, crs=crs)
            except (AttributeError, RuntimeError) as exc:
                LOGGER.warning("Failed to plot continents: %s", exc)
        if self.borders:
            try:
                self.borders.plot_borders(ax, crs=crs)
            except (AttributeError, RuntimeError) as exc:
                LOGGER.warning("Failed to plot borders: %s", exc)


def _build_map_features() -> MapFeatures:
    """Factory function to create a MapFeatures instance with available components."""
    continents = Coastlines()
    borders = Borders()

    return MapFeatures(continents=continents, borders=borders)


# Construct once at import time
map_features = _build_map_features()
