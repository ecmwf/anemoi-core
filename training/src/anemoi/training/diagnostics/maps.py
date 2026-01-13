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
import numpy as np
from matplotlib.collections import LineCollection

from anemoi.training import diagnostics

LOGGER = logging.getLogger(__name__)


class EquirectangularProjection:
    """Class to convert lat/lon coordinates to Equirectangular coordinates."""

    def __init__(self) -> None:
        """Initialise the EquirectangularProjection object with offset."""
        self.x_offset = 0.0
        self.y_offset = 0.0

    def __call__(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        x = np.array([v - 2 * np.pi if v > np.pi else v for v in lon_rad], dtype=lon_rad.dtype)
        y = lat_rad
        return x, y

    @staticmethod
    def inverse(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon = np.degrees(x)
        lat = np.degrees(y)
        return lon, lat


class Coastlines:
    """Class to plot coastlines from a GeoJSON file."""

    def __init__(self, projection: EquirectangularProjection = None) -> None:
        """Initialise the Coastlines object.

        Parameters
        ----------
        projection : Any, optional
            Projection Object, by default None

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

        # Get the path to "continents.json" within your library
        self.continents_file = files(diagnostics) / "continents.json"

        # Load GeoJSON data from the file
        with self.continents_file.open("rt") as file:
            self.data = json.load(file)

        if projection is None:
            self.projection = EquirectangularProjection()

        self.process_data()

    # Function to extract LineString coordinates
    @staticmethod
    def extract_coordinates(feature: dict) -> list:
        return feature["geometry"]["coordinates"]

    def process_data(self) -> None:
        lines = []
        for feature in self.data["features"]:
            coordinates = self.extract_coordinates(feature)
            x, y = zip(*coordinates, strict=False)  # Unzip the coordinates into separate x and y lists
            lines.append(list(zip(*self.projection(x, y), strict=False)))  # Convert lat/lon to Cartesian coordinates
        self.lines = LineCollection(lines, linewidth=0.5, color="black")

    def plot_continents(self, ax: plt.Axes) -> None:
        # Add the lines to the axis as a collection
        # Note that we have to provide a copy of the lines, because of Matplotlib
        ax.add_collection(copy.copy(self.lines))


def lambert_conformal_from_latlon_points(latlon: np.ndarray) -> object:
    """Build a Cartopy Lambert Conformal projection suited to a given set of (lat, lon) points.

    The projection is centered on the midpoint of the latitude/longitude
    extent of the input, and uses two standard parallels placed at ±25% of
    the latitude span around the central latitude. This gives a reasonable,
    low-distortion projection for regional maps covering mid-latitudes.

    Parameters
    ----------
    latlon : numpy.ndarray
        Array of shape (N, 2) with columns ``[latitude, longitude]`` in degrees.
        Longitudes may be in the range [-180, 180] or [0, 360]; values are used
        as-is to compute the central longitude.

    Returns
    -------
    object
        A ``cartopy.crs.LambertConformal`` instance configured with:
        - ``central_latitude`` at the midpoint of the latitude extent,
        - ``central_longitude`` at the midpoint of the longitude extent,
        - ``standard_parallels`` at ±25% of the latitude span around the center.

    Raises
    ------
    ModuleNotFoundError
        If ``cartopy`` is not installed. Install via the
        ``optional-dependencies.plotting`` extra.

    Notes
    -----
    - This heuristic works well for many regional plots. If your domain is very
      tall/narrow or crosses the dateline, you may want to choose the
      ``central_longitude`` or ``standard_parallels`` explicitly.
    - Input is not validated; ensure ``latlon`` has at least two points and a
      non-zero latitude span for meaningful standard parallels.
    """
    assert isinstance(latlon, (np.ndarray, list)), "Input must be a numpy array or list."
    latlon = np.asanyarray(latlon)

    # Shape must be (N, 2)
    assert latlon.ndim == 2, f"Input must be 2D, but got {latlon.ndim}D."
    assert latlon.shape[1] == 2, f"Input must have 2 columns [lat, lon], but got {latlon.shape[1]}."

    # Ensure latlon has at least two points
    assert latlon.shape[0] >= 2, "At least two points are required to calculate a span."

    # Latitude Range for physical reality
    assert np.all((latlon[:, 0] >= -90) & (latlon[:, 0] <= 90)), "Latitudes must be between -90 and 90."

    try:
        import cartopy.crs as ccrs
    except ModuleNotFoundError as e:
        error_msg = "Module cartopy not found. Install with optional-dependencies.plotting."
        raise ModuleNotFoundError(error_msg) from e

    lat_min, lon_min = latlon.min(axis=0)
    lat_max, lon_max = latlon.max(axis=0)

    # Ensure non-zero latitude span
    lat_span = lat_max - lat_min
    assert lat_span > 0, "Latitude span must be greater than zero to compute standard parallels."

    central_latitude = (lat_min + lat_max) / 2
    central_longitude = (lon_min + lon_max) / 2

    std_parallel_1 = central_latitude - lat_span * 0.25
    std_parallel_2 = central_latitude + lat_span * 0.25

    return ccrs.LambertConformal(
        central_latitude=central_latitude,
        central_longitude=central_longitude,
        standard_parallels=[std_parallel_1, std_parallel_2],
    )
