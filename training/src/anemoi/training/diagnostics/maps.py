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
