import copy
import json

import numpy as np
from matplotlib.collections import LineCollection

from anemoi.training import diagnostics


class EquirectangularProjection:
    """Class to convert lat/lon coordinates to Equirectangular coordinates."""

    def __init__(self) -> None:
        self.x_offset = 0.0
        self.y_offset = 0.0

    def __call__(self, lon, lat):
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        x = [v - 2 * np.pi if v > np.pi else v for v in lon_rad]
        y = lat_rad
        return x, y

    def inverse(self, x, y):
        lon = np.degrees(x)
        lat = np.degrees(y)
        return lon, lat


class Coastlines:
    """Class to plot coastlines from a GeoJSON file."""

    def __init__(self, projection=None) -> None:
        try:
            # this requires python 3.9 or newer
            from importlib.resources import files
        except ImportError:
            try:
                from importlib_resources import files
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError("Please install importlib_resources on Python <=3.8.") from e

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
    def extract_coordinates(feature):
        return feature["geometry"]["coordinates"]

    def process_data(self) -> None:
        lines = []
        for feature in self.data["features"]:
            coordinates = self.extract_coordinates(feature)
            x, y = zip(*coordinates)  # Unzip the coordinates into separate x and y lists

            lines.append(list(zip(*self.projection(x, y))))  # Convert lat/lon to Cartesian coordinates
        self.lines = LineCollection(lines, linewidth=0.5, color="black")

    def plot_continents(self, ax) -> None:
        # Add the lines to the axis as a collection
        # Note that we have to provide a copy of the lines, because of Matplotlib
        ax.add_collection(copy.copy(self.lines))
