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
        return np.degrees(x), np.degrees(y)


class Coastlines:
    """Class to plot coastlines from a GeoJSON file."""

    def __init__(self, projection: EquirectangularProjection | None = None) -> None:
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
        lines = []
        for feature in self.data["features"]:
            coords = feature["geometry"]["coordinates"]
            x, y = zip(*coords, strict=False)
            lines.append(list(zip(*self.projection(x, y), strict=False)))
        self.lines = LineCollection(lines, linewidth=0.5, color="black")

    def plot_continents(self, ax: plt.Axes) -> None:
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
class Borders:
    """Class to add Cartopy political borders to a GeoAxes."""

    def __init__(self, scale: str = "50m") -> None:
        try:
            import cartopy.feature as cfeature

            self.cfeature = cfeature
        except ModuleNotFoundError as e:
            msg = "Please install cartopy to enable border plotting."
            raise ModuleNotFoundError(msg) from e
        self.scale = scale

    def plot_borders(self, ax: plt.Axes) -> None:
        if not hasattr(ax, "add_feature"):
            LOGGER.warning("Axis is not a GeoAxes; skipping border plotting.")
            return
        ax.add_feature(self.cfeature.BORDERS.with_scale(self.scale), linestyle=":", zorder=1)


class MapFeatures:
    """Container class for optional map features (coastlines, borders, etc.)."""

    def __init__(self, continents: Coastlines | None = None, borders: Borders | None = None) -> None:
        self.continents = continents
        self.borders = borders

    def plot(self, ax: plt.Axes) -> None:
        """Plot all enabled map features on the given axis."""
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
    continents = Coastlines()
    borders = None
    try:
        borders = Borders(scale="50m")
    except (ModuleNotFoundError, ImportError):
        LOGGER.warning("Borders disabled (Cartopy likely not available).")
    return MapFeatures(continents=continents, borders=borders)


# Construct once at import time
map_features = _build_map_features()
