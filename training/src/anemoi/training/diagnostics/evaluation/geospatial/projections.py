# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class BaseProjection:
    """Base class for map projections: callable(lon, lat) -> (x, y), optional inverse(x, y) -> (lon, lat)."""

    def __call__(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (lon, lat) in degrees to (x, y)."""
        raise NotImplementedError

    def inverse(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert (x, y) back to (lon, lat) in degrees."""
        msg = f"{self.__class__.__name__} does not implement inverse."
        raise NotImplementedError(msg)


class EquirectangularProjection(BaseProjection):
    """Convert lat/lon in degrees to equirectangular (radians) x, y."""

    def __init__(self) -> None:
        self.x_offset = 0.0
        self.y_offset = 0.0

    def __call__(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon_rad = np.radians(np.asanyarray(lon))
        lat_rad = np.radians(np.asanyarray(lat))
        x = np.array(
            [v - 2 * np.pi if v > np.pi else v for v in lon_rad],
            dtype=lon_rad.dtype,
        )
        y = lat_rad
        return x, y

    @staticmethod
    def inverse(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.degrees(x), np.degrees(y)


class MapProjection(BaseProjection):
    """Unified interface for map projections used in diagnostic plots.

    Wraps either an :class:`EquirectangularProjection` or a Cartopy CRS
    (``cartopy.crs.*``) and exposes a consistent API for both coordinate
    transforms and Cartopy axes setup.

    Coordinate concepts
    -------------------
    ``axes_crs``
        The Cartopy CRS that defines how the *map axes* renders — e.g.
        ``ccrs.Robinson()``.  Passed to
        ``plt.subplots(subplot_kw={"projection": axes_crs})``.
        ``None`` for equirectangular (plain matplotlib axes, no Cartopy).

    ``data_crs``
        The CRS that describes the *input data* coordinate system — always
        ``ccrs.PlateCarree()`` for us, since data is in degrees lat/lon.
        Passed to ``ax.scatter(..., transform=data_crs)`` so Cartopy knows
        how to reproject the points into ``axes_crs`` before drawing.
        ``None`` for equirectangular (no reprojection needed).

    The rendering pipeline is::

        rendered = axes_crs.reproject(data_points, from_crs=data_crs)

    For equirectangular both are ``None`` because coordinates are handled
    directly without Cartopy axes.
    """

    def __init__(
        self,
        backend: BaseProjection | object,
    ) -> None:
        """Backend: EquirectangularProjection, or a Cartopy CRS (e.g. ccrs.LambertConformal)."""
        self._backend = backend
        self._is_cartopy = hasattr(backend, "transform_points")

    def __call__(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (lon, lat) in degrees to (x, y). Used by Coastlines/Borders."""
        lon = np.asanyarray(lon)
        lat = np.asanyarray(lat)
        if self._is_cartopy:
            import cartopy.crs as ccrs

            pts = self._backend.transform_points(ccrs.Geodetic(), lon, lat)
            return pts[..., 0].copy(), pts[..., 1].copy()
        return self._backend(lon, lat)

    def inverse(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert (x, y) back to (lon, lat) in degrees."""
        x = np.asanyarray(x)
        y = np.asanyarray(y)
        if self._is_cartopy:
            import cartopy.crs as ccrs

            pts = ccrs.Geodetic().transform_points(self._backend, x, y)
            return pts[..., 0].copy(), pts[..., 1].copy()
        return self._backend.inverse(x, y)

    def project(self, latlons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N, 2) [lat, lon] in degrees to (x, y). Use in plots."""
        latlons = np.asanyarray(latlons)
        lat, lon = latlons[:, 0], latlons[:, 1]
        return self(lon, lat)

    @classmethod
    def equirectangular(cls) -> "MapProjection":
        """Equirectangular projection (lon/lat in radians)."""
        return cls(EquirectangularProjection())

    @classmethod
    def lambert_conformal(cls, latlon: np.ndarray) -> "MapProjection":
        """Lambert Conformal using Cartopy's ccrs.LambertConformal fitted to the given points.

        Parameters
        ----------
        latlon : np.ndarray
            Shape (N, 2) with columns [latitude, longitude] in degrees.

        Returns
        -------
        MapProjection
            Uses ccrs.LambertConformal; requires Cartopy.
        """
        axes_crs = lambert_conformal_from_latlon_points(latlon)
        return cls(axes_crs)

    @classmethod
    def from_kind(cls, latlons: np.ndarray, kind: str = "equirectangular") -> "MapProjection":
        """Build a MapProjection from a kind string.

        Built-in kinds: ``'equirectangular'`` (no cartopy) and ``'lambert_conformal'``
        (auto-fitted to the data domain).  Any other string is resolved as a
        ``cartopy.crs`` class name in snake_case, e.g. ``'robinson'`` →
        ``cartopy.crs.Robinson()``.  Requires cartopy for all non-equirectangular kinds.

        Notes
        -----
        Dynamic Cartopy projections are instantiated with **default constructor
        arguments**. For example, ``'orthographic'`` centres on (longitude=0,
        latitude=0). If the defaults are unsuitable for your domain, use
        ``'lambert_conformal'`` (auto-fitted) or subclass :class:`MapProjection`
        to pass custom parameters.
        """
        if kind == "equirectangular":
            return cls.equirectangular()
        if kind == "lambert_conformal":
            latlons = np.asanyarray(latlons)
            lat_span = latlons[:, 0].max() - latlons[:, 0].min()
            lon_span = latlons[:, 1].max() - latlons[:, 1].min()
            if lat_span > 60 or lon_span > 180:
                LOGGER.warning(
                    "Lambert Conformal is designed for regional domains. "
                    "The current domain spans %.1f° latitude and %.1f° longitude — "
                    "falling back to equirectangular.",
                    lat_span,
                    lon_span,
                )
                return cls.equirectangular()
            return cls.lambert_conformal(latlons)
        # Fall back to dynamic cartopy CRS lookup: snake_case → CamelCase class name
        try:
            import cartopy.crs as ccrs
        except ModuleNotFoundError as e:
            msg = f"projection_kind='{kind}' requires cartopy. Install with: pip install anemoi-training[plotting]"
            raise ModuleNotFoundError(msg) from e
        crs_name = "".join(part.capitalize() for part in kind.split("_"))
        crs_cls = getattr(ccrs, crs_name, None)
        if crs_cls is None:
            msg = (
                f"Unknown projection_kind='{kind}'. "
                f"Use 'equirectangular', 'lambert_conformal', or any cartopy.crs class name in snake_case "
                f"(e.g. 'robinson', 'mollweide', 'orthographic')."
            )
            raise ValueError(msg)
        LOGGER.info(
            "Instantiating cartopy.crs.%s() with default constructor arguments. "
            "To use non-default parameters (e.g. a different central longitude), "
            "use 'lambert_conformal' (auto-fitted) or subclass MapProjection.",
            crs_name,
        )
        return cls(crs_cls())

    def axes_crs(self) -> object | None:
        """Cartopy CRS for ``plt.subplots(subplot_kw={"projection": axes_crs})``.

        Returns ``None`` for equirectangular (plain matplotlib axes, no Cartopy).
        """
        return self._backend if self._is_cartopy else None

    @classmethod
    def for_plot(
        cls,
        latlons: np.ndarray,
        kind: str = "equirectangular",
    ) -> tuple[tuple[np.ndarray, np.ndarray], object | None, object | None]:
        """Return everything needed to set up a projected plot.

        Parameters
        ----------
        latlons : np.ndarray
            Shape (N, 2) with columns [latitude, longitude] in degrees.
        kind : str
            Projection kind string (see :meth:`from_kind`).

        Returns
        -------
        (pc_lon, pc_lat), axes_crs, data_crs
            ``axes_crs`` — pass to ``plt.subplots(subplot_kw={"projection": axes_crs})``.
            ``data_crs`` — pass to ``ax.scatter(..., transform=data_crs)`` and
            ``ax.set_extent(..., crs=data_crs)``. Always ``ccrs.PlateCarree()`` when
            not None (data lives in degrees lat/lon).
            Both are ``None`` for equirectangular (plain matplotlib, no Cartopy).
        """
        projection = cls.from_kind(latlons, kind)
        _axes_crs = projection.axes_crs()
        if _axes_crs is not None:
            # Cartopy axes: keep data in degrees and pass PlateCarree as data_crs.
            # Cartopy reprojects from data_crs into axes_crs internally before drawing.
            import cartopy.crs as ccrs

            latlons = np.asanyarray(latlons)
            pc_lon, pc_lat = latlons[:, 1], latlons[:, 0]
            return (pc_lon, pc_lat), _axes_crs, ccrs.PlateCarree()
        pc_lon, pc_lat = projection.project(latlons)
        return (pc_lon, pc_lat), None, None


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

    # LCC requires |lat_1 + lat_2| > 0 (parallels must not cancel).
    # When the domain straddles the equator symmetrically, nudge both parallels
    # slightly northward to satisfy the constraint.
    if abs(std_parallel_1 + std_parallel_2) < 1e-6:
        std_parallel_1 += 1e-3
        std_parallel_2 += 1e-3

    return ccrs.LambertConformal(
        central_latitude=central_latitude,
        central_longitude=central_longitude,
        standard_parallels=[std_parallel_1, std_parallel_2],
    )
