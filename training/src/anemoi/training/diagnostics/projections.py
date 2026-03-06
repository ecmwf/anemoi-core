import numpy as np


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


class Projection(BaseProjection):
    """Single interface for projections: use in both maps (callable) and plots (.project(latlons)).

    Backend is either EquirectangularProjection or Cartopy's ccrs.LambertConformal
    (or any Cartopy CRS with transform_points). Use Projection.equirectangular() or
    Projection.lambert_conformal(latlon) to build.
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
        """Project (N, 2) [lat, lon] in degrees to (proj0, proj1). Use in plots."""
        latlons = np.asanyarray(latlons)
        lat, lon = latlons[:, 0], latlons[:, 1]
        return self(lon, lat)

    @classmethod
    def equirectangular(cls) -> "Projection":
        """Equirectangular projection (lon/lat in radians)."""
        return cls(EquirectangularProjection())

    @classmethod
    def lambert_conformal(cls, latlon: np.ndarray) -> "Projection":
        """Lambert Conformal using Cartopy's ccrs.LambertConformal fitted to the given points.

        Parameters
        ----------
        latlon : np.ndarray
            Shape (N, 2) with columns [latitude, longitude] in degrees.

        Returns
        -------
        Projection
            Uses ccrs.LambertConformal; requires Cartopy.
        """
        crs = lambert_conformal_from_latlon_points(latlon)
        return cls(crs)

    @classmethod
    def from_kind(cls, latlons: np.ndarray, kind: str = "equirectangular") -> "Projection":
        """Build a Projection from a kind string (equirectangular or lambert_conformal)."""
        if kind == "equirectangular":
            return cls.equirectangular()
        if kind == "lambert_conformal":
            return cls.lambert_conformal(latlons)
        raise ValueError(kind)

    def crs_for_axes(self) -> object | None:
        """CRS for plt.subplots(subplot_kw={"projection": ...}). None for equirectangular."""
        return self._backend if self._is_cartopy else None

    @classmethod
    def for_plot(
        cls,
        latlons: np.ndarray,
        kind: str = "equirectangular",
    ) -> tuple[tuple[np.ndarray, np.ndarray], object | None, object | None]:
        """Projection data for plotting: (pc_lon, pc_lat), proj for axes, transform for scatter.

        Returns
        -------
        (pc_lon, pc_lat), proj, transform
            Use proj in subplot_kw={"projection": proj} when proj is not None.
            Pass transform to scatter. Reuse this in all plot functions.
        """
        projection = cls.from_kind(latlons, kind)
        pc_lon, pc_lat = projection.project(latlons)
        proj = projection.crs_for_axes()
        return (pc_lon, pc_lat), proj, proj


def equirectangular_projection(latlons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 2) [lat, lon] in degrees to (pc_lat, pc_lon). Backward-compat wrapper."""
    x, y = Projection.equirectangular().project(latlons)
    return y, x  # pc_lat, pc_lon


def projection_crs_for_axes(
    latlons: np.ndarray,
    kind: str = "equirectangular",
) -> object | None:
    """Return projection for map axes: None for equirectangular, Cartopy Lambert for lambert_conformal.

    For equirectangular we use our own projection for coordinates only; no Cartopy axes CRS.
    For lambert_conformal we return ccrs.LambertConformal for subplot_kw={"projection": proj}.

    Parameters
    ----------
    latlons : np.ndarray
        Shape (N, 2) with columns [latitude, longitude] in degrees.
    kind : str
        "equirectangular" (no axes CRS; use Projection.equirectangular() for coords only)
        or "lambert_conformal" (ccrs.LambertConformal fitted to latlons).

    Returns
    -------
    None or cartopy.crs.LambertConformal
        None for equirectangular (regular axes). Lambert CRS for lambert_conformal (requires Cartopy).
    """
    if kind == "equirectangular":
        return None
    if kind == "lambert_conformal":
        return lambert_conformal_from_latlon_points(latlons)
    raise ValueError(kind)


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
