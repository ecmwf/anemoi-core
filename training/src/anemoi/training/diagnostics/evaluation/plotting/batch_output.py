# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Built-in plot function adapters for :class:`BatchOutputPlot`.

Each adapter wraps an underlying plotting function with the shared
:class:`BatchOutputPlotFn` signature. All plot-specific kwargs have
code-defined defaults, so they are **optional** in the YAML config —
only specify them to override the default.

Built-in plot functions and their optional kwargs
-------------------------------------------------

``sample_plot_fn``
    Sample map plot (input vs prediction vs truth per variable).

    - ``per_sample`` (int, default ``6``): number of samples per variable row.
    - ``accumulation_levels_plot`` (list, default ``DEFAULT_ACCUMULATION_LEVELS``):
      colour levels in mm for precipitation fields.
    - ``prediction_label`` (str, default ``"pred"``): label for the prediction panel.
    - ``auxiliary_label`` (str, default ``"corrupted targets"``): label for the
      auxiliary panel (only shown when ``with_auxiliary: true`` on the callback).

``spectrum_plot_fn``
    Power spectrum plot.

    - ``min_delta`` (float, default ``None``): minimum delta for spectrum plot.

``histogram_plot_fn``
    Histogram plot.

    - ``log_scale`` (bool, default ``False``): use log scale on the y-axis.

``ensemble_plot_fn``
    Ensemble spread/mean/error map plot.

    - ``accumulation_levels_plot`` (list, default ``DEFAULT_ACCUMULATION_LEVELS``):
      colour levels in mm for precipitation fields.

``zonal_cross_section_plot_fn``
    Zonal-mean pressure-latitude or pressure-longitude cross-section (truth / pred / error).

    - ``variable`` (str, default ``"t"``): variable prefix; selects ``t_50``, ``t_100``, … by parsing the pressure-level suffix.
    - ``axis`` (str, default ``"lat"``): ``"lat"`` puts latitude on x-axis, ``"lon"`` puts longitude on x-axis.
    - ``lat_min``, ``lat_max`` (float or None, default ``None``): latitude bounding box; ``None`` means no restriction.
    - ``lon_min``, ``lon_max`` (float or None, default ``None``): longitude bounding box; ``None`` means no restriction.
    - ``n_bins`` (int, default ``72``): number of bins along the plot axis (72 → 2.5° bins for latitude).

Rendering settings (datashader, projection, colormaps, precip_and_related_fields)
are read from the ``settings`` object passed by the callback — configure them
under ``diagnostics.plot.settings`` in the YAML, not inside ``plot_fn``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# --- Thin adapters around the existing evaluation.plotting.* functions ------
#
# These preserve the current visual output while letting all four map-style
# plots be driven by a single callback. See the "Plot function contracts"
# section of docs/modules/diagnostics.rst for the expected plot_fn signature.


def sample_plot_fn(
    parameters: dict,
    *,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latlons: np.ndarray,
    auxiliary: np.ndarray | None = None,
    settings: Any | None = None,
    per_sample: int = 6,
    accumulation_levels_plot: list | None = None,
    prediction_label: str = "pred",
    auxiliary_label: str = "corrupted targets",
    **_kwargs: Any,
) -> Figure:
    """Adapter for ``plot_predicted_multilevel_flat_sample`` (PlotSample)."""
    from anemoi.training.diagnostics.evaluation.plotting.sample import plot_predicted_multilevel_flat_sample
    from anemoi.training.diagnostics.evaluation.plotting.settings import DEFAULT_ACCUMULATION_LEVELS

    levels = accumulation_levels_plot if accumulation_levels_plot is not None else DEFAULT_ACCUMULATION_LEVELS
    return plot_predicted_multilevel_flat_sample(
        parameters,
        per_sample,
        latlons,
        levels,
        x,
        y_true,
        y_pred,
        datashader=getattr(settings, "datashader", True),
        precip_and_related_fields=getattr(settings, "precip_and_related_fields", None),
        colormaps=getattr(settings, "colormaps", None),
        projection_kind=getattr(settings, "projection_kind", "equirectangular"),
        prediction_label=prediction_label,
        auxiliary=auxiliary,
        auxiliary_label=auxiliary_label,
    )


def spectrum_plot_fn(
    parameters: dict,
    *,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latlons: np.ndarray,
    auxiliary: np.ndarray | None = None,  # noqa: ARG001
    settings: Any | None = None,  # noqa: ARG001
    min_delta: float | None = None,
    **_kwargs: Any,
) -> Figure:
    """Adapter for ``plot_power_spectrum`` (PlotSpectrum)."""
    from anemoi.training.diagnostics.evaluation.plotting.spectrum import plot_power_spectrum

    return plot_power_spectrum(parameters, latlons, x, y_true, y_pred, min_delta=min_delta)


def histogram_plot_fn(
    parameters: dict,
    *,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latlons: np.ndarray | None = None,  # noqa: ARG001
    auxiliary: np.ndarray | None = None,  # noqa: ARG001
    settings: Any | None = None,
    log_scale: bool = False,
    **_kwargs: Any,
) -> Figure:
    """Adapter for ``plot_histogram`` (PlotHistogram)."""
    from anemoi.training.diagnostics.evaluation.plotting.histogram import plot_histogram

    return plot_histogram(
        parameters,
        x,
        y_true,
        y_pred,
        getattr(settings, "precip_and_related_fields", None),
        log_scale,
    )


def zonal_cross_section_plot_fn(
    parameters: dict,
    *,
    x: np.ndarray,  # noqa: ARG001
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latlons: np.ndarray,
    auxiliary: np.ndarray | None = None,  # noqa: ARG001
    settings: Any | None = None,  # noqa: ARG001
    variable: str = "t",
    axis: str = "lat",
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    n_bins: int = 72,
    **_kwargs: Any,
) -> Figure:
    """Zonal-mean pressure-latitude or pressure-longitude cross-section.

    Grid points are filtered by the lat/lon bounding box (``None`` means no
    restriction on that bound), then binned along the plot axis and averaged
    over the orthogonal axis, producing a clean (n_levels × n_bins) field.

    Parameters
    ----------
    variable : str
        Variable prefix to select (e.g. ``"t"`` picks ``t_50``, ``t_100``, …).
        Pressure level is parsed from the numeric suffix.
    axis : str
        ``"lat"`` → latitude on x-axis, average over selected longitudes.
        ``"lon"`` → longitude on x-axis, average over selected latitudes.
    lat_min, lat_max : float or None
        Latitude bounds (degrees). ``None`` means no restriction.
    lon_min, lon_max : float or None
        Longitude bounds (degrees). ``None`` means no restriction.
    n_bins : int
        Number of bins along the plot axis (default 72 → 2.5° for lat).
    """
    import matplotlib.pyplot as plt

    lats, lons = latlons[:, 0], latlons[:, 1]
    mask = np.ones(len(lats), dtype=bool)
    if lat_min is not None:
        mask &= lats >= lat_min
    if lat_max is not None:
        mask &= lats <= lat_max
    if lon_min is not None:
        mask &= lons >= lon_min
    if lon_max is not None:
        mask &= lons <= lon_max

    plot_coords = lats[mask] if axis == "lat" else lons[mask]
    axis_label = "Latitude (°)" if axis == "lat" else "Longitude (°)"
    axis_range = (-90.0, 90.0) if axis == "lat" else (-180.0, 180.0)

    filtered = {}
    for idx, v in parameters.items():
        name = v[0] if isinstance(v, tuple) else v
        if name.startswith(variable + "_"):
            try:
                filtered[idx] = (name, int(name.split("_")[-1]))
            except ValueError:
                continue

    if not filtered or not mask.any():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data in selected region", ha="center", va="center", transform=ax.transAxes)
        return fig

    sorted_items = sorted(filtered.items(), key=lambda kv: kv[1][1])
    indices = [idx for idx, _ in sorted_items]
    pressure_levels = [meta[1] for _, meta in sorted_items]

    bins = np.linspace(axis_range[0], axis_range[1], n_bins + 1)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.clip(np.digitize(plot_coords, bins) - 1, 0, n_bins - 1)

    def zonal_mean(arr: np.ndarray, var_idx: int) -> np.ndarray:
        col = arr[..., var_idx].squeeze()[mask]
        return np.array([
            col[bin_idx == b].mean() if np.any(bin_idx == b) else np.nan
            for b in range(n_bins)
        ])

    truth_grid = np.stack([zonal_mean(y_true, i) for i in indices])  # (n_levels, n_bins)
    pred_grid = np.stack([zonal_mean(y_pred, i) for i in indices])
    err_grid = pred_grid - truth_grid

    vmin, vmax = np.nanmin(truth_grid), np.nanmax(truth_grid)
    emax = np.nanmax(np.abs(err_grid))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, data, title, kw in zip(
        axs,
        [truth_grid, pred_grid, err_grid],
        [f"{variable} truth", f"{variable} pred", f"{variable} error (pred − truth)"],
        [
            {"vmin": vmin, "vmax": vmax, "cmap": "RdBu_r"},
            {"vmin": vmin, "vmax": vmax, "cmap": "RdBu_r"},
            {"vmin": -emax, "vmax": emax, "cmap": "bwr"},
        ],
    ):
        im = ax.pcolormesh(bin_centres, pressure_levels, data, **kw)
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Pressure (hPa)")
        ax.invert_yaxis()
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

    return fig


def ensemble_plot_fn(
    parameters: dict,
    *,
    x: np.ndarray,  # noqa: ARG001
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latlons: np.ndarray,
    auxiliary: np.ndarray | None = None,  # noqa: ARG001
    settings: Any | None = None,
    accumulation_levels_plot: list | None = None,
    **_kwargs: Any,
) -> Figure:
    """Adapter for ``plot_predicted_ensemble`` (PlotEnsSample)."""
    from anemoi.training.diagnostics.evaluation.plotting.ensemble import plot_predicted_ensemble
    from anemoi.training.diagnostics.evaluation.plotting.settings import DEFAULT_ACCUMULATION_LEVELS

    levels = accumulation_levels_plot if accumulation_levels_plot is not None else DEFAULT_ACCUMULATION_LEVELS
    return plot_predicted_ensemble(
        parameters=parameters,
        latlons=latlons,
        clevels=levels,
        y_true=np.asarray(y_true).squeeze(),
        y_pred=np.asarray(y_pred).squeeze(),
        datashader=getattr(settings, "datashader", True),
        precip_and_related_fields=getattr(settings, "precip_and_related_fields", None),
        colormaps=getattr(settings, "colormaps", None),
        projection_kind=getattr(settings, "projection_kind", "equirectangular"),
    )
