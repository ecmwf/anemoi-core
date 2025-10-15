# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from dataclasses import dataclass

import datashader as dsh
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
from datashader.mpl_ext import dsshow
from matplotlib.collections import LineCollection
from matplotlib.collections import PathCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import Colormap
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from pyshtools.expand import SHGLQ
from pyshtools.expand import SHExpandGLQ
from scipy.interpolate import griddata
from torch import Tensor
from torch import nn

from anemoi.training.diagnostics.maps import Coastlines
from anemoi.training.diagnostics.maps import EquirectangularProjection
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

LOGGER = logging.getLogger(__name__)

continents = Coastlines()
LAYOUT = "tight"


@dataclass
class LatLonData:
    latitudes: np.ndarray
    longitudes: np.ndarray
    data: np.ndarray


def equirectangular_projection(latlons: np.array) -> np.array:
    pc = EquirectangularProjection()
    lat, lon = latlons[:, 0], latlons[:, 1]
    pc_lon, pc_lat = pc(lon, lat)
    return pc_lat, pc_lon


def argsort_variablename_variablelevel(data: list[str], metadata_variables: dict | None = None) -> list[int]:
    """Custom sort key to process the strings.

    Sort parameter names by alpha part, then by numeric part at last
    position (variable level) if available, then by the original string.

    Parameters
    ----------
    data : list[str]
        List of strings to sort.
    metadata_variables : dict, optional
        Dictionary of variable names and indices, by default None

    Returns
    -------
    list[int]
        Sorted indices of the input list.
    """
    extract_variable_group_and_level = ExtractVariableGroupAndLevel(
        {"default": ""},
        metadata_variables,
    )

    def custom_sort_key(index: int) -> tuple:
        s = data[index]  # Access the element by index
        _, alpha_part, numeric_part = extract_variable_group_and_level.get_group_and_level(s)
        if numeric_part is None:
            numeric_part = float("inf")
        return (alpha_part, numeric_part, s)

    # Generate argsort indices
    return sorted(range(len(data)), key=custom_sort_key)


def init_plot_settings() -> None:
    """Initialize matplotlib plot settings."""
    small_font_size = 8
    medium_font_size = 10

    mplstyle.use("fast")
    plt.rcParams["path.simplify_threshold"] = 0.9

    plt.rc("font", size=small_font_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_font_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium_font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_font_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_font_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_font_size)  # legend fontsize
    plt.rc("figure", titlesize=small_font_size)  # fontsize of the figure title


def _hide_axes_ticks(ax: plt.Axes) -> None:
    """Hide x/y-axis ticks.

    Parameters
    ----------
    ax : matplotlib.axes
        Axes object handle

    """
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)


def plot_loss(
    x: np.ndarray,
    colors: np.ndarray,
    xticks: dict[str, int] | None = None,
    legend_patches: list | None = None,
) -> Figure:
    """Plots data for one multilevel sample.

    Parameters
    ----------
    x : np.ndarray
        Data for Plotting of shape (npred,)
    colors : np.ndarray
        Colors for the bars.
    xticks : dict, optional
        Dictionary of xticks, by default None
    legend_patches : list, optional
        List of legend patches, by default None

    Returns
    -------
    Figure
        The figure object handle.

    """
    # create plot
    # more space for legend
    # TODO(who?): make figsize more flexible depending on the number of bars
    figsize = (8, 3) if legend_patches else (4, 3)
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout=LAYOUT)
    # histogram plot
    ax.bar(np.arange(x.size), x, color=colors, log=1)

    # add xticks and legend if given
    if xticks:
        ax.set_xticks(list(xticks.values()), list(xticks.keys()), rotation=60)
    if legend_patches:
        # legend outside and to the right of the plot
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc="upper left")

    return fig


def plot_power_spectrum(
    parameters: dict[str, int],
    latlons: np.ndarray,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_delta: float | None = None,
) -> Figure:
    """Plots power spectrum.

    NB: this can be very slow for large data arrays
    call it as infrequently as possible!

    Parameters
    ----------
    parameters : dict[str, int]
        Dictionary of variable names and indices
    latlons : np.ndarray
        lat/lon coordinates array, shape (lat*lon, 2)
    x : np.ndarray
        Input data of shape (lat*lon, nvar*level)
    y_true : np.ndarray
        Expected data of shape (lat*lon, nvar*level)
    y_pred : np.ndarray
        Predicted data of shape (lat*lon, nvar*level)
    min_delta: float, optional
        Minimum distance between lat/lon points, if None defaulted to 1km

    Returns
    -------
    Figure
        The figure object handle.

    """
    min_delta = min_delta or 0.0003
    n_plots_x, n_plots_y = len(parameters), 1

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize, layout=LAYOUT)
    if n_plots_x == 1:
        ax = [ax]

    pc_lat, pc_lon = equirectangular_projection(latlons)

    pc_lon = np.array(pc_lon)
    pc_lat = np.array(pc_lat)
    # Calculate delta_lat on the projected grid
    delta_lat = abs(np.diff(pc_lat))
    non_zero_delta_lat = delta_lat[delta_lat != 0]
    min_delta_lat = np.min(abs(non_zero_delta_lat))

    if min_delta_lat < min_delta:
        LOGGER.warning(
            "Min. distance between lat/lon points is < specified minimum distance. Defaulting to min_delta=%s.",
            min_delta,
        )
        min_delta_lat = min_delta

    # Define a regular grid for interpolation
    n_pix_lat = int(np.floor(abs(pc_lat.max() - pc_lat.min()) / min_delta_lat))
    n_pix_lon = (n_pix_lat - 1) * 2 + 1  # 2*lmax + 1
    regular_pc_lon = np.linspace(pc_lon.min(), pc_lon.max(), n_pix_lon)
    regular_pc_lat = np.linspace(pc_lat.min(), pc_lat.max(), n_pix_lat)
    grid_pc_lon, grid_pc_lat = np.meshgrid(regular_pc_lon, regular_pc_lat)

    for plot_idx, (variable_idx, (variable_name, output_only)) in enumerate(parameters.items()):
        yt = (y_true if y_true.ndim == 1 else y_true[..., variable_idx]).reshape(-1)
        yp = (y_pred if y_pred.ndim == 1 else y_pred[..., variable_idx]).reshape(-1)

        # check for any nan in yt
        nan_flag = np.isnan(yt).any()

        method = "linear" if nan_flag else "cubic"
        if output_only:
            xt = (x if x.ndim == 1 else x[..., variable_idx]).reshape(-1)
            yt_i = griddata((pc_lon, pc_lat), (yt - xt), (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)
            yp_i = griddata((pc_lon, pc_lat), (yp - xt), (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)
        else:
            yt_i = griddata((pc_lon, pc_lat), yt, (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)
            yp_i = griddata((pc_lon, pc_lat), yp, (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)

        # Masking NaN values
        if nan_flag:
            mask = np.isnan(yt_i)
            if mask.any():
                yt_i = np.where(mask, 0.0, yt_i)
                yp_i = np.where(mask, 0.0, yp_i)

        amplitude_t = np.array(compute_spectra(yt_i))
        amplitude_p = np.array(compute_spectra(yp_i))

        ax[plot_idx].loglog(
            np.arange(1, amplitude_t.shape[0]),
            amplitude_t[1 : (amplitude_t.shape[0])],
            label="Truth (data)",
        )
        ax[plot_idx].loglog(
            np.arange(1, amplitude_p.shape[0]),
            amplitude_p[1 : (amplitude_p.shape[0])],
            label="Predicted",
        )

        ax[plot_idx].legend()
        ax[plot_idx].set_title(variable_name)

        ax[plot_idx].set_xlabel("$k$")
        ax[plot_idx].set_ylabel("$P(k)$")
        ax[plot_idx].set_aspect("auto", adjustable=None)
    return fig


def compute_spectra(field: np.ndarray) -> np.ndarray:
    """Compute spectral variability of a field by wavenumber.

    Parameters
    ----------
    field : np.ndarray
        lat lon field to calculate the spectra of

    Returns
    -------
    np.ndarray
        spectra of field by wavenumber

    """
    field = np.array(field)

    # compute real and imaginary parts of power spectra of field
    lmax = field.shape[0] - 1  # maximum degree of expansion
    zero_w = SHGLQ(lmax)
    coeffs_field = SHExpandGLQ(field, w=zero_w[1], zero=zero_w[0])

    # Re**2 + Im**2
    coeff_amp = coeffs_field[0, :, :] ** 2 + coeffs_field[1, :, :] ** 2

    # sum over meridional direction
    return np.sum(coeff_amp, axis=0)


def plot_histogram(
    parameters: dict[str, int],
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    precip_and_related_fields: list | None = None,
) -> Figure:
    """Plots histogram.

    NB: this can be very slow for large data arrays
    call it as infrequently as possible!

    Parameters
    ----------
    parameters : dict[str, int]
        Dictionary of variable names and indices
    x : np.ndarray
        Input data of shape (lat*lon, nvar*level)
    y_true : np.ndarray
        Expected data of shape (lat*lon, nvar*level)
    y_pred : np.ndarray
        Predicted data of shape (lat*lon, nvar*level)
    precip_and_related_fields : list, optional
        List of precipitation-like variables, by default []

    Returns
    -------
    Figure
        The figure object handle.

    """
    precip_and_related_fields = precip_and_related_fields or []

    n_plots_x, n_plots_y = len(parameters), 1

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize, layout=LAYOUT)
    if n_plots_x == 1:
        ax = [ax]

    for plot_idx, (variable_idx, (variable_name, output_only)) in enumerate(parameters.items()):
        yt = (y_true if y_true.ndim == 1 else y_true[..., variable_idx]).reshape(-1)
        yp = (y_pred if y_pred.ndim == 1 else y_pred[..., variable_idx]).reshape(-1)
        # postprocessed outputs so we need to handle possible NaNs

        # Calculate the histogram and handle NaNs
        if output_only:
            # histogram of true increment and predicted increment
            xt = (x if x.ndim == 1 else x[..., variable_idx]).reshape(-1) * int(output_only)
            yt_xt = yt - xt
            yp_xt = yp - xt
            # enforce the same binning for both histograms
            bin_min = min(np.nanmin(yt_xt), np.nanmin(yp_xt))
            bin_max = max(np.nanmax(yt_xt), np.nanmax(yp_xt))
            hist_yt, bins_yt = np.histogram(yt_xt[~np.isnan(yt_xt)], bins=100, density=True, range=[bin_min, bin_max])
            hist_yp, bins_yp = np.histogram(yp_xt[~np.isnan(yp_xt)], bins=100, density=True, range=[bin_min, bin_max])
        else:
            # enforce the same binning for both histograms
            bin_min = min(np.nanmin(yt), np.nanmin(yp))
            bin_max = max(np.nanmax(yt), np.nanmax(yp))
            hist_yt, bins_yt = np.histogram(yt[~np.isnan(yt)], bins=100, density=True, range=[bin_min, bin_max])
            hist_yp, bins_yp = np.histogram(yp[~np.isnan(yp)], bins=100, density=True, range=[bin_min, bin_max])

        # Visualization trick for tp
        if variable_name in precip_and_related_fields:
            # in-place multiplication does not work here because variables are different numpy types
            hist_yt = hist_yt * bins_yt[:-1]
            hist_yp = hist_yp * bins_yp[:-1]
        # Plot the modified histogram
        ax[plot_idx].bar(bins_yt[:-1], hist_yt, width=np.diff(bins_yt), color="blue", alpha=0.7, label="Truth (data)")
        ax[plot_idx].bar(bins_yp[:-1], hist_yp, width=np.diff(bins_yp), color="red", alpha=0.7, label="Predicted")

        ax[plot_idx].set_title(variable_name)
        ax[plot_idx].set_xlabel(variable_name)
        ax[plot_idx].set_ylabel("Density")
        ax[plot_idx].legend()
        ax[plot_idx].set_aspect("auto", adjustable=None)

    return fig


def plot_predicted_multilevel_flat_sample(
    parameters: dict[str, int],
    n_plots_per_sample: int,
    latlons: np.ndarray,
    clevels: float,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    datashader: bool = False,
    precip_and_related_fields: list | None = None,
    colormaps: dict[str, Colormap] | None = None,
) -> Figure:
    """Plots data for one multilevel latlon-"flat" sample.

    NB: this can be very slow for large data arrays
    call it as infrequently as possible!

    Parameters
    ----------
    parameters : dict[str, int]
        Dictionary of variable names and indices
    n_plots_per_sample : int
        Number of plots per sample
    latlons : np.ndarray
        lat/lon coordinates array, shape (lat*lon, 2)
    clevels : float
        Accumulation levels used for precipitation related plots
    x : np.ndarray
        Input data of shape (lat*lon, nvar*level)
    y_true : np.ndarray
        Expected data of shape (lat*lon, nvar*level)
    y_pred : np.ndarray
        Predicted data of shape (lat*lon, nvar*level)
    datashader: bool, optional
        Scatter plot, by default False
    precip_and_related_fields : list, optional
        List of precipitation-like variables, by default []
    colormaps : dict[str, Colormap], optional
        Dictionary of colormaps, by default None

    Returns
    -------
    Figure
        The figure object handle.

    """
    n_plots_x, n_plots_y = len(parameters), n_plots_per_sample

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize, layout=LAYOUT)

    pc_lat, pc_lon = equirectangular_projection(latlons)
    if colormaps is None:
        colormaps = {}

    for plot_idx, (variable_idx, (variable_name, output_only)) in enumerate(parameters.items()):
        xt = (x if x.ndim == 1 else x[..., variable_idx]).reshape(-1) * int(output_only)
        yt = (y_true if y_true.ndim == 1 else y_true[..., variable_idx]).reshape(-1)
        yp = (y_pred if y_pred.ndim == 1 else y_pred[..., variable_idx]).reshape(-1)

        # get the colormap for the variable as defined in config file
        cmap = colormaps.default.get_cmap() if colormaps.get("default") else cm.get_cmap("viridis")
        error_cmap = colormaps.error.get_cmap() if colormaps.get("error") else cm.get_cmap("bwr")
        for key in colormaps:
            if key not in ["default", "error"] and variable_name in colormaps[key].variables:
                cmap = colormaps[key].get_cmap()
                continue
        if n_plots_x > 1:
            plot_flat_sample(
                fig,
                ax[plot_idx, :],
                pc_lon,
                pc_lat,
                xt,
                yt,
                yp,
                variable_name,
                clevels,
                datashader,
                precip_and_related_fields,
                cmap=cmap,
                error_cmap=error_cmap,
            )
        else:
            plot_flat_sample(
                fig,
                ax,
                pc_lon,
                pc_lat,
                xt,
                yt,
                yp,
                variable_name,
                clevels,
                datashader,
                precip_and_related_fields,
                cmap=cmap,
                error_cmap=error_cmap,
            )

    return fig


def plot_flat_sample(
    fig: Figure,
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    input_: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    vname: str,
    clevels: float,
    datashader: bool = False,
    precip_and_related_fields: list | None = None,
    cmap: Colormap | None = None,
    error_cmap: Colormap | None = None,
) -> None:
    """Plot a "flat" 1D sample.

    Data on non-rectangular (reduced Gaussian) grids.

    Parameters
    ----------
    fig : Figure
        Figure object handle
    ax : matplotlib.axes
        Axis object handle
    lon : np.ndarray
        longitude coordinates array, shape (lon,)
    lat : np.ndarray
        latitude coordinates array, shape (lat,)
    input_ : np.ndarray
        Input data of shape (lat*lon,)
    truth : np.ndarray
        Expected data of shape (lat*lon,)
    pred : np.ndarray
        Predicted data of shape (lat*lon,)
    vname : str
        Variable name
    clevels : float
        Accumulation levels used for precipitation related plots
    datashader: bool, optional
        Datashader plott, by default True
    precip_and_related_fields : list, optional
        List of precipitation-like variables, by default []
    cmap : Colormap, optional
        Colormap for the plot
    error_cmap : Colormap, optional
        Colormap for the error plot

    Returns
    -------
    None
    """
    precip_and_related_fields = precip_and_related_fields or []
    if vname in precip_and_related_fields:
        # converting to mm from m
        truth *= 1000.0
        pred *= 1000.0
        if sum(input_) != 0:
            input_ *= 1000.0
    data = [None for _ in range(6)]
    # truth, prediction and prediction error always plotted
    data[1:4] = [truth, pred, truth - pred]
    # default titles for 6 plots
    titles = [
        f"{vname} input",
        f"{vname} target",
        f"{vname} pred",
        f"{vname} pred err",
        f"{vname} increment [pred - input]",
        f"{vname} persist err",
    ]
    # colormaps
    cmaps = [cmap] * 3 + [error_cmap] * 3
    # normalizations for significant colormaps
    norms = [None for _ in range(6)]
    norms[3:6] = [TwoSlopeNorm(vcenter=0.0)] * 3  # center the error colormaps at 0

    if vname in precip_and_related_fields:
        # Defining the actual precipitation accumulation levels in mm
        cummulation_lvls = clevels
        norm = BoundaryNorm(cummulation_lvls, len(cummulation_lvls) + 1)

        norms[1] = norm
        norms[2] = norm

    else:
        combined_data = np.concatenate((input_, truth, pred))
        # For 'errors', only persistence and increments need identical colorbar-limits

        norm = Normalize(vmin=np.nanmin(combined_data), vmax=np.nanmax(combined_data))

        norms[1] = norm
        norms[2] = norm

    if np.nansum(input_) != 0:
        # prognostic fields: plot input and increment as well
        data[0] = input_
        data[4] = pred - input_
        data[5] = truth - input_
        combined_error = np.concatenate(((pred - input_), (truth - input_)))
        # ensure vcenter is between minimum and maximum error
        norm_error = TwoSlopeNorm(
            vmin=min(-0.00001, np.nanmin(combined_error)),
            vcenter=0.0,
            vmax=max(0.00001, np.nanmax(combined_error)),
        )
        norms[0] = norm
        norms[4] = norm_error
        norms[5] = norm_error

    else:
        # diagnostic fields: omit input and increment plots
        ax[0].axis("off")
        ax[4].axis("off")
        ax[5].axis("off")

    for ii in range(6):
        if data[ii] is not None:
            single_plot(
                fig,
                ax[ii],
                lon,
                lat,
                data[ii],
                cmap=cmaps[ii],
                norm=norms[ii],
                title=titles[ii],
                datashader=datashader,
                transform=None,
            )


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
    try:
        import cartopy.crs as ccrs

    except ModuleNotFoundError as e:
        error_msg = "Module cartopy not found. Install with optional-dependencies.plotting."
        raise ModuleNotFoundError(error_msg) from e

    lat_min, lon_min = latlon.min(axis=0)
    lat_max, lon_max = latlon.max(axis=0)

    central_latitude = (lat_min + lat_max) / 2
    central_longitude = (lon_min + lon_max) / 2

    lat_span = lat_max - lat_min
    std_parallel_1 = central_latitude - lat_span * 0.25
    std_parallel_2 = central_latitude + lat_span * 0.25

    return ccrs.LambertConformal(
        central_latitude=central_latitude,
        central_longitude=central_longitude,
        standard_parallels=[std_parallel_1, std_parallel_2],
    )


def plot_predicted_multilevel_flat_recon(
    parameters: dict[str, int],
    n_plots_per_sample: int,
    latlons: np.ndarray,
    clevels: float,
    sample: np.ndarray,
    reconstruction: np.ndarray,
    difference: np.ndarray,
    datashader: bool = True,
    precip_and_related_fields: list | None = None,
    colormaps: dict[str, Colormap] | None = None,
) -> Figure:
    """Plot input, reconstruction, and error maps for multiple flat 2D fields.

    Parameters
    ----------
    parameters : dict[str, int]
        Dictionary of variable names and indices
    n_plots_per_sample : int
        Number of plots per variable (typically 3: input, reconstruction, error)
    latlons : np.ndarray
        Lat/lon coordinates array of shape (N, 2)
    clevels : float
        Accumulation levels for precipitation variables
    sample : np.ndarray
        Ground truth input data, shape (N, nvar*level)
    reconstruction : np.ndarray
        Reconstructed prediction data, shape (N, nvar*level)
    difference : np.ndarray
        Error or difference array, same shape as above
    datashader : bool, optional
        Whether to use scatter plots for irregular grids
    precip_and_related_fields : list[str], optional
        List of variable names to rescale (e.g., precipitation fields)
    colormaps : dict[str, Colormap], optional
        Optional dictionary of colormaps: can include 'default', 'error', and per-variable maps

    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    output_vars = [(i, vname) for i, (vname, output_only) in (parameters.items()) if output_only]
    n_vars = len(output_vars)
    proj = lambert_conformal_from_latlon_points(latlons)
    datashader = False
    fig, ax = plt.subplots(
        n_vars,
        n_plots_per_sample,
        figsize=(n_plots_per_sample * 4, n_vars * 3),
        layout="constrained",
        subplot_kw={"projection": proj},
    )

    if n_vars == 1:
        ax = np.expand_dims(ax, axis=0)

    pc_lat, pc_lon = latlons[:, 0], latlons[:, 1]
    colormaps = colormaps or {}

    for row_idx, (var_idx, var_name) in enumerate(output_vars):
        sample_t = sample[..., var_idx].squeeze()
        reconstruction_t = reconstruction[..., var_idx].squeeze()
        difference_t = difference[..., var_idx].squeeze()

        # Colormaps
        cmap = colormaps.get("default", cm.get_cmap("viridis"))
        error_cmap = colormaps.get("error", cm.get_cmap("magma"))
        for key, cmap_obj in colormaps.items():
            if key not in ["default", "error"] and hasattr(cmap_obj, "variables") and var_name in cmap_obj.variables:
                cmap = cmap_obj.get_cmap()

        plot_flat_recon(
            fig,
            ax[row_idx],
            pc_lon,
            pc_lat,
            sample_t,
            reconstruction_t,
            difference_t,
            var_name,
            clevels,
            datashader,
            precip_and_related_fields,
            cmap=cmap,
            error_cmap=error_cmap,
        )

    return fig


def plot_flat_recon(
    fig: Figure,
    ax: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    sample: np.ndarray,
    reconstruction: np.ndarray,
    difference: np.ndarray,
    vname: str,
    clevels: float,
    datashader: bool = False,
    precip_and_related_fields: list | None = None,
    cmap: Colormap | None = None,
    error_cmap: Colormap | None = None,
) -> None:
    """Plot one variable's input, reconstruction, and error map on a flat lat/lon grid.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure object
    ax : np.ndarray
        Array of three Axes: [input, reconstruction, error]
    lon : np.ndarray
        Longitudes, shape (N,)
    lat : np.ndarray
        Latitudes, shape (N,)
    sample : np.ndarray
        Ground truth data, shape (N,)
    reconstruction : np.ndarray
        Predicted data, shape (N,)
    difference : np.ndarray
        Difference data (e.g., squared error), shape (N,)
    vname : str
        Variable name
    clevels : float
        Levels for precipitation colorbar
    datashader : bool
        If True, use scatter plots (recommended for unstructured grids)
    precip_and_related_fields : list[str] or None
        Variables to scale (e.g., precipitation from m → mm)
    cmap : Colormap
        Colormap for input and reconstruction
    error_cmap : Colormap
        Colormap for error map
    """
    try:
        import cartopy.crs as ccrs

    except ModuleNotFoundError as e:
        error_msg = "Module cartopy not found. Install with optional-dependencies.plotting."
        raise ModuleNotFoundError(error_msg) from e

    precip_and_related_fields = precip_and_related_fields or []

    if vname in precip_and_related_fields:
        scale_factor = 1000.0
        sample *= scale_factor
        reconstruction *= scale_factor
        difference *= scale_factor

    data = [sample, reconstruction, difference]
    titles = [f"{vname} input", f"{vname} reconstruction", f"{vname} error map"]
    cmaps = [cmap, cmap, error_cmap]
    norms = [None, None, None]

    # Normalize for input and reconstruction
    combined_data = np.concatenate([sample, reconstruction])
    if vname in precip_and_related_fields:
        norm = BoundaryNorm(clevels, len(clevels) + 1)
        norms[0] = norm
        norms[1] = norm
    else:
        norm = Normalize(vmin=np.nanmin(combined_data), vmax=np.nanmax(combined_data))
        norms[0] = norm
        norms[1] = norm

    # Clip extreme errors for more readable plots
    clipped_diff = np.clip(difference, 0, np.nanpercentile(difference, 99))

    # Set a fixed linear color normalization
    norms[2] = Normalize(vmin=0.0, vmax=np.nanmax(clipped_diff))

    for i in range(3):
        if data[i] is not None:
            single_plot(
                fig,
                ax[i],
                lon,
                lat,
                data[i],
                cmap=cmaps[i],
                norm=norms[i],
                title=titles[i],
                datashader=datashader,
                transform=ccrs.PlateCarree() if not datashader else None,
            )


def single_plot(
    fig: Figure,
    ax: plt.axes,
    lon: np.array,
    lat: np.array,
    data: np.array,
    cmap: Colormap | None = None,
    norm: str | None = None,
    title: str | None = None,
    datashader: bool = False,
    transform: object | None = None,
) -> None:
    """Plot a single lat-lon map.

    Plotting can be made either using datashader plot or Datashader(bin) plots.
    By default it uses Datashader since it is faster and more efficient.

    Parameters
    ----------
    fig : Figure
        Figure object handle
    ax : matplotlib.axes
        Axis object handle
    lon : np.ndarray
        longitude coordinates array, shape (lon,)
    lat : np.ndarray
        latitude coordinates array, shape (lat,)
    data : np.ndarray
        Data to plot
    cmap : Colormap, optional
        Colormap, if None use "viridis"
    norm : str, optional
        Normalization string from matplotlib, by default None
    title : str, optional
        Title for plot, by default None
    datashader: bool, optional
        Scatter plot, by default False
    transform:
        Projection for the plot, by default None

    Returns
    -------
    None
    """
    if cmap is None:
        cmap = "viridis"
    if not datashader:
        psc = ax.scatter(
            lon,
            lat,
            c=data,
            cmap=cmap,
            s=1,
            alpha=1.0,
            norm=norm,
            rasterized=False,
            transform=transform,
        )

        # Add map features
        try:
            import cartopy.feature as cfeature

        except ModuleNotFoundError as e:
            error_msg = "Module cartopy not found. Install with optional-dependencies.plotting."
            raise ModuleNotFoundError(error_msg) from e

        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), zorder=1, alpha=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=":", zorder=1)
    else:
        df = pd.DataFrame({"val": data, "x": lon, "y": lat})
        # Adjust binning to match the resolution of the data
        lower_limit = 25
        upper_limit = 500
        n_pixels = max(min(int(np.floor(data.shape[0] * 0.004)), upper_limit), lower_limit)
        psc = dsshow(
            df,
            dsh.Point("x", "y"),
            dsh.mean("val"),
            cmap=cmap,
            plot_width=n_pixels,
            plot_height=n_pixels,
            norm=norm,
            aspect="auto",
            ax=ax,
        )
        continents.plot_continents(ax)

    if title is not None:
        ax.set_title(title)

    ax.set_aspect("auto", adjustable=None)
    _hide_axes_ticks(ax)
    fig.colorbar(psc, ax=ax)


def get_scatter_frame(
    ax: plt.Axes,
    data: np.ndarray,
    latlons: np.ndarray,
    cmap: str = "viridis",
    vmin: int | None = None,
    vmax: int | None = None,
) -> [plt.Axes, PathCollection]:
    """Create a scatter plot for a single frame of an animation."""
    pc_lat, pc_lon = equirectangular_projection(latlons)

    scatter_frame = ax.scatter(
        pc_lon,
        pc_lat,
        c=data,
        cmap=cmap,
        s=5,
        alpha=1.0,
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim((-np.pi, np.pi))
    ax.set_ylim((-np.pi / 2, np.pi / 2))
    continents.plot_continents(ax)
    ax.set_aspect("auto", adjustable=None)
    _hide_axes_ticks(ax)
    return ax, scatter_frame


def edge_plot(
    fig: Figure,
    ax: plt.Axes,
    src_coords: np.ndarray,
    dst_coords: np.ndarray,
    data: np.ndarray,
    cmap: str = "coolwarm",
    title: str | None = None,
) -> None:
    """Lat-lon line plot.

    Parameters
    ----------
    fig : Figure
        Figure object handle
    ax : matplotlib.axes
        Axis object handle
    src_coords : np.ndarray of shape (num_edges, 2)
        Source latitudes and longitudes.
    dst_coords : np.ndarray of shape (num_edges, 2)
        Destination latitudes and longitudes.
    data : np.ndarray of shape (num_edges, 1)
        Data to plot
    cmap : str, optional
        Colormap string from matplotlib, by default "viridis".
    title : str, optional
        Title for plot, by default None
    """
    edge_lines = np.stack([src_coords, dst_coords], axis=1)
    lc = LineCollection(edge_lines, cmap=cmap, linewidths=1)
    lc.set_array(data)

    psc = ax.add_collection(lc)

    xmin, xmax = edge_lines[:, 0, 0].min(), edge_lines[:, 0, 0].max()
    ymin, ymax = edge_lines[:, 1, 1].min(), edge_lines[:, 1, 1].max()
    ax.set_xlim((xmin - 0.1, xmax + 0.1))
    ax.set_ylim((ymin - 0.1, ymax + 0.1))

    continents.plot_continents(ax)

    if title is not None:
        ax.set_title(title)

    ax.set_aspect("auto", adjustable=None)
    _hide_axes_ticks(ax)
    fig.colorbar(psc, ax=ax)


def plot_graph_node_features(
    model: nn.Module,
    trainable_tensors: dict[str, Tensor],
    datashader: bool = False,
) -> Figure:
    """Plot trainable graph node features.

    Parameters
    ----------
    model: AneomiModelEncProcDec
        Model object
    trainable_tensors: dict[str, torch.Tensor]
        Node trainable tensors
    datashader: bool, optional
        Scatter plot, by default False

    Returns
    -------
    Figure
        Figure object handle
    """
    nrows = len(trainable_tensors)
    ncols = max(tt.shape[1] for tt in trainable_tensors.values())

    figsize = (ncols * 4, nrows * 3)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, layout=LAYOUT)

    for row, (mesh, trainable_tensor) in enumerate(trainable_tensors.items()):
        latlons = model.node_attributes.get_coordinates(mesh).cpu().numpy()
        node_features = trainable_tensor.cpu().detach().numpy()

        lat, lon = latlons[:, 0], latlons[:, 1]

        for i in range(ncols):
            ax_ = ax[row, i] if ncols > 1 else ax[row]
            single_plot(
                fig,
                ax_,
                lon=lon,
                lat=lat,
                data=node_features[..., i],
                title=f"{mesh} trainable feature #{i + 1}",
                datashader=datashader,
                transform=None,
            )

    return fig


def plot_graph_edge_features(
    model: nn.Module,
    trainable_modules: dict[tuple[str, str], Tensor],
    q_extreme_limit: float = 0.05,
) -> Figure:
    """Plot trainable graph edge features.

    Parameters
    ----------
    model: AneomiModelEncProcDec
        Model object
    trainable_modules: dict[tuple[str, str], torch.Tensor]
        Edge trainable tensors.
    q_extreme_limit : float, optional
        Plot top & bottom quantile of edges trainable values, by default 0.05 (5%).

    Returns
    -------
    Figure
        Figure object handle
    """
    nrows = len(trainable_modules)
    ncols = max(tt.trainable.trainable.shape[1] for tt in trainable_modules.values())
    figsize = (ncols * 4, nrows * 3)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, layout=LAYOUT)

    for row, ((src, dst), graph_mapper) in enumerate(trainable_modules.items()):
        src_coords = model.node_attributes.get_coordinates(src).cpu().numpy()
        dst_coords = model.node_attributes.get_coordinates(dst).cpu().numpy()
        edge_index = graph_mapper.edge_index_base.cpu().numpy()
        edge_features = graph_mapper.trainable.trainable.cpu().detach().numpy()

        for i in range(ncols):
            ax_ = ax[row, i] if ncols > 1 else ax[row]
            feature = edge_features[..., i]

            # Get mask of feature values over top and bottom percentiles
            top_perc = np.quantile(feature, 1 - q_extreme_limit)
            bottom_perc = np.quantile(feature, q_extreme_limit)

            mask = (feature >= top_perc) | (feature <= bottom_perc)

            edge_plot(
                fig,
                ax_,
                src_coords[edge_index[0, mask]][:, ::-1],
                dst_coords[edge_index[1, mask]][:, ::-1],
                feature[mask],
                title=f"{src} -> {dst} trainable feature #{i + 1}",
            )

    return fig
