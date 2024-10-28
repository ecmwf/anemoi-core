# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.colors import TwoSlopeNorm
from pyshtools.expand import SHGLQ
from pyshtools.expand import SHExpandGLQ
from scipy.interpolate import griddata

from anemoi.training.diagnostics.maps import Coastlines
from anemoi.training.diagnostics.maps import EquirectangularProjection

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

continents = Coastlines()


@dataclass
class LatLonData:
    latitudes: np.ndarray
    longitudes: np.ndarray
    data: np.ndarray


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
    figsize = (8, 3) if legend_patches else (4, 3)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # histogram plot
    ax.bar(np.arange(x.size), x, color=colors, log=1)

    # add xticks and legend if given
    if xticks:
        ax.set_xticks(list(xticks.values()), list(xticks.keys()), rotation=60)
    if legend_patches:
        # legend outside and to the right of the plot
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    return fig


def plot_power_spectrum(
    parameters: dict[str, int],
    latlons: np.ndarray,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
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

    Returns
    -------
    Figure
        The figure object handle.

    """
    n_plots_x, n_plots_y = len(parameters), 1

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize)

    pc = EquirectangularProjection()
    lat, lon = latlons[:, 0], latlons[:, 1]
    pc_lon, pc_lat = pc(lon, lat)
    pc_lon = np.array(pc_lon)
    pc_lat = np.array(pc_lat)
    # Calculate delta_lon and delta_lat on the projected grid
    delta_lon = abs(np.diff(pc_lon))
    non_zero_delta_lon = delta_lon[delta_lon != 0]
    delta_lat = abs(np.diff(pc_lat))
    non_zero_delta_lat = delta_lat[delta_lat != 0]

    # Define a regular grid for interpolation
    n_pix_lon = int(np.floor(abs(pc_lon.max() - pc_lon.min()) / abs(np.min(non_zero_delta_lon))))  # around 400 for O96
    n_pix_lat = int(np.floor(abs(pc_lat.max() - pc_lat.min()) / abs(np.min(non_zero_delta_lat))))  # around 192 for O96
    regular_pc_lon = np.linspace(pc_lon.min(), pc_lon.max(), n_pix_lon)
    regular_pc_lat = np.linspace(pc_lat.min(), pc_lat.max(), n_pix_lat)
    grid_pc_lon, grid_pc_lat = np.meshgrid(regular_pc_lon, regular_pc_lat)

    for plot_idx, (variable_idx, (variable_name, output_only)) in enumerate(parameters.items()):
        yt = y_true[..., variable_idx].squeeze()
        yp = y_pred[..., variable_idx].squeeze()

        # check for any nan in yt
        nan_flag = np.isnan(yt).any()

        method = "linear" if nan_flag else "cubic"
        if output_only:
            xt = x[..., variable_idx].squeeze()
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
    fig.tight_layout()
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
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize)

    for plot_idx, (variable_idx, (variable_name, output_only)) in enumerate(parameters.items()):
        yt = y_true[..., variable_idx].squeeze()
        yp = y_pred[..., variable_idx].squeeze()
        # postprocessed outputs so we need to handle possible NaNs

        # Calculate the histogram and handle NaNs
        if output_only:
            # histogram of true increment and predicted increment
            xt = x[..., variable_idx].squeeze() * int(output_only)
            yt_xt = yt - xt
            yp_xt = yp - xt
            # enforce the same binning for both histograms
            bin_min = min(np.nanmin(yt_xt), np.nanmin(yp_xt))
            bin_max = max(np.nanmax(yt_xt), np.nanmax(yp_xt))
            hist_yt, bins_yt = np.histogram(yt_xt[~np.isnan(yt_xt)], bins=100, range=[bin_min, bin_max])
            hist_yp, bins_yp = np.histogram(yp_xt[~np.isnan(yp_xt)], bins=100, range=[bin_min, bin_max])
        else:
            # enforce the same binning for both histograms
            bin_min = min(np.nanmin(yt), np.nanmin(yp))
            bin_max = max(np.nanmax(yt), np.nanmax(yp))
            hist_yt, bins_yt = np.histogram(yt[~np.isnan(yt)], bins=100, range=[bin_min, bin_max])
            hist_yp, bins_yp = np.histogram(yp[~np.isnan(yp)], bins=100, range=[bin_min, bin_max])

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

    fig.tight_layout()
    return fig


def plot_predicted_multilevel_flat_sample(
    parameters: dict[str, int],
    n_plots_per_sample: int,
    latlons: np.ndarray,
    clevels: float,
    cmap_precip: str,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    precip_and_related_fields: list | None = None,
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
    cmap_precip: str
        Colors used for each accumulation level
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
    n_plots_x, n_plots_y = len(parameters), n_plots_per_sample

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize)

    pc = EquirectangularProjection()
    lat, lon = latlons[:, 0], latlons[:, 1]
    pc_lon, pc_lat = pc(lon, lat)

    for plot_idx, (variable_idx, (variable_name, output_only)) in enumerate(parameters.items()):
        xt = x[..., variable_idx].squeeze() * int(output_only)
        yt = y_true[..., variable_idx].squeeze()
        yp = y_pred[..., variable_idx].squeeze()
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
                cmap_precip,
                precip_and_related_fields,
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
                cmap_precip,
                precip_and_related_fields,
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
    cmap_precip: str,
    precip_and_related_fields: list | None = None,
) -> None:
    """Plot a "flat" 1D sample.

    Data on non-rectangular (reduced Gaussian) grids.

    Parameters
    ----------
    fig : _type_
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
    cmap_precip: str
        Colors used for each accumulation level
    precip_and_related_fields : list, optional
        List of precipitation-like variables, by default []

    """
    precip_and_related_fields = precip_and_related_fields or []
    if vname in precip_and_related_fields:
        # Create a custom colormap for precipitation
        nws_precip_colors = cmap_precip
        precip_colormap = ListedColormap(nws_precip_colors)

        # Defining the actual precipitation accumulation levels in mm
        cummulation_lvls = clevels
        norm = BoundaryNorm(cummulation_lvls, len(cummulation_lvls) + 1)

        # converting to mm from m
        truth *= 1000.0
        pred *= 1000.0
        scatter_plot(fig, ax[1], lon=lon, lat=lat, data=truth, cmap=precip_colormap, norm=norm, title=f"{vname} target")
        scatter_plot(fig, ax[2], lon=lon, lat=lat, data=pred, cmap=precip_colormap, norm=norm, title=f"{vname} pred")
        scatter_plot(
            fig,
            ax[3],
            lon=lon,
            lat=lat,
            data=truth - pred,
            cmap="bwr",
            norm=TwoSlopeNorm(vcenter=0.0),
            title=f"{vname} pred err",
        )
    elif vname == "mwd":
        cyclic_colormap = "twilight"

        def error_plot_in_degrees(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
            """Calculate error between two arrays in degrees in range [-180, 180]."""
            tmp = (array1 - array2) % 360
            return np.where(tmp > 180, tmp - 360, tmp)

        sample_shape = truth.shape
        pred = np.maximum(np.zeros(sample_shape), np.minimum(360 * np.ones(sample_shape), (pred)))
        scatter_plot(fig, ax[1], lon=lon, lat=lat, data=truth, cmap=cyclic_colormap, title=f"{vname} target")
        scatter_plot(fig, ax[2], lon=lon, lat=lat, data=pred, cmap=cyclic_colormap, title=f"capped {vname} pred")
        err_plot = error_plot_in_degrees(truth, pred)
        scatter_plot(
            fig,
            ax[3],
            lon=lon,
            lat=lat,
            data=err_plot,
            cmap="bwr",
            norm=TwoSlopeNorm(vcenter=0.0),
            title=f"{vname} pred err: {np.nanmean(np.abs(err_plot)):.{4}f} deg.",
        )
    else:
        scatter_plot(fig, ax[1], lon=lon, lat=lat, data=truth, title=f"{vname} target")
        scatter_plot(fig, ax[2], lon=lon, lat=lat, data=pred, title=f"{vname} pred")
        scatter_plot(
            fig,
            ax[3],
            lon=lon,
            lat=lat,
            data=truth - pred,
            cmap="bwr",
            norm=TwoSlopeNorm(vcenter=0.0),
            title=f"{vname} pred err",
        )

    if sum(input_) != 0:
        if vname == "mwd":
            scatter_plot(fig, ax[0], lon=lon, lat=lat, data=input_, cmap=cyclic_colormap, title=f"{vname} input")
            err_plot = error_plot_in_degrees(pred, input_)
            scatter_plot(
                fig,
                ax[4],
                lon=lon,
                lat=lat,
                data=err_plot,
                cmap="bwr",
                norm=TwoSlopeNorm(vcenter=0.0),
                title=f"{vname} increment [pred - input] % 360",
            )
            err_plot = error_plot_in_degrees(truth, input_)
            scatter_plot(
                fig,
                ax[5],
                lon=lon,
                lat=lat,
                data=err_plot,
                cmap="bwr",
                norm=TwoSlopeNorm(vcenter=0.0),
                title=f"{vname} persist err: {np.nanmean(np.abs(err_plot)):.{4}f} deg.",
            )
        else:
            scatter_plot(fig, ax[0], lon=lon, lat=lat, data=input_, title=f"{vname} input")
            scatter_plot(
                fig,
                ax[4],
                lon=lon,
                lat=lat,
                data=pred - input_,
                cmap="bwr",
                norm=TwoSlopeNorm(vcenter=0.0),
                title=f"{vname} increment [pred - input]",
            )
            scatter_plot(
                fig,
                ax[5],
                lon=lon,
                lat=lat,
                data=truth - input_,
                cmap="bwr",
                norm=TwoSlopeNorm(vcenter=0.0),
                title=f"{vname} persist err",
            )
    else:
        ax[0].axis("off")
        ax[4].axis("off")
        ax[5].axis("off")


def scatter_plot(
    fig: Figure,
    ax: plt.Axes,
    *,
    lon: np.array,
    lat: np.array,
    data: np.array,
    cmap: str = "viridis",
    norm: str | None = None,
    title: str | None = None,
) -> None:
    """Lat-lon scatter plot: can work with arbitrary grids.

    Parameters
    ----------
    fig : _type_
        Figure object handle
    ax : matplotlib.axes
        Axis object handle
    lon : np.ndarray
        longitude coordinates array, shape (lon,)
    lat : np.ndarray
        latitude coordinates array, shape (lat,)
    data : _type_
        Data to plot
    cmap : str, optional
        Colormap string from matplotlib, by default "viridis"
    norm : str, optional
        Normalization string from matplotlib, by default None
    title : str, optional
        Title for plot, by default None

    """
    psc = ax.scatter(
        lon,
        lat,
        c=data,
        cmap=cmap,
        s=1,
        alpha=1.0,
        norm=norm,
        rasterized=True,
    )
    ax.set_xlim((-np.pi, np.pi))
    ax.set_ylim((-np.pi / 2, np.pi / 2))

    continents.plot_continents(ax)

    if title is not None:
        ax.set_title(title)

    ax.set_aspect("auto", adjustable=None)
    _hide_axes_ticks(ax)
    fig.colorbar(psc, ax=ax)


def plot_graph_features(
    latlons: np.ndarray,
    features: np.ndarray,
) -> Figure:
    """Plot trainable graph features.

    Parameters
    ----------
    latlons : np.ndarray
        Latitudes and longitudes
    features : np.ndarray
        Trainable Features

    Returns
    -------
    Figure
        Figure object handle

    """
    nplots = features.shape[-1]
    figsize = (nplots * 4, 3)
    fig, ax = plt.subplots(1, nplots, figsize=figsize)

    lat, lon = latlons[:, 0], latlons[:, 1]

    pc = EquirectangularProjection()
    pc_lon, pc_lat = pc(lon, lat)

    for i in range(nplots):
        ax_ = ax[i] if nplots > 1 else ax
        scatter_plot(fig, ax_, lon=pc_lon, lat=pc_lat, data=features[..., i])

    return fig
