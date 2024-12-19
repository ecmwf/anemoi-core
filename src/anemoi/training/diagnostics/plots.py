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

import datashader as dsh
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
from datashader.mpl_ext import dsshow
from matplotlib.collections import LineCollection
from matplotlib.collections import PathCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from pyshtools.expand import SHGLQ
from pyshtools.expand import SHExpandGLQ
from scipy.interpolate import griddata

from anemoi.training.diagnostics.maps import Coastlines
from anemoi.training.diagnostics.maps import EquirectangularProjection

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from torch import nn, Tensor

from dataclasses import dataclass

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
    cmap_precip: str,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    datashader: bool = False,
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
    datashader: bool, optional
        Scatter plot, by default False
    precip_and_related_fields : list, optional
        List of precipitation-like variables, by default []

    Returns
    -------
    Figure
        The figure object handle.

    """
    n_plots_x, n_plots_y = len(parameters), n_plots_per_sample

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize, layout=LAYOUT)

    pc_lat, pc_lon = equirectangular_projection(latlons)

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
                datashader,
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
                datashader,
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
    datashader: bool = False,
    precip_and_related_fields: list | None = None,
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
    cmap_precip: str
        Colors used for each accumulation level
    datashader: bool, optional
        Datashader plott, by default True
    precip_and_related_fields : list, optional
        List of precipitation-like variables, by default []

    Returns
    -------
    None
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
        single_plot(
            fig,
            ax[1],
            lon,
            lat,
            truth,
            cmap=precip_colormap,
            norm=norm,
            title=f"{vname} target",
            datashader=datashader,
        )
        single_plot(
            fig,
            ax[2],
            lon,
            lat,
            pred,
            cmap=precip_colormap,
            norm=norm,
            title=f"{vname} pred",
            datashader=datashader,
        )
        single_plot(
            fig,
            ax[3],
            lon,
            lat,
            truth - pred,
            cmap="bwr",
            norm=TwoSlopeNorm(vcenter=0.0),
            title=f"{vname} pred err",
            datashader=datashader,
        )
    elif vname == "mwd":
        cyclic_colormap = "twilight"

        def error_plot_in_degrees(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
            """Calculate error between two arrays in degrees in range [-180, 180]."""
            tmp = (array1 - array2) % 360
            return np.where(tmp > 180, tmp - 360, tmp)

        sample_shape = truth.shape
        pred = np.maximum(np.zeros(sample_shape), np.minimum(360 * np.ones(sample_shape), (pred)))
        single_plot(
            fig,
            ax[1],
            lon=lon,
            lat=lat,
            data=truth,
            cmap=cyclic_colormap,
            title=f"{vname} target",
            datashader=datashader,
        )
        single_plot(
            fig,
            ax[2],
            lon=lon,
            lat=lat,
            data=pred,
            cmap=cyclic_colormap,
            title=f"capped {vname} pred",
            datashader=datashader,
        )
        err_plot = error_plot_in_degrees(truth, pred)
        single_plot(
            fig,
            ax[3],
            lon=lon,
            lat=lat,
            data=err_plot,
            cmap="bwr",
            norm=TwoSlopeNorm(vcenter=0.0),
            title=f"{vname} pred err: {np.nanmean(np.abs(err_plot)):.{4}f} deg.",
            datashader=datashader,
        )
    else:
        combined_data = np.concatenate((input_, truth, pred))
        # For 'errors', only persistence and increments need identical colorbar-limits
        combined_error = np.concatenate(((pred - input_), (truth - input_)))
        norm = Normalize(vmin=np.nanmin(combined_data), vmax=np.nanmax(combined_data))
        single_plot(fig, ax[1], lon, lat, truth, norm=norm, title=f"{vname} target", datashader=datashader)
        single_plot(fig, ax[2], lon, lat, pred, norm=norm, title=f"{vname} pred", datashader=datashader)
        single_plot(
            fig,
            ax[3],
            lon,
            lat,
            truth - pred,
            cmap="bwr",
            norm=TwoSlopeNorm(vcenter=0.0),
            title=f"{vname} pred err",
            datashader=datashader,
        )

    if sum(input_) != 0:
        if vname == "mwd":
            single_plot(
                fig,
                ax[0],
                lon=lon,
                lat=lat,
                data=input_,
                cmap=cyclic_colormap,
                title=f"{vname} input",
                datashader=datashader,
            )
            err_plot = error_plot_in_degrees(pred, input_)
            single_plot(
                fig,
                ax[4],
                lon=lon,
                lat=lat,
                data=err_plot,
                cmap="bwr",
                norm=TwoSlopeNorm(vcenter=0.0),
                title=f"{vname} increment [pred - input] % 360",
                datashader=datashader,
            )
            err_plot = error_plot_in_degrees(truth, input_)
            single_plot(
                fig,
                ax[5],
                lon=lon,
                lat=lat,
                data=err_plot,
                cmap="bwr",
                norm=TwoSlopeNorm(vcenter=0.0),
                title=f"{vname} persist err: {np.nanmean(np.abs(err_plot)):.{4}f} deg.",
                datashader=datashader,
            )
        elif vname in precip_and_related_fields:
            # Create a custom colormap for precipitation
            nws_precip_colors = cmap_precip
            precip_colormap = ListedColormap(nws_precip_colors)

            # Defining the actual precipitation accumulation levels in mm
            cummulation_lvls = clevels
            norm = BoundaryNorm(cummulation_lvls, len(cummulation_lvls) + 1)

            # converting to mm from m
            input_ *= 1000.0
            truth *= 1000.0
            pred *= 1000.0
            single_plot(
                fig,
                ax[0],
                lon=lon,
                lat=lat,
                data=input_,
                cmap=precip_colormap,
                title=f"{vname} input",
                datashader=datashader,
            )
            single_plot(
                fig,
                ax[4],
                lon=lon,
                lat=lat,
                data=pred - input_,
                cmap="bwr",
                norm=TwoSlopeNorm(vcenter=0.0),
                title=f"{vname} increment [pred - input]",
                datashader=datashader,
            )
            single_plot(
                fig,
                ax[5],
                lon=lon,
                lat=lat,
                data=truth - input_,
                cmap="bwr",
                norm=TwoSlopeNorm(vcenter=0.0),
                title=f"{vname} persist err",
                datashader=datashader,
            )
        else:
            single_plot(fig, ax[0], lon, lat, input_, norm=norm, title=f"{vname} input", datashader=datashader)
            single_plot(
                fig,
                ax[4],
                lon,
                lat,
                pred - input_,
                cmap="bwr",
                norm=TwoSlopeNorm(vmin=combined_error.min(), vcenter=0.0, vmax=combined_error.max()),
                title=f"{vname} increment [pred - input]",
                datashader=datashader,
            )
            single_plot(
                fig,
                ax[5],
                lon,
                lat,
                truth - input_,
                cmap="bwr",
                norm=TwoSlopeNorm(vmin=combined_error.min(), vcenter=0.0, vmax=combined_error.max()),
                title=f"{vname} persist err",
                datashader=datashader,
            )
    else:
        ax[0].axis("off")
        ax[4].axis("off")
        ax[5].axis("off")


def single_plot(
    fig: Figure,
    ax: plt.axes,
    lon: np.array,
    lat: np.array,
    data: np.array,
    cmap: str = "viridis",
    norm: str | None = None,
    title: str | None = None,
    datashader: bool = False,
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
    cmap : str, optional
        Colormap string from matplotlib, by default "viridis"
    norm : str, optional
        Normalization string from matplotlib, by default None
    title : str, optional
        Title for plot, by default None
    datashader: bool, optional
        Scatter plot, by default False

    Returns
    -------
    None
    """
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
        )
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

    xmin, xmax = max(lon.min(), -np.pi), min(lon.max(), np.pi)
    ymin, ymax = max(lat.min(), -np.pi / 2), min(lat.max(), np.pi / 2)
    ax.set_xlim((xmin - 0.1, xmax + 0.1))
    ax.set_ylim((ymin - 0.1, ymax + 0.1))

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
