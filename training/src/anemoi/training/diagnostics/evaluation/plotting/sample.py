# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datashader as dsh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datashader.mpl_ext import dsshow
from matplotlib.collections import PathCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import Colormap
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from anemoi.training.diagnostics.evaluation.geospatial.maps import map_features
from anemoi.training.diagnostics.evaluation.geospatial.projections import MapProjection
from anemoi.training.diagnostics.evaluation.plotting.settings import LAYOUT
from anemoi.training.diagnostics.evaluation.plotting.settings import _hide_axes_ticks


def _scale_precip_fields(
    vname: str,
    precip_fields: list,
    input_: np.ndarray,
    truth: np.ndarray | None,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Convert precipitation fields from m to mm."""
    if vname not in precip_fields:
        return input_, truth, pred

    if truth is not None:
        truth = truth * 1000.0

    pred = pred * 1000.0

    if np.nansum(input_) != 0:
        input_ = input_ * 1000.0

    return input_, truth, pred


def _scale_auxiliary_precip_field(
    vname: str,
    precip_fields: list,
    auxiliary: np.ndarray | None,
) -> np.ndarray | None:
    """Convert auxiliary precipitation-like fields from m to mm."""
    if auxiliary is not None and vname in precip_fields:
        return auxiliary * 1000.0

    return auxiliary


def _compute_main_norm(
    vname: str,
    precip_fields: list,
    clevels: float,
    input_: np.ndarray,
    truth: np.ndarray | None,
    pred: np.ndarray,
) -> Normalize:
    """Compute normalization for main (non-error) plots."""
    if vname in precip_fields:
        return BoundaryNorm(clevels, len(clevels) + 1)

    combined = np.concatenate((input_, pred)) if truth is None else np.concatenate((input_, truth, pred))

    return Normalize(
        vmin=np.nanmin(combined),
        vmax=np.nanmax(combined),
    )


def single_plot(
    fig: Figure,
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    data: np.ndarray,
    cmap: Colormap | None = None,
    norm: Normalize | None = None,
    title: str | None = None,
    datashader: bool = False,
    transform: object | None = None,
) -> None:
    """Plot a single lat-lon map.

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
    else:
        df = pd.DataFrame({"val": data, "x": lon, "y": lat})
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

    if transform is not None:
        # Clamp to PlateCarree's valid range to avoid NaN from non-equirectangular
        # projections (e.g. Robinson, Mollweide) when the data spans the whole globe.
        x0 = max(lon.min() - 0.1, -180.0)
        x1 = min(lon.max() + 0.1, 180.0)
        y0 = max(lat.min() - 0.1, -90.0)
        y1 = min(lat.max() + 0.1, 90.0)
        ax.set_extent([x0, x1, y0, y1], crs=transform)
    else:
        xmin, xmax = max(lon.min(), -np.pi), min(lon.max(), np.pi)
        ymin, ymax = max(lat.min(), -np.pi / 2), min(lat.max(), np.pi / 2)
        ax.set_xlim((xmin - 0.1, xmax + 0.1))
        ax.set_ylim((ymin - 0.1, ymax + 0.1))

    map_features.plot(ax)

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
    pc_lon, pc_lat = MapProjection.equirectangular().project(latlons)

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

    map_features.plot(ax)

    ax.set_aspect("auto", adjustable=None)
    _hide_axes_ticks(ax)
    return ax, scatter_frame


def _build_flat_sample_data(
    ax: plt.Axes,
    input_: np.ndarray,
    truth: np.ndarray | None,
    pred: np.ndarray,
    auxiliary: np.ndarray | None,
    diagnostic_only: bool,
) -> list[np.ndarray | None]:
    """Build flat sample panels before normalization and plotting."""
    n_panels = 7 if auxiliary is not None else 6
    data = [None for _ in range(n_panels)]

    if truth is not None:
        data[1:4] = [truth, pred, truth - pred]
        if not diagnostic_only:
            data[5] = truth - input_
    else:
        data[2] = pred
        if not diagnostic_only:
            data[4] = pred - input_
        ax[1].axis("off")
        ax[3].axis("off")
        ax[5].axis("off")

    if auxiliary is not None:
        data[6] = auxiliary if diagnostic_only else auxiliary - input_

    return data


def plot_flat_sample(
    fig: Figure,
    ax: plt.Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    input_: np.ndarray,
    truth: np.ndarray | None,
    pred: np.ndarray,
    vname: str,
    clevels: float,
    datashader: bool = False,
    precip_and_related_fields: list | None = None,
    cmap: Colormap | None = None,
    error_cmap: Colormap | None = None,
    transform: object | None = None,
    prediction_label: str = "pred",
    auxiliary: np.ndarray | None = None,
    auxiliary_label: str = "corrupted targets",
    diagnostic_only: bool = False,
) -> None:
    """Plot a "flat" 1D sample.

    Data on non-rectangular (reduced Gaussian) grids.
    When truth is None (e.g. autoencoder), only input, pred and increment are plotted.

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
    truth : np.ndarray or None
        Expected data of shape (lat*lon,). If None, only input and pred (and pred-input) are plotted.
    pred : np.ndarray
        Predicted data of shape (lat*lon,)
    vname : str
        Variable name
    clevels : float
        Accumulation levels used for precipitation related plots
    datashader: bool, optional
        Datashader plot, by default True
    precip_and_related_fields : list, optional
        List of precipitation-like variables, by default []
    cmap : Colormap, optional
        Colormap for the plot
    error_cmap : Colormap, optional
        Colormap for the error plot
    prediction_label : str, optional
        Label used for the prediction panel titles, by default "pred"
    auxiliary : np.ndarray or None, optional
        Optional extra field (e.g. corrupted targets) plotted as a 7th panel, by default None
    auxiliary_label : str, optional
        Label used for the auxiliary panel title, by default "corrupted targets"
    diagnostic_only : bool, optional
        If True, suppress the input/increment/persistence panels, by default False

    Returns
    -------
    None
    """
    precip_and_related_fields = precip_and_related_fields or []
    input_, truth, pred = _scale_precip_fields(vname, precip_and_related_fields, input_, truth, pred)
    auxiliary = _scale_auxiliary_precip_field(vname, precip_and_related_fields, auxiliary)

    n_panels = 7 if auxiliary is not None else 6
    data = _build_flat_sample_data(ax, input_, truth, pred, auxiliary, diagnostic_only)

    titles = [
        f"{vname} input",
        f"{vname} target",
        f"{vname} {prediction_label}",
        f"{vname} {prediction_label} err",
        f"{vname} increment [{prediction_label} - input]",
        f"{vname} persist err",
    ]
    if auxiliary is not None:
        titles.append(f"{vname} {auxiliary_label}")

    cmaps = [cmap] * 3 + [error_cmap] * 3 + ([error_cmap] if auxiliary is not None else [])
    norms = [None for _ in range(n_panels)]
    norms[3:6] = [TwoSlopeNorm(vcenter=0.0)] * 3

    main_norm = _compute_main_norm(vname, precip_and_related_fields, clevels, input_, truth, pred)
    norms[1] = main_norm
    norms[2] = main_norm

    if not diagnostic_only:
        data[0] = input_
        if data[4] is None:
            data[4] = pred - input_
        combined_error = np.concatenate(((pred - input_), (truth - input_))) if truth is not None else (pred - input_)
        norm_error = TwoSlopeNorm(
            vmin=min(-0.00001, np.nanmin(combined_error)),
            vcenter=0.0,
            vmax=max(0.00001, np.nanmax(combined_error)),
        )
        norms[0] = main_norm
        norms[4] = norm_error
        if truth is not None:
            norms[5] = norm_error
        if auxiliary is not None:
            norms[6] = norm_error
    else:
        ax[0].axis("off")
        ax[4].axis("off")
        ax[5].axis("off")
        if auxiliary is not None:
            auxiliary_delta = data[6]
            norms[6] = TwoSlopeNorm(
                vmin=min(-0.00001, np.nanmin(auxiliary_delta)),
                vcenter=0.0,
                vmax=max(0.00001, np.nanmax(auxiliary_delta)),
            )

    for ii in range(n_panels):
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
                transform=transform,
            )


def plot_predicted_multilevel_flat_sample(
    parameters: dict[int, tuple[str, bool]],
    n_plots_per_sample: int,
    latlons: np.ndarray,
    clevels: float,
    x: np.ndarray,
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
    datashader: bool = False,
    precip_and_related_fields: list | None = None,
    colormaps: dict[str, Colormap] | None = None,
    projection_kind: str = "equirectangular",
    prediction_label: str = "pred",
    auxiliary: np.ndarray | None = None,
    auxiliary_label: str = "corrupted targets",
) -> Figure:
    """Plots data for one multilevel latlon-"flat" sample.

    NB: this can be very slow for large data arrays
    call it as infrequently as possible!

    Parameters
    ----------
    parameters : dict[int, tuple[str, bool]]
        Variable index -> (variable_name, diagnostic_only).
    n_plots_per_sample : int
        Number of plots per sample
    latlons : np.ndarray
        lat/lon coordinates array, shape (lat*lon, 2)
    clevels : float
        Accumulation levels used for precipitation related plots
    x : np.ndarray
        Input data of shape (lat*lon, nvar*level)
    y_true : np.ndarray or None
        Expected data of shape (lat*lon, nvar*level). If None, only x and y_pred are plotted.
    y_pred : np.ndarray
        Predicted data of shape (lat*lon, nvar*level)
    datashader: bool, optional
        Scatter plot, by default False
    precip_and_related_fields : list, optional
        List of precipitation-like variables, by default []
    colormaps : dict[str, Colormap], optional
        Dictionary of colormaps, by default None
    projection_kind : str, optional
        Map projection kind, by default "equirectangular"
    prediction_label : str, optional
        Label used for the prediction panel titles, by default "pred"
    auxiliary : np.ndarray or None, optional
        Optional extra field (e.g. corrupted targets) of shape (lat*lon, nvar*level),
        plotted as an additional panel per variable, by default None
    auxiliary_label : str, optional
        Label used for the auxiliary panel title, by default "corrupted targets"

    Returns
    -------
    Figure
        The figure object handle.

    """
    n_plots_x = len(parameters)
    n_plots_y = max(n_plots_per_sample, 7 if auxiliary is not None else 6)

    plot_kind = "equirectangular" if datashader else projection_kind
    (pc_lon, pc_lat), proj, transform = MapProjection.for_plot(latlons, plot_kind)

    figsize = (n_plots_y * 4, n_plots_x * 3)
    subplot_kw = {"projection": proj} if proj is not None else {}
    fig, axs = plt.subplots(n_plots_x, n_plots_y, figsize=figsize, layout=LAYOUT, subplot_kw=subplot_kw)

    if colormaps is None:
        colormaps = {}

    for plot_idx, (variable_idx, (variable_name, diagnostic_only)) in enumerate(parameters.items()):
        xt = (x if x.ndim == 1 else x[..., variable_idx]).reshape(-1) * (0 if diagnostic_only else 1)
        yt = (
            (y_true.reshape(-1) if y_true.ndim == 1 else y_true[..., variable_idx].reshape(-1))
            if y_true is not None
            else None
        )
        yp = (y_pred if y_pred.ndim == 1 else y_pred[..., variable_idx]).reshape(-1)
        ya = None if auxiliary is None else (auxiliary if auxiliary.ndim == 1 else auxiliary[..., variable_idx])
        ya = None if ya is None else ya.reshape(-1)

        cmap = colormaps["default"].get_cmap() if colormaps.get("default") else plt.colormaps["viridis"]
        error_cmap = colormaps["error"].get_cmap() if colormaps.get("error") else plt.colormaps["bwr"]
        for key in colormaps:
            if key not in ["default", "error"] and variable_name in colormaps[key].variables:
                cmap = colormaps[key].get_cmap()

        ax = axs[plot_idx, :] if n_plots_x > 1 else axs
        plot_flat_sample(
            fig=fig,
            ax=ax,
            lon=pc_lon,
            lat=pc_lat,
            input_=xt,
            truth=yt,
            pred=yp,
            vname=variable_name,
            clevels=clevels,
            datashader=datashader,
            precip_and_related_fields=precip_and_related_fields,
            cmap=cmap,
            error_cmap=error_cmap,
            transform=transform,
            prediction_label=prediction_label,
            auxiliary=ya,
            auxiliary_label=auxiliary_label,
            diagnostic_only=diagnostic_only,
        )
    return fig
