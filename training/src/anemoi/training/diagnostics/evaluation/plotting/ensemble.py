# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import Colormap
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from anemoi.training.diagnostics.evaluation.geospatial.projections import Projection
from anemoi.training.diagnostics.evaluation.plotting.sample import single_plot
from anemoi.training.diagnostics.evaluation.plotting.settings import _hide_axes_ticks

LOGGER = logging.getLogger(__name__)


def plot_rank_histograms(
    parameters: dict[int, str],
    rh: np.ndarray,
) -> Figure:
    """Plots one rank histogram per target variable.

    Parameters
    ----------
    parameters : Dict[int, str]
        Dictionary of target variables
    rh : np.ndarray
        Rank histogram data of shape (nens, nvar)

    Returns
    -------
    Figure
        The figure object handle.
    """
    fig, ax = plt.subplots(1, len(parameters), figsize=(len(parameters) * 4.5, 4))
    n_ens = rh.shape[0] - 1
    rh = rh.astype(float)

    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    for plot_idx, (_variable_idx, variable_name) in enumerate(parameters.items()):
        rh_ = rh[:, plot_idx]
        ax[plot_idx].bar(np.arange(0, n_ens + 1), rh_ / rh_.sum(), linewidth=1, color="blue", width=0.7)
        ax[plot_idx].hlines(rh_.mean() / rh_.sum(), xmin=-0.5, xmax=n_ens + 0.5, linestyles="--", colors="red")
        ax[plot_idx].set_title(f"{variable_name[0]} ranks")
        _hide_axes_ticks(ax[plot_idx])

    fig.tight_layout()
    return fig


def plot_ensemble_sample(
    fig: Figure,
    axs: list[plt.Axes],
    pc_lon: np.ndarray,
    pc_lat: np.ndarray,
    truth: np.ndarray,
    pred_ens: np.ndarray,
    vname: np.ndarray,
    clevels: float,
    ens_dim: int = 0,
    datashader: bool = True,
    precip_and_related_fields: list | None = None,
    cmap: Colormap | None = None,
    error_cmap: Colormap | None = None,
    transform: object | None = None,
) -> None:
    """Use this when plotting ensembles.

    Each member is defined on "flat" (reduced Gaussian) grids.

    Parameters
    ----------
    fig: figure
        Figure object handle
    axs: list[matplotlib.axes]
        List of axis object handles
    pc_lon : np.ndarray
        Projected Longitude coordinates array
    pc_lat : np.ndarray
        Projected Latitude coordinates array
    truth : np.ndarray
        True values
    pred_ens : np.ndarray
        Ensemble array
    vname : np.ndarray
        Variable name
    clevels : float
        Accumulation levels used for precipitation related plots
    ens_dim : int, optional
        Ensemble dimension, by default 0
    datashader : bool, optional
        Datashader plot, by default True
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
    precip_and_related_fields = precip_and_related_fields if precip_and_related_fields is not None else []
    if vname in precip_and_related_fields:
        truth *= 1000.0
        pred_ens *= 1000.0
        norm = BoundaryNorm(clevels, len(clevels) + 1)
    else:
        combined_data = np.concatenate((truth.flatten(), pred_ens.flatten()))
        norm = Normalize(vmin=np.nanmin(combined_data), vmax=np.nanmax(combined_data))

    if len(pred_ens.shape) == 2:
        nens = pred_ens.shape[ens_dim]
        ens_mean, ens_sd = pred_ens.mean(axis=ens_dim), pred_ens.std(axis=ens_dim)
    else:
        nens = 1
        ens_mean = pred_ens
        ens_sd = np.zeros(pred_ens.shape)

    single_plot(
        fig,
        axs[0],
        pc_lon,
        pc_lat,
        truth,
        cmap=cmap,
        norm=norm,
        title=f"{vname[0]} target",
        datashader=datashader,
        transform=transform,
    )
    single_plot(
        fig,
        axs[1],
        pc_lon,
        pc_lat,
        ens_mean,
        cmap=cmap,
        norm=norm,
        title=f"{vname[0]} pred mean",
        datashader=datashader,
        transform=transform,
    )
    single_plot(
        fig,
        axs[2],
        pc_lon,
        pc_lat,
        ens_mean - truth,
        cmap=error_cmap,
        norm=TwoSlopeNorm(vcenter=0.0),
        title=f"{vname[0]} ens mean err",
        datashader=datashader,
        transform=transform,
    )
    single_plot(
        fig,
        axs[3],
        pc_lon,
        pc_lat,
        ens_sd,
        title=f"{vname[0]} ens sd",
        datashader=datashader,
        transform=transform,
    )

    for i_ens in range(nens):
        single_plot(
            fig,
            axs[i_ens + 4],
            pc_lon,
            pc_lat,
            np.take(pred_ens, i_ens, axis=ens_dim) - ens_mean,
            cmap=error_cmap,
            norm=TwoSlopeNorm(vcenter=0.0),
            title=f"{vname[0]}_{i_ens + 1} - mean",
            datashader=datashader,
            transform=transform,
        )


def plot_predicted_ensemble(
    parameters: dict[int, str],
    n_plots_per_sample: int,
    latlons: np.ndarray,
    clevels: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    datashader: bool = True,
    precip_and_related_fields: list | None = None,
    colormaps: dict[str, Colormap] | None = None,
    projection_kind: str = "equirectangular",
) -> Figure:
    """Plots data for one ensemble member.

    Parameters
    ----------
    parameters : Dict[int, str]
        Dictionary of target variables
    n_plots_per_sample : int
        Number of plots per sample
    latlons : np.ndarray
        Latitudes and longitudes
    clevels : float
        Accumulation levels used for precipitation related plots
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    datashader : bool, optional
        Datashader plot, by default True
    precip_and_related_fields : list, optional
        List of precipitation-like variables, by default None
    colormaps : dict[str, Colormap], optional
        Dictionary of colormaps, by default None
    projection_kind : str, optional
        Map projection kind, by default "equirectangular"

    Returns
    -------
    Figure
        The figure object handle.
    """
    nens = y_pred.shape[0] if len(y_pred.shape) == 3 else 1

    n_plots_per_sample = 4  # target, pred mean, mean error, ens sd
    n_plots_x, n_plots_y = len(parameters), nens + n_plots_per_sample
    LOGGER.debug("n_plots_x = %d, n_plots_y = %d", n_plots_x, n_plots_y)

    plot_kind = "equirectangular" if datashader else projection_kind
    (pc_lon, pc_lat), proj, transform = Projection.for_plot(latlons, plot_kind)

    figsize = (n_plots_y * 4, n_plots_x * 3)
    subplot_kw = {"projection": proj} if proj is not None else {}
    fig, axs = plt.subplots(n_plots_x, n_plots_y, figsize=figsize, subplot_kw=subplot_kw)

    colormaps = colormaps if colormaps is not None else {}
    precip_and_related_fields = precip_and_related_fields if precip_and_related_fields is not None else []

    for plot_idx, (variable_idx, value) in enumerate(parameters.items()):
        variable_name = value[0] if isinstance(value, tuple) else value
        yp = y_pred[..., variable_idx].squeeze()
        yt = y_true[..., variable_idx].squeeze()
        _axs = axs[plot_idx, :] if n_plots_x > 1 else axs

        cmap = colormaps.default.get_cmap() if colormaps.get("default") else cm.get_cmap("viridis")
        error_cmap = colormaps.error.get_cmap() if colormaps.get("error") else cm.get_cmap("bwr")
        for key in colormaps:
            if key not in ["default", "error"] and variable_name in colormaps[key].variables:
                cmap = colormaps[key].get_cmap()
                continue

        plot_ensemble_sample(
            fig=fig,
            axs=_axs,
            pc_lon=pc_lon,
            pc_lat=pc_lat,
            truth=yt,
            pred_ens=yp,
            vname=variable_name,
            clevels=clevels,
            ens_dim=0,
            datashader=datashader,
            precip_and_related_fields=precip_and_related_fields,
            cmap=cmap,
            error_cmap=error_cmap,
            transform=transform,
        )

    return fig
