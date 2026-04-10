# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from anemoi.training.diagnostics.evaluation.plotting.settings import LAYOUT


def plot_histogram(
    parameters: dict[int, tuple[str, bool]],
    x: np.ndarray,
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
    precip_and_related_fields: list | None = None,
    log_scale: bool = False,
) -> Figure:
    """Plots histogram.

    NB: this can be very slow for large data arrays
    call it as infrequently as possible!
    When y_true is None (e.g. autoencoder), only x and y_pred are plotted.

    Parameters
    ----------
    parameters : dict[int, tuple[str, bool]]
        Variable index -> (variable_name, diagnostic_only)
    x : np.ndarray
        Input data of shape (lat*lon, nvar*level)
    y_true : np.ndarray or None
        Expected data of shape (lat*lon, nvar*level). If None, only x and y_pred are plotted.
    y_pred : np.ndarray
        Predicted data of shape (lat*lon, nvar*level)
    precip_and_related_fields : list, optional
        List of precipitation-like variables, by default []
    log_scale : bool, optional
        Plot histograms with a log-scale, by default False

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

    for plot_idx, (variable_idx, (variable_name, diagnostic_only)) in enumerate(parameters.items()):
        xt = (x if x.ndim == 1 else x[..., variable_idx]).reshape(-1) * (0 if diagnostic_only else 1)
        yt = (
            (y_true.reshape(-1) if y_true.ndim == 1 else y_true[..., variable_idx].reshape(-1))
            if y_true is not None
            else None
        )
        yp = (y_pred if y_pred.ndim == 1 else y_pred[..., variable_idx]).reshape(-1)

        if not diagnostic_only:
            yp_xt = yp - xt
            if yt is not None:
                yt_xt = yt - xt
                bin_min = min(np.nanmin(yt_xt), np.nanmin(yp_xt))
                bin_max = max(np.nanmax(yt_xt), np.nanmax(yp_xt))
                hist_ref, bins_ref = np.histogram(
                    yt_xt[~np.isnan(yt_xt)],
                    bins=100,
                    density=True,
                    range=[bin_min, bin_max],
                )
            else:
                bin_min = min(np.nanmin(xt), np.nanmin(yp))
                bin_max = max(np.nanmax(xt), np.nanmax(yp))
                hist_ref, bins_ref = np.histogram(xt[~np.isnan(xt)], bins=100, density=True, range=[bin_min, bin_max])
            hist_yp, bins_yp = np.histogram(
                yp_xt[~np.isnan(yp_xt)] if yt is not None else yp[~np.isnan(yp)],
                bins=100,
                density=True,
                range=[bin_min, bin_max],
            )
        else:
            if yt is not None:
                bin_min = min(np.nanmin(yt), np.nanmin(yp))
                bin_max = max(np.nanmax(yt), np.nanmax(yp))
                hist_ref, bins_ref = np.histogram(yt[~np.isnan(yt)], bins=100, density=True, range=[bin_min, bin_max])
            else:
                bin_min = min(np.nanmin(xt), np.nanmin(yp))
                bin_max = max(np.nanmax(xt), np.nanmax(yp))
                hist_ref, bins_ref = np.histogram(xt[~np.isnan(xt)], bins=100, density=True, range=[bin_min, bin_max])
            hist_yp, bins_yp = np.histogram(yp[~np.isnan(yp)], bins=100, density=True, range=[bin_min, bin_max])

        if variable_name in precip_and_related_fields:
            hist_ref = hist_ref * bins_ref[:-1]
            hist_yp = hist_yp * bins_yp[:-1]

        ax[plot_idx].bar(
            bins_ref[:-1],
            hist_ref,
            width=np.diff(bins_ref),
            color="blue",
            alpha=0.7,
            label="Input" if y_true is None else "Truth (data)",
        )
        ax[plot_idx].bar(bins_yp[:-1], hist_yp, width=np.diff(bins_yp), color="red", alpha=0.7, label="Predicted")
        ax[plot_idx].set_title(variable_name)
        ax[plot_idx].set_xlabel(variable_name)
        ax[plot_idx].set_ylabel("Density")
        if log_scale:
            ax[plot_idx].set_yscale("log")
        ax[plot_idx].legend()
        ax[plot_idx].set_aspect("auto", adjustable=None)

    return fig
