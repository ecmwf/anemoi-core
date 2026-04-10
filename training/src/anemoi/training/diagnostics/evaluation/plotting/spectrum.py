# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.interpolate import griddata

from anemoi.training.diagnostics.evaluation.geospatial.projections import Projection
from anemoi.training.diagnostics.evaluation.plotting.settings import LAYOUT

LOGGER = logging.getLogger(__name__)


def _interpolate_field(
    pc_lon: np.ndarray,
    pc_lat: np.ndarray,
    grid_pc_lon: np.ndarray,
    grid_pc_lat: np.ndarray,
    xt: np.ndarray,
    yp: np.ndarray,
    yt: np.ndarray | None,
    diagnostic_only: bool,
    method: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Interpolate predicted and reference fields."""
    if not diagnostic_only:
        yp_field = yp - xt
        yt_field = (yt - xt) if yt is not None else None
        xt_field = xt if yt is None else None
    else:
        yp_field = yp
        yt_field = yt
        xt_field = xt if yt is None else None

    yp_i = griddata((pc_lon, pc_lat), yp_field, (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)

    yt_i = None
    xt_i = None

    if yt_field is not None:
        yt_i = griddata((pc_lon, pc_lat), yt_field, (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)
    elif xt_field is not None:
        xt_i = griddata((pc_lon, pc_lat), xt_field, (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)

    return yp_i, yt_i, xt_i


def _apply_nan_mask(
    yp_i: np.ndarray,
    yt_i: np.ndarray | None,
    xt_i: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Mask NaNs consistently across fields."""
    ref_i = yt_i if yt_i is not None else xt_i
    if ref_i is None:
        return yp_i, yt_i, xt_i

    mask = np.isnan(ref_i)
    if not mask.any():
        return yp_i, yt_i, xt_i

    yp_i = np.where(mask, 0.0, yp_i)

    if yt_i is not None:
        yt_i = np.where(mask, 0.0, yt_i)
    elif xt_i is not None:
        xt_i = np.where(mask, 0.0, xt_i)

    return yp_i, yt_i, xt_i


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
    try:
        from pyshtools.expand import SHGLQ
        from pyshtools.expand import SHExpandGLQ
    except ImportError as e:
        error_msg = (
            "pyshtools is required to compute spherical harmonic power spectra. "
            "It can be installed with the `plotting` dependency. `pip install anemoi-training[plotting]`.",
        )
        raise ImportError(error_msg) from e

    field = np.array(field)

    lmax = field.shape[0] - 1
    zero_w = SHGLQ(lmax)
    coeffs_field = SHExpandGLQ(field, w=zero_w[1], zero=zero_w[0])

    coeff_amp = coeffs_field[0, :, :] ** 2 + coeffs_field[1, :, :] ** 2

    return np.sum(coeff_amp, axis=0)


def plot_power_spectrum(
    parameters: dict[int, tuple[str, bool]],
    latlons: np.ndarray,
    x: np.ndarray,
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
    min_delta: float | None = None,
) -> Figure:
    """Plots power spectrum.

    NB: this can be very slow for large data arrays
    call it as infrequently as possible!
    When y_true is None (e.g. autoencoder), only x and y_pred are plotted.

    Parameters
    ----------
    parameters : dict[int, tuple[str, bool]]
        Variable index -> (variable_name, diagnostic_only). diagnostic_only True for
        diagnostic variables (plot raw output); False for prognostic (plot increments).
    latlons : np.ndarray
        lat/lon coordinates array, shape (lat*lon, 2)
    x : np.ndarray
        Input data of shape (lat*lon, nvar*level)
    y_true : np.ndarray or None
        Expected data of shape (lat*lon, nvar*level). If None, only x and y_pred are plotted.
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

    pc_lon, pc_lat = Projection.equirectangular().project(latlons)

    delta_lat = abs(np.diff(pc_lat))
    non_zero_delta_lat = delta_lat[delta_lat != 0]
    min_delta_lat = np.min(abs(non_zero_delta_lat))

    if min_delta_lat < min_delta:
        LOGGER.warning(
            "Min. distance between lat/lon points is < specified minimum distance. Defaulting to min_delta=%s.",
            min_delta,
        )
        min_delta_lat = min_delta

    n_pix_lat = int(np.floor(abs(pc_lat.max() - pc_lat.min()) / min_delta_lat))
    n_pix_lon = (n_pix_lat - 1) * 2 + 1
    regular_pc_lon = np.linspace(pc_lon.min(), pc_lon.max(), n_pix_lon)
    regular_pc_lat = np.linspace(pc_lat.min(), pc_lat.max(), n_pix_lat)
    grid_pc_lon, grid_pc_lat = np.meshgrid(regular_pc_lon, regular_pc_lat)

    for plot_idx, (variable_idx, (variable_name, diagnostic_only)) in enumerate(parameters.items()):
        xt = (x if x.ndim == 1 else x[..., variable_idx]).reshape(-1)
        yt = (
            (y_true.reshape(-1) if y_true.ndim == 1 else y_true[..., variable_idx].reshape(-1))
            if y_true is not None
            else None
        )
        yp = (y_pred if y_pred.ndim == 1 else y_pred[..., variable_idx]).reshape(-1)

        nan_flag = np.isnan(yt).any() if yt is not None else np.isnan(xt).any()
        method = "linear" if nan_flag else "cubic"

        yp_i, yt_i, xt_i = _interpolate_field(
            pc_lon,
            pc_lat,
            grid_pc_lon,
            grid_pc_lat,
            xt,
            yp,
            yt,
            diagnostic_only,
            method,
        )

        if nan_flag:
            yp_i, yt_i, xt_i = _apply_nan_mask(yp_i, yt_i, xt_i)

        amplitude_p = np.array(compute_spectra(yp_i))
        if yt is not None:
            amplitude_t = np.array(compute_spectra(yt_i))
            ax[plot_idx].loglog(
                np.arange(1, amplitude_t.shape[0]),
                amplitude_t[1 : (amplitude_t.shape[0])],
                label="Truth (data)",
            )
        else:
            amplitude_x = np.array(compute_spectra(xt_i))
            ax[plot_idx].loglog(
                np.arange(1, amplitude_x.shape[0]),
                amplitude_x[1 : (amplitude_x.shape[0])],
                label="Input",
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
