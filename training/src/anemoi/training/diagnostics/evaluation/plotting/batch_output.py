# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared contract and adapters for spatial-map plot functions.

Every spatial-map plot function used by :class:`BatchOutputPlot` follows the
same signature. This decouples the *what to draw* from the *how to loop over
samples/datasets/focus areas*, so new plots can be added without writing a new
callback subclass or schema.

Contract
--------
A spatial-map plot function must accept (all as keyword-only where noted)::

    fn(parameters: dict[int, tuple[str, bool]],
       *,
       x: np.ndarray,
       y_true: np.ndarray,
       y_pred: np.ndarray,
       latlons: np.ndarray | None = None,
       auxiliary: np.ndarray | None = None,
       settings: "PlottingSettings" | None = None,
       **plot_specific_kwargs) -> matplotlib.figure.Figure

Bind plot-specific kwargs (e.g. ``per_sample``, ``min_delta``, ``log_scale``,
``colormaps``) in the YAML using Hydra's ``_partial_: true``.
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

    return plot_predicted_multilevel_flat_sample(
        parameters,
        per_sample,
        latlons,
        accumulation_levels_plot,
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
    n_plots_per_sample: int = 4,
    **_kwargs: Any,
) -> Figure:
    """Adapter for ``plot_predicted_ensemble`` (PlotEnsSample)."""
    from anemoi.training.diagnostics.evaluation.plotting.ensemble import plot_predicted_ensemble

    return plot_predicted_ensemble(
        parameters=parameters,
        n_plots_per_sample=n_plots_per_sample,
        latlons=latlons,
        clevels=accumulation_levels_plot,
        y_true=np.asarray(y_true).squeeze(),
        y_pred=np.asarray(y_pred).squeeze(),
        datashader=getattr(settings, "datashader", True),
        precip_and_related_fields=getattr(settings, "precip_and_related_fields", None),
        colormaps=getattr(settings, "colormaps", None),
        projection_kind=getattr(settings, "projection_kind", "equirectangular"),
    )
