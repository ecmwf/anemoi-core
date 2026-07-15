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
    - ``n_plots_per_sample`` (int, default ``4``): number of fixed panels per
      variable row (target, pred mean, mean error, ens sd).

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
    from anemoi.training.diagnostics.evaluation.plotting.settings import DEFAULT_ACCUMULATION_LEVELS

    levels = accumulation_levels_plot if accumulation_levels_plot is not None else DEFAULT_ACCUMULATION_LEVELS
    return plot_predicted_ensemble(
        parameters=parameters,
        n_plots_per_sample=n_plots_per_sample,
        latlons=latlons,
        clevels=levels,
        y_true=np.asarray(y_true).squeeze(),
        y_pred=np.asarray(y_pred).squeeze(),
        datashader=getattr(settings, "datashader", True),
        precip_and_related_fields=getattr(settings, "precip_and_related_fields", None),
        colormaps=getattr(settings, "colormaps", None),
        projection_kind=getattr(settings, "projection_kind", "equirectangular"),
    )
