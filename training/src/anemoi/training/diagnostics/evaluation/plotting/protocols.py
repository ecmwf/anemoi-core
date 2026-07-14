# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Protocols and validation for callback plot functions.

Each callback accepts a ``plot_fn`` plug-in. These Protocols document what
parameters a conforming ``plot_fn`` must accept, and :func:`validate_plot_fn`
enforces this at callback ``__init__`` time — before training starts.

Writing a custom plot function
------------------------------
Implement the required keyword parameters for the target callback and accept
``**kwargs`` for anything the callback may forward that you don't need::

    # For BatchOutputPlot — must accept: parameters, x, y_true, y_pred, latlons
    def my_plot_fn(parameters, *, x, y_true, y_pred, latlons, **kwargs):
        ...

    # For GraphFeaturePlot — must accept: dataset_name, node_attributes,
    #                                      node_trainable_tensors, edge_trainable_modules
    def my_graph_fn(*, dataset_name, node_attributes,
                    node_trainable_tensors, edge_trainable_modules, **kwargs):
        ...

Pass it in the YAML config::

    plot_fn:
      _target_: my_package.my_module.my_plot_fn
      _partial_: true
      my_extra_kwarg: 42
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from typing import Any
from typing import runtime_checkable

from typing_extensions import Protocol

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np
    from matplotlib.figure import Figure


@runtime_checkable
class BatchOutputPlotFn(Protocol):
    """Protocol for plot functions used with :class:`BatchOutputPlot`.

    A conforming callable must accept at minimum these keyword parameters:

    Parameters
    ----------
    parameters : dict
        Mapping of output index → ``(variable_name, is_diagnostic)``
    x : np.ndarray
        Model input tensor ``(multistep, latlon, nvar)``
    y_true : np.ndarray
        Ground-truth tensor ``(rollout, latlon, nvar)``
    y_pred : np.ndarray
        Model prediction tensor ``(rollout, latlon, nvar)``
    latlons : np.ndarray
        Lat/lon coordinates ``(latlon, 2)``
    **kwargs
        Extra keyword arguments forwarded by the callback (e.g. ``settings``,
        ``auxiliary``, plot-specific kwargs from Hydra ``_partial_: true``).

    Returns
    -------
    Figure
        A matplotlib Figure that the callback will log and close.
    """

    def __call__(
        self,
        parameters: dict,
        *,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        latlons: np.ndarray,
        **kwargs: Any,
    ) -> Figure: ...


@runtime_checkable
class LossPlotFn(Protocol):
    """Protocol for plot functions used with :class:`LossCurvePlot`.

    A conforming callable must accept at minimum these keyword parameters:

    Parameters
    ----------
    loss : np.ndarray
        Per-variable loss array of shape ``(n_vars,)`` or ``(rollout, n_vars)``.
    parameter_names : list[str]
        Variable names in output-index order.
    **kwargs
        Extra keyword arguments forwarded by the callback (e.g.
        ``parameter_groups``, ``metadata_variables``, ``settings``).

    Returns
    -------
    Figure
        A matplotlib Figure that the callback will log and close.
    """

    def __call__(
        self,
        loss: np.ndarray,
        *,
        parameter_names: list[str],
        **kwargs: Any,
    ) -> Figure: ...


@runtime_checkable
class GraphPlotFn(Protocol):
    """Protocol for plot functions used with :class:`GraphFeaturePlot`.

    A conforming callable must accept at minimum these keyword parameters:

    Parameters
    ----------
    dataset_name : str
        Name of the dataset whose node features are being plotted.
    node_attributes : NamedNodesAttributes
        Node attribute container from the model.
    node_trainable_tensors : dict[str, Tensor]
        Mapping of node-set name → trainable parameter tensor.
    edge_trainable_modules : dict[tuple[str, str], Any]
        Mapping of ``(src, dst)`` → graph mapper module with trainable params.
    **kwargs
        Extra keyword arguments (e.g. ``q_extreme_limit``, ``settings``).

    Yields
    ------
    tuple[Figure, str]
        ``(figure, tag)`` pairs; the callback logs each figure under ``tag``.
    """

    def __call__(
        self,
        *,
        dataset_name: str,
        node_attributes: Any,
        node_trainable_tensors: dict,
        edge_trainable_modules: dict,
        **kwargs: Any,
    ) -> Generator[tuple[Figure, str], None, None]: ...


# Required keyword parameters for each protocol — checked at validate time.
_REQUIRED: dict[type, frozenset[str]] = {
    BatchOutputPlotFn: frozenset({"parameters", "x", "y_true", "y_pred", "latlons"}),
    LossPlotFn: frozenset({"loss", "parameter_names"}),
    GraphPlotFn: frozenset({"dataset_name", "node_attributes", "node_trainable_tensors", "edge_trainable_modules"}),
}

# Built-in plot function _target_ paths, keyed by protocol.
# Used by the Pydantic schema to offer a Literal union for known functions.
KNOWN_PLOT_FN_TARGETS: dict[type, tuple[str, ...]] = {
    BatchOutputPlotFn: (
        "anemoi.training.diagnostics.evaluation.plotting.batch_output.sample_plot_fn",
        "anemoi.training.diagnostics.evaluation.plotting.batch_output.spectrum_plot_fn",
        "anemoi.training.diagnostics.evaluation.plotting.batch_output.histogram_plot_fn",
        "anemoi.training.diagnostics.evaluation.plotting.batch_output.ensemble_plot_fn",
    ),
    LossPlotFn: ("anemoi.training.diagnostics.evaluation.plotting.loss.loss_plot_fn",),
    GraphPlotFn: ("anemoi.training.diagnostics.evaluation.plotting.graph.graph_plot_fn",),
}


def validate_plot_fn(plot_fn: Any, protocol: type, callback_name: str) -> None:
    """Validate that *plot_fn* has the required parameters for *protocol*.

    Called by callback ``__init__`` methods. Unwraps ``functools.partial``
    (and Hydra partials) before inspecting the signature. Raises
    :exc:`TypeError` with a clear message if required parameters are missing.

    Parameters
    ----------
    plot_fn : callable
        The plot function (or Hydra / functools partial) to validate.
    protocol : type
        The Protocol the callback expects, e.g. :class:`BatchOutputPlotFn`.
    callback_name : str
        Human-readable callback class name, used in the error message.

    Raises
    ------
    TypeError
        If *plot_fn* is missing required parameters for *protocol*.
    """
    required = _REQUIRED.get(protocol, frozenset())
    if not required:
        return

    # Unwrap functools.partial / Hydra partial to the underlying function.
    fn = plot_fn
    while hasattr(fn, "func"):
        fn = fn.func

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return  # uninspectable callable — let it fail at call time

    params = set(sig.parameters)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    # **kwargs covers all requirements.
    if has_var_keyword:
        return

    missing = required - params
    if missing:
        known = KNOWN_PLOT_FN_TARGETS.get(protocol, ())
        raise TypeError(
            f"{callback_name} received a plot_fn ({fn.__qualname__!r}) that is missing "
            f"required parameter(s): {', '.join(sorted(missing))}.\n"
            f"A {protocol.__name__}-compatible function must accept: "
            f"{', '.join(sorted(required))}.\n" + (f"Built-in options: {', '.join(known)}" if known else ""),
        )
