# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any
from typing import Protocol

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from anemoi.training.diagnostics.evaluation.plotting.settings import LAYOUT
from anemoi.training.diagnostics.evaluation.plotting.settings import argsort_variablename_variablelevel

LOGGER = logging.getLogger(__name__)


class LossPlotFn(Protocol):
    """Typing Protocol for :class:`PlotLoss`-compatible plot functions.

    Implementations receive the raw per-variable loss array in the model's
    output order plus the naming/grouping context, and return a single
    :class:`~matplotlib.figure.Figure`.

    The callback additionally forwards per-rollout-step context
    (``step_index``, ``metric_name``, ``task_kwargs``) so that functions such
    as a per-step annotated bar chart can title/label figures without having
    to hard-code the rollout schedule. Implementations are free to ignore any
    of these — the recommended shape is ``**kwargs`` catch-all — but the
    parameters are declared here so that IDE/mypy signatures document what is
    passed through. Any additional keyword arguments bound via the
    ``plot_fn:`` YAML block (``_partial_: true``) are also forwarded through
    ``**kwargs``.
    """

    def __call__(
        self,
        loss: np.ndarray,
        *,
        parameter_names: list[str],
        parameter_groups: dict[str, list[str]] | None = None,
        metadata_variables: dict | None = None,
        step_index: int | None = None,
        metric_name: str | None = None,
        task_kwargs: dict | None = None,
        settings: Any = None,
        **kwargs: Any,
    ) -> Figure: ...


def sort_and_color_by_parameter_group(
    parameter_names: list[str],
    parameter_groups: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, list]:
    """Sort parameters by group and prepare bar colours and legend patches.

    Parameters
    ----------
    parameter_names : list[str]
        Ordered list of parameter (variable) names.
    parameter_groups : dict[str, list[str]] | None, optional
        Explicit grouping of parameter names. Keys are group labels, values
        are lists of parameter names belonging to that group. Parameters not
        listed are auto-grouped by their name prefix.

    Returns
    -------
    tuple
        sort_by_parameter_group : np.ndarray of int
            Index permutation that sorts ``parameter_names`` into group order.
        bar_colors : np.ndarray
            Per-parameter colour array (same length as ``parameter_names``).
        xticks : dict
            Mapping from group label to its x-tick position.
        legend_patches : list[mpatches.Patch]
            Coloured legend patches, one per group.
    """
    parameter_groups = parameter_groups or {}

    def _auto_group(name: str) -> str:
        parts = name.split("_")
        return parts[0] if len(parts) == 1 else name[: -len(parts[-1]) - 1]

    if len(parameter_names) <= 15:
        parameters_to_groups = np.array(parameter_names)
        sort_by_parameter_group = np.arange(len(parameter_names), dtype=int)
    else:
        parameters_to_groups = np.array(
            [
                next(
                    (
                        group_name
                        for group_name, group_parameters in parameter_groups.items()
                        if name in group_parameters
                    ),
                    _auto_group(name),
                )
                for name in parameter_names
            ],
        )

        unique_group_list, group_inverse, group_counts = np.unique(
            parameters_to_groups,
            return_inverse=True,
            return_counts=True,
        )

        unique_group_list = np.array(
            [
                (unique_group_list[tn] if count > 1 or unique_group_list[tn] in parameter_groups else "other")
                for tn, count in enumerate(group_counts)
            ],
        )
        parameters_to_groups = unique_group_list[group_inverse]
        unique_group_list, group_inverse = np.unique(parameters_to_groups, return_inverse=True)

        sort_by_parameter_group = np.argsort(group_inverse, kind="stable")

    sorted_parameter_names = np.array(parameter_names)[sort_by_parameter_group]
    parameters_to_groups = parameters_to_groups[sort_by_parameter_group]
    unique_group_list, group_inverse, group_counts = np.unique(
        parameters_to_groups,
        return_inverse=True,
        return_counts=True,
    )

    cmap = "tab10" if len(unique_group_list) <= 10 else "tab20"
    if len(unique_group_list) > 20:
        LOGGER.warning("More than 20 groups detected, but colormap has only 20 colors.")

    bar_color_per_group = (
        np.tile("k", len(group_counts))
        if not np.any(group_counts - 1)
        else plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_group_list)))
    )

    x_tick_positions = np.cumsum(group_counts) - group_counts / 2 - 0.5
    xticks = dict(zip(unique_group_list, x_tick_positions, strict=False))

    legend_patches = []
    for group_idx, group in enumerate(unique_group_list):
        text_label = f"{group}: "
        string_length = len(text_label)
        for ii in np.where(group_inverse == group_idx)[0]:
            text_label += sorted_parameter_names[ii] + ", "
            string_length += len(sorted_parameter_names[ii]) + 2
            if string_length > 50:
                text_label += "\n"
                string_length = 0
        legend_patches.append(mpatches.Patch(color=bar_color_per_group[group_idx], label=text_label[:-2]))

    return (
        sort_by_parameter_group,
        bar_color_per_group[group_inverse],
        xticks,
        legend_patches,
    )


def plot_loss(
    x: np.ndarray,
    colors: np.ndarray,
    xticks: dict[str, int] | None = None,
    legend_patches: list | None = None,
) -> Figure:
    """Plots per-variable loss as a grouped, coloured bar chart.

    Parameters
    ----------
    x : np.ndarray
        Per-variable loss values of shape (npred,)
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
    figsize = (8, 3) if legend_patches else (4, 3)
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout=LAYOUT)
    ax.bar(np.arange(x.size), x, color=colors, log=1)

    if xticks:
        ax.set_xticks(list(xticks.values()), list(xticks.keys()), rotation=60)
    if legend_patches:
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc="upper left")

    return fig


def loss_plot_fn(
    loss,
    *,
    parameter_names,
    parameter_groups=None,
    metadata_variables=None,
    settings=None,  # noqa: ARG001
    **_kwargs,
) -> Figure:
    """Default plug-in function for :class:`PlotLoss`.

    Applies the standard presentation order (sort by variable + level via
    :func:`argsort_variablename_variablelevel`), then the group-sorting /
    colouring (via :func:`sort_and_color_by_parameter_group`) and finally
    delegates the actual rendering to :func:`plot_loss`. Custom ``plot_fn``
    implementations receive the raw output-index-ordered ``loss`` array plus
    ``parameter_names``, ``parameter_groups`` and ``metadata_variables`` and
    are free to ignore or replace any of these steps.
    """
    parameter_names = list(parameter_names)
    argsort_indices = argsort_variablename_variablelevel(
        parameter_names,
        metadata_variables=metadata_variables,
    )
    parameter_names = [parameter_names[i] for i in argsort_indices]
    loss = np.asarray(loss)[argsort_indices]

    sort_by_parameter_group, colors, xticks, legend_patches = sort_and_color_by_parameter_group(
        parameter_names,
        parameter_groups or {},
    )
    return plot_loss(loss[sort_by_parameter_group], colors, xticks, legend_patches)
