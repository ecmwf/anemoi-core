# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from anemoi.training.diagnostics.evaluation.plotting.settings import LAYOUT

LOGGER = logging.getLogger(__name__)


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
