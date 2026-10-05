# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from anemoi.training.diagnostics.evaluation.plotting.settings import LAYOUT
from anemoi.training.diagnostics.evaluation.plotting.settings import argsort_variablename_variablelevel

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from anemoi.training.diagnostics.callbacks.plot import PlottingSettings

LOGGER = logging.getLogger(__name__)


def assign_parameter_groups(
    parameter_names: list[str],
    parameter_groups: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """Map each parameter name to a group label.

    Parameters listed in ``parameter_groups`` take that group's label. The rest
    are grouped by their name prefix (``cris_141`` -> ``cris``); prefix groups
    with a single member are folded into ``"other"``.

    Parameters
    ----------
    parameter_names : list[str]
        Ordered list of parameter (variable) names.
    parameter_groups : dict[str, list[str]] | None, optional
        Explicit grouping of parameter names.

    Returns
    -------
    np.ndarray
        Group label per parameter, in the order of ``parameter_names``.
    """
    parameter_groups = parameter_groups or {}

    def _auto_group(name: str) -> str:
        parts = name.split("_")
        return parts[0] if len(parts) == 1 else name[: -len(parts[-1]) - 1]

    parameters_to_groups = np.array(
        [
            next(
                (group_name for group_name, group_parameters in parameter_groups.items() if name in group_parameters),
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
    return unique_group_list[group_inverse]


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
    if len(parameter_names) <= 15:
        parameters_to_groups = np.array(parameter_names)
        sort_by_parameter_group = np.arange(len(parameter_names), dtype=int)
    else:
        parameters_to_groups = assign_parameter_groups(parameter_names, parameter_groups)
        _, group_inverse = np.unique(parameters_to_groups, return_inverse=True)

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
    loss: np.ndarray,
    *,
    parameter_names: list[str],
    parameter_groups: dict[str, list[str]] | None = None,
    metadata_variables: dict[str, Any] | None = None,
    settings: PlottingSettings | None = None,  # noqa: ARG001
    **_kwargs,
) -> Figure:
    """Default plug-in function for :class:`LossCurvePlot`.

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


def _group_colors(n_groups: int) -> np.ndarray:
    """Return one colour per group, keeping neighbouring groups distinct.

    ``tab20`` stores each hue as a dark/light pair, so neighbouring groups would
    otherwise share a hue. Use all the dark shades first, then the light ones.
    """
    if n_groups <= 10:
        return plt.get_cmap("tab10")(np.arange(n_groups))
    if n_groups > 20:
        LOGGER.warning("More than 20 groups detected, but colormap has only 20 colors.")
    order = np.r_[np.arange(0, 20, 2), np.arange(1, 20, 2)]
    return plt.get_cmap("tab20")(order[np.arange(n_groups) % 20])


_LEGEND_NCOLS = 3
_LEGEND_FONTSIZE = 7
# Characters per wrapped legend line, sized for three columns on a 14-inch figure.
_LEGEND_WRAP = 65


def _variable_legend_labels(
    names: np.ndarray,
    group_index: np.ndarray,
    group_names: np.ndarray,
    group_count: np.ndarray,
) -> tuple[list[str], float]:
    """Build one wrapped legend label per group and the height in inches the legend needs.

    ``names`` must already be in plotting order, so each label lists its
    group's variables in the order of their bars.
    """
    labels = [
        "\n".join(
            textwrap.wrap(
                f"{group} (n={count}): " + ", ".join(names[group_index == g]),
                width=_LEGEND_WRAP,
                break_on_hyphens=False,
            ),
        )
        for g, (group, count) in enumerate(zip(group_names, group_count, strict=True))
    ]
    # The legend fills columns top to bottom, the first columns taking any extra entry.
    line_counts = [label.count("\n") + 1 for label in labels]
    entries_per_column = np.diff(np.linspace(0, len(labels), _LEGEND_NCOLS + 1).round()).astype(int)
    entries_per_column = np.sort(entries_per_column)[::-1]
    column_heights = []
    start = 0
    for n_entries in entries_per_column:
        column = line_counts[start : start + n_entries]
        # 1.2 line spacing per text line plus matplotlib's default 0.5 em label spacing.
        column_heights.append(1.2 * sum(column) + 0.5 * len(column))
        start += n_entries
    return labels, max(column_heights) * _LEGEND_FONTSIZE / 72 + 0.2


def loss_contribution_plot_fn(
    loss: np.ndarray,
    *,
    parameter_names: list[str],
    parameter_groups: dict[str, list[str]] | None = None,
    metadata_variables: dict[str, Any] | None = None,
    metric_name: str | None = None,
    settings: PlottingSettings | None = None,  # noqa: ARG001
    top_n: int = 25,
    variable_legend: bool = True,
    **_kwargs,
) -> Figure:
    """Plug-in function for :class:`LossCurvePlot` showing each variable's share of the loss.

    Draws three panels:

    - the share of the total loss from each parameter group, largest first,
      labelled with the percentage and the number of variables in the group;
    - the ``top_n`` variables by share of the total loss, coloured by group;
    - every variable's loss on a log axis, grouped on shaded bands, so the
      spread inside a group is visible.

    Below them, an optional legend lists each group's variables in the
    left-to-right order of their bars in the last panel.

    The total loss is the mean (or sum) of the per-variable losses, so a
    variable's share is its loss divided by the sum over variables. Non-finite
    losses count as zero towards the shares.

    Parameters
    ----------
    loss : np.ndarray
        Per-variable loss of shape (n_parameters,), in model-output order.
    parameter_names : list[str]
        Variable names, in the same order as ``loss``.
    parameter_groups : dict[str, list[str]] | None, optional
        Explicit grouping, see :func:`assign_parameter_groups`.
    metadata_variables : dict | None, optional
        Variable metadata used to order variables by name and level.
    metric_name : str | None, optional
        Step suffix from the task (e.g. ``"_rstep0"``), used in the title.
    settings : PlottingSettings | None, optional
        Unused, accepted for protocol compatibility.
    top_n : int, optional
        Number of variables shown in the top-contributors panel, by default 25.
    variable_legend : bool, optional
        Whether to list each group's variables below the plots, by default True.

    Returns
    -------
    Figure
        The figure object handle.
    """
    parameter_names = list(parameter_names)
    order = argsort_variablename_variablelevel(parameter_names, metadata_variables=metadata_variables)
    names = np.array(parameter_names)[order]
    loss = np.asarray(loss, dtype=float)[order]

    group_names, group_index = np.unique(
        assign_parameter_groups(list(names), parameter_groups),
        return_inverse=True,
    )
    # Stable sort keeps the name/level order inside each group.
    by_group = np.argsort(group_index, kind="stable")
    names, loss, group_index = names[by_group], loss[by_group], group_index[by_group]

    finite_loss = np.where(np.isfinite(loss), loss, 0.0)
    total = finite_loss.sum()
    share = 100 * finite_loss / total if total > 0 else np.zeros_like(finite_loss)
    group_share = np.bincount(group_index, weights=share, minlength=len(group_names))
    group_count = np.bincount(group_index, minlength=len(group_names))
    colors = _group_colors(len(group_names))

    # Height ratios in inches, so the legend row grows without squeezing the plots.
    height_ratios = [6, 4]
    legend_labels = []
    if variable_legend:
        legend_labels, legend_height = _variable_legend_labels(names, group_index, group_names, group_count)
        height_ratios.append(legend_height)
    fig = plt.figure(figsize=(14, sum(height_ratios)), layout="constrained")
    grid = fig.add_gridspec(len(height_ratios), 2, height_ratios=height_ratios)
    title = "Loss contribution by variable"
    if metric_name:
        title += f" ({metric_name.lstrip('_')})"
    fig.suptitle(f"{title}, total {total:.4g}")

    # Panel 1: share of the total loss per group, largest at the top.
    ax_group = fig.add_subplot(grid[0, 0])
    group_order = np.argsort(group_share)
    ax_group.barh(
        np.arange(len(group_names)),
        group_share[group_order],
        color=colors[group_order],
        edgecolor="white",
    )
    ax_group.set_yticks(np.arange(len(group_names)), group_names[group_order])
    for y, g in enumerate(group_order):
        ax_group.annotate(
            f"{group_share[g]:.1f}% (n={group_count[g]})",
            (group_share[g], y),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax_group.set_xlim(0, max(group_share.max(), 1) * 1.3)
    ax_group.set_xlabel("Share of total loss [%]")
    ax_group.set_title("By group")

    # Panel 2: the variables contributing most to the total loss.
    ax_top = fig.add_subplot(grid[0, 1])
    top = np.argsort(share)[::-1][: min(top_n, share.size)][::-1]
    ax_top.barh(np.arange(top.size), share[top], color=colors[group_index[top]], edgecolor="white")
    ax_top.set_yticks(np.arange(top.size), names[top], fontsize=8)
    for y, v in enumerate(top):
        ax_top.annotate(
            f"{share[v]:.1f}%",
            (share[v], y),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax_top.set_xlim(0, max(share[top].max(), 1) * 1.2)
    ax_top.set_xlabel("Share of total loss [%]")
    ax_top.set_title(f"Top {top.size} variables")

    # Panel 3: every variable, grouped on alternating bands with a gap between groups.
    ax_all = fig.add_subplot(grid[1, :])
    gap = max(1, round(0.01 * names.size))
    x = np.arange(names.size) + gap * group_index
    ax_all.bar(x, np.where(loss > 0, loss, np.nan), width=0.8, color=colors[group_index], log=True)
    centres = []
    for g in range(len(group_names)):
        members = x[group_index == g]
        lo, hi = members.min() - 0.5 - gap / 2, members.max() + 0.5 + gap / 2
        if g % 2 == 0:
            ax_all.axvspan(lo, hi, color="0.93", zorder=0, linewidth=0)
        centres.append((lo + hi) / 2)
    ax_all.set_xticks(centres, group_names, rotation=90, fontsize=8)
    ax_all.set_xlim(x.min() - 0.5 - gap / 2, x.max() + 0.5 + gap / 2)
    ax_all.set_ylabel("Loss")
    ax_all.set_title("All variables")
    ax_all.grid(axis="y", which="major", color="0.85", linewidth=0.5)
    ax_all.set_axisbelow(True)

    for ax in (ax_group, ax_top, ax_all):
        ax.spines[["top", "right"]].set_visible(False)

    # Legend: each group's variables in the order their bars appear in panel 3.
    if legend_labels:
        ax_legend = fig.add_subplot(grid[2, :])
        ax_legend.axis("off")
        ax_legend.legend(
            handles=[mpatches.Patch(color=color) for color in colors],
            labels=legend_labels,
            loc="upper left",
            bbox_to_anchor=(0, 0, 1, 1),
            mode="expand",
            ncols=_LEGEND_NCOLS,
            fontsize=_LEGEND_FONTSIZE,
            frameon=False,
            borderaxespad=0,
        )

    return fig
