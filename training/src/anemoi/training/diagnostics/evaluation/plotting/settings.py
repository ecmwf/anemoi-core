# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

LAYOUT = "tight"


def argsort_variablename_variablelevel(data: list[str], metadata_variables: dict | None = None) -> list[int]:
    """Custom sort key to process the strings.

    Sort parameter names by alpha part, then by numeric part at last
    position (variable level) if available, then by the original string.

    Parameters
    ----------
    data : list[str]
        List of strings to sort.
    metadata_variables : dict, optional
        Dictionary of variable names and indices, by default None

    Returns
    -------
    list[int]
        Sorted indices of the input list.
    """
    extract_variable_group_and_level = ExtractVariableGroupAndLevel(
        {"default": ""},
        metadata_variables,
    )

    def custom_sort_key(index: int) -> tuple:
        s = data[index]
        _, alpha_part, numeric_part = extract_variable_group_and_level.get_group_and_level(s)
        if numeric_part is None:
            numeric_part = float("inf")
        return (alpha_part, numeric_part, s)

    return sorted(range(len(data)), key=custom_sort_key)


def init_plot_settings() -> None:
    """Initialize matplotlib plot settings."""
    small_font_size = 8
    medium_font_size = 10

    mplstyle.use("fast")
    plt.rcParams["path.simplify_threshold"] = 0.9

    plt.rc("font", size=small_font_size)
    plt.rc("axes", titlesize=small_font_size)
    plt.rc("axes", labelsize=medium_font_size)
    plt.rc("xtick", labelsize=small_font_size)
    plt.rc("ytick", labelsize=small_font_size)
    plt.rc("legend", fontsize=small_font_size)
    plt.rc("figure", titlesize=small_font_size)


def _hide_axes_ticks(ax: plt.Axes) -> None:
    """Hide x/y-axis ticks.

    Parameters
    ----------
    ax : matplotlib.axes
        Axes object handle

    """
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
