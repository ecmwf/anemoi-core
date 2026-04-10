# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plotters — pure plotting objects with no dependency on training or config.

Each Plotter class accepts already-processed numpy arrays / tensors and returns
matplotlib Figure objects.  They can therefore be reused outside of the training
package (e.g. in anemoi-inference) without any pytorch-lightning dependency.

Hierarchy
---------
BasePlotter
├── GraphFeaturePlotter
├── LossPlotter
└── BaseSpatialPlotter
    ├── SamplePlotter
    ├── SpectrumPlotter
    ├── HistogramPlotter
    └── EnsemblePlotter
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from anemoi.training.diagnostics.evaluation.geospatial.focus_area import build_spatial_mask
from anemoi.training.diagnostics.evaluation.plotting.graph import plot_graph_edge_features
from anemoi.training.diagnostics.evaluation.plotting.graph import plot_graph_node_features
from anemoi.training.diagnostics.evaluation.plotting.histogram import plot_histogram
from anemoi.training.diagnostics.evaluation.plotting.loss import plot_loss
from anemoi.training.diagnostics.evaluation.plotting.sample import plot_predicted_multilevel_flat_sample
from anemoi.training.diagnostics.evaluation.plotting.settings import argsort_variablename_variablelevel
from anemoi.training.diagnostics.evaluation.plotting.settings import init_plot_settings
from anemoi.training.diagnostics.evaluation.plotting.spectrum import plot_power_spectrum

if TYPE_CHECKING:
    from typing import Any

    import torch
    from matplotlib.figure import Figure

    from anemoi.models.layers.graph import NamedNodesAttributes
    from anemoi.training.diagnostics.evaluation.geospatial.focus_area import SpatialMask
    from anemoi.training.utils.custom_colormaps import CustomColormap

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class BasePlotter:
    """Base plotter — initialises matplotlib settings on construction."""

    def __init__(self) -> None:
        init_plot_settings()


class BaseSpatialPlotter(BasePlotter):
    """Plotter base for plots that support optional spatial masking.

    Builds a :class:`SpatialMask` from the ``focus_area`` dict at construction
    time.  Subclass ``_plot()`` methods should call
    ``self.focus_mask.apply(graph_data, latlons, *fields)`` before drawing.

    Parameters
    ----------
    focus_area : dict | None, optional
        Focus-area configuration with optional keys ``mask_attr_name``,
        ``latlon_bbox``, and ``name``.  ``None`` produces a no-op mask.
    """

    def __init__(self, focus_area: dict | None = None) -> None:
        super().__init__()
        self.focus_mask: SpatialMask = build_spatial_mask(
            node_attribute_name=focus_area.get("mask_attr_name") if focus_area is not None else None,
            latlon_bbox=focus_area.get("latlon_bbox") if focus_area is not None else None,
            name=focus_area.get("name") if focus_area is not None else None,
        )


# ---------------------------------------------------------------------------
# Concrete plotters
# ---------------------------------------------------------------------------


class GraphFeaturePlotter(BasePlotter):
    """Plots trainable node and edge features of a graph model.

    Parameters
    ----------
    datashader : bool, optional
        Whether to use datashader for scatter plots, by default False
    q_extreme_limit : float, optional
        Quantile used to select extreme edge values to plot, by default 0.05
    """

    def __init__(self, datashader: bool = False, q_extreme_limit: float = 0.05) -> None:
        super().__init__()
        self.datashader = datashader
        self.q_extreme_limit = q_extreme_limit

    def get_node_trainable_tensors(self, node_attributes: NamedNodesAttributes) -> dict[str, torch.Tensor]:
        """Extract trainable node tensors from node attributes.

        Parameters
        ----------
        node_attributes : NamedNodesAttributes
            Node attributes object.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from node-set name to its trainable parameter tensor.
        """
        return {
            name: tt.trainable for name, tt in node_attributes.trainable_tensors.items() if tt.trainable is not None
        }

    @staticmethod
    def _resolve_edge_provider(provider: Any, dataset_name: str) -> Any:
        """Resolve an edge provider to the specific dataset variant if needed."""
        if provider is None:
            return None
        if isinstance(provider, (dict,)):
            if dataset_name in provider:
                return provider[dataset_name]
            return None
        # Also handle torch.nn.ModuleDict without importing torch at module level
        if type(provider).__name__ == "ModuleDict":
            if dataset_name in provider:
                return provider[dataset_name]
            return None
        return provider

    @staticmethod
    def _has_trainable_edge_params(provider: Any) -> bool:
        """Return True if *provider* holds a trainable parameter."""
        if provider is None:
            return False
        trainable_module = getattr(provider, "trainable", None)
        if trainable_module is None:
            return False
        # Graph providers have TrainableTensor -> .trainable;
        # parameter is nested as .trainable.trainable.
        trainable_parameter = getattr(trainable_module, "trainable", None)
        return trainable_parameter is not None

    def get_edge_trainable_modules(
        self,
        model: Any,
        dataset_name: str,
    ) -> dict[tuple[str, str], Any]:
        """Extract trainable edge modules for a given dataset.

        Parameters
        ----------
        model : torch.nn.Module
            The model carrying encoder/processor/decoder graph providers.
        dataset_name : str
            Dataset name used to select per-dataset providers.

        Returns
        -------
        dict[tuple[str, str], Any]
            Mapping from ``(src_nodes, dst_nodes)`` to graph mapper module,
            for each edge-set that has a trainable parameter.
        """
        trainable_modules = {}
        provider_specs = (
            ("encoder_graph_provider", (dataset_name, model._graph_name_hidden)),
            ("decoder_graph_provider", (model._graph_name_hidden, dataset_name)),
            ("processor_graph_provider", (model._graph_name_hidden, model._graph_name_hidden)),
        )
        for provider_name, edge_key in provider_specs:
            provider = self._resolve_edge_provider(getattr(model, provider_name, None), dataset_name)
            if self._has_trainable_edge_params(provider):
                trainable_modules[edge_key] = provider
        return trainable_modules

    def plot_nodes(
        self,
        node_attributes: NamedNodesAttributes,
        trainable_tensors: dict[str, torch.Tensor],
    ) -> Figure:
        """Plot trainable node features.

        Parameters
        ----------
        node_attributes : NamedNodesAttributes
            Node attributes object carrying coordinates.
        trainable_tensors : dict[str, torch.Tensor]
            Mapping from node-set name to its trainable parameter tensor.

        Returns
        -------
        Figure
            Matplotlib figure with one row per node-set, one column per feature.
        """
        return plot_graph_node_features(node_attributes, trainable_tensors, datashader=self.datashader)

    def plot_edges(
        self,
        node_attributes: NamedNodesAttributes,
        trainable_modules: dict[tuple[str, str], torch.nn.Module],
    ) -> Figure:
        """Plot trainable edge features.

        Parameters
        ----------
        node_attributes : NamedNodesAttributes
            Node attributes object carrying coordinates.
        trainable_modules : dict[tuple[str, str], torch.nn.Module]
            Mapping from (src, dst) node-set pair to graph mapper module.

        Returns
        -------
        Figure
            Matplotlib figure with one row per edge-set, one column per feature.
        """
        return plot_graph_edge_features(node_attributes, trainable_modules, q_extreme_limit=self.q_extreme_limit)


class LossPlotter(BasePlotter):
    """Plots the per-variable unsqueezed loss, grouped and coloured by parameter group.

    Parameters
    ----------
    parameter_groups : dict[str, list[str]] | None, optional
        Explicit grouping of parameter names.  Keys are group labels, values are
        lists of parameter names belonging to that group.  Parameters not listed
        are auto-grouped by their name prefix.
    """

    def __init__(self, parameter_groups: dict[str, list[str]] | None = None) -> None:
        super().__init__()
        self.parameter_groups = parameter_groups or {}

    def sort_and_color_by_parameter_group(
        self,
        parameter_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, dict, list]:
        """Sort parameters by group and prepare bar colours and legend patches.

        Parameters
        ----------
        parameter_names : list[str]
            Ordered list of parameter (variable) names.

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
                            for group_name, group_parameters in self.parameter_groups.items()
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
                    (unique_group_list[tn] if count > 1 or unique_group_list[tn] in self.parameter_groups else "other")
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

    def plot(
        self,
        parameter_names: list[str],
        loss: np.ndarray,
        metadata_variables: dict | None = None,
    ) -> Figure:
        """Plot per-variable loss as a bar chart grouped by parameter group.

        Internally sorts ``parameter_names`` by variable name/level before
        applying the group colouring, so callers can pass position-ordered names.

        Parameters
        ----------
        parameter_names : list[str]
            Position-ordered list of variable names (matches last dim of ``loss``).
        loss : np.ndarray of shape (n_parameters,)
            Per-variable loss values, already reduced to 1-D.
        metadata_variables : dict | None, optional
            Variable metadata used by :func:`argsort_variablename_variablelevel`
            to determine sort order.  Pass ``None`` to use name-only sorting.

        Returns
        -------
        Figure
            Matplotlib figure with a grouped, coloured bar chart.
        """
        argsort_indices = argsort_variablename_variablelevel(
            parameter_names,
            metadata_variables=metadata_variables,
        )
        parameter_names = [parameter_names[i] for i in argsort_indices]
        loss = loss[argsort_indices]

        sort_by_parameter_group, colors, xticks, legend_patches = self.sort_and_color_by_parameter_group(
            parameter_names,
        )
        return plot_loss(loss[sort_by_parameter_group], colors, xticks, legend_patches)


class SamplePlotter(BaseSpatialPlotter):
    """Plots a post-processed forecast sample: input, target and prediction fields.

    Parameters
    ----------
    per_sample : int
        Maximum number of fields shown per sample figure.
    accumulation_levels_plot : list[float]
        Accumulation levels used for precipitation colormaps.
    precip_and_related_fields : list[str] | None, optional
        Names of precipitation-like fields that use a special colormap.
    colormaps : dict[str, CustomColormap] | None, optional
        Variable-specific colormaps keyed by ``"default"``, ``"error"``, or
        variable name group.
    datashader : bool, optional
        Whether to use datashader for scatter rendering, by default False.
    projection_kind : str, optional
        Map projection kind (e.g. ``"equirectangular"``), by default
        ``"equirectangular"``.
    focus_area : dict | None, optional
        Focus-area configuration forwarded to :class:`BaseSpatialPlotter`.
    """

    def __init__(
        self,
        per_sample: int,
        accumulation_levels_plot: list[float],
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict[str, CustomColormap] | None = None,
        datashader: bool = False,
        projection_kind: str = "equirectangular",
        focus_area: dict | None = None,
    ) -> None:
        super().__init__(focus_area=focus_area)
        self.per_sample = per_sample
        self.accumulation_levels_plot = accumulation_levels_plot
        self.precip_and_related_fields = precip_and_related_fields
        self.colormaps = colormaps
        self.datashader = datashader
        self.projection_kind = projection_kind

    def plot(
        self,
        plot_parameters_dict: dict[int, tuple[str, bool]],
        latlons: np.ndarray,
        x: np.ndarray,
        y_true: np.ndarray | None,
        y_pred: np.ndarray,
    ) -> Figure:
        """Plot one forecast sample.

        Parameters
        ----------
        plot_parameters_dict : dict[int, tuple[str, bool]]
            Mapping from variable index to ``(name, is_diagnostic)``.
        latlons : np.ndarray of shape (n_grid, 2)
            Latitude/longitude coordinates in degrees (already masked if needed).
        x : np.ndarray
            Input (analysis) field array.
        y_true : np.ndarray | None
            Target field array, or ``None`` for single-output tasks.
        y_pred : np.ndarray
            Predicted field array.

        Returns
        -------
        Figure
            Matplotlib figure with input, target, prediction and error panels.
        """
        return plot_predicted_multilevel_flat_sample(
            plot_parameters_dict,
            self.per_sample,
            latlons,
            self.accumulation_levels_plot,
            x,
            y_true,
            y_pred,
            datashader=self.datashader,
            precip_and_related_fields=self.precip_and_related_fields,
            colormaps=self.colormaps,
            projection_kind=self.projection_kind,
        )


class SpectrumPlotter(BaseSpatialPlotter):
    """Plots power spectra comparing target and prediction.

    Parameters
    ----------
    min_delta : float | None, optional
        Minimum increment magnitude to include in the plot, by default None.
    focus_area : dict | None, optional
        Focus-area configuration forwarded to :class:`BaseSpatialPlotter`.
    """

    def __init__(self, min_delta: float | None = None, focus_area: dict | None = None) -> None:
        super().__init__(focus_area=focus_area)
        self.min_delta = min_delta

    def plot(
        self,
        plot_parameters_dict: dict[int, tuple[str, bool]],
        latlons: np.ndarray,
        x: np.ndarray,
        y_true: np.ndarray | None,
        y_pred: np.ndarray,
    ) -> Figure:
        """Plot power spectra for a set of variables.

        Parameters
        ----------
        plot_parameters_dict : dict[int, tuple[str, bool]]
            Mapping from variable index to ``(name, is_diagnostic)``.
        latlons : np.ndarray of shape (n_grid, 2)
            Latitude/longitude coordinates in degrees (already masked if needed).
        x : np.ndarray
            Input (analysis) field array.
        y_true : np.ndarray | None
            Target field array, or ``None`` for single-output tasks.
        y_pred : np.ndarray
            Predicted field array.

        Returns
        -------
        Figure
            Matplotlib figure with power spectrum curves.
        """
        return plot_power_spectrum(
            plot_parameters_dict,
            latlons,
            x,
            y_true,
            y_pred,
            min_delta=self.min_delta,
        )


class HistogramPlotter(BaseSpatialPlotter):
    """Plots histograms comparing target and prediction distributions.

    Parameters
    ----------
    precip_and_related_fields : list[str] | None, optional
        Names of precipitation-like fields that use a special histogram method.
    log_scale : bool, optional
        Whether to use a logarithmic y-axis, by default False.
    focus_area : dict | None, optional
        Focus-area configuration forwarded to :class:`BaseSpatialPlotter`.
    """

    def __init__(
        self,
        precip_and_related_fields: list[str] | None = None,
        log_scale: bool = False,
        focus_area: dict | None = None,
    ) -> None:
        super().__init__(focus_area=focus_area)
        self.precip_and_related_fields = precip_and_related_fields
        self.log_scale = log_scale

    def plot(
        self,
        plot_parameters_dict: dict[int, tuple[str, bool]],
        x: np.ndarray,
        y_true: np.ndarray | None,
        y_pred: np.ndarray,
    ) -> Figure:
        """Plot histograms for a set of variables.

        Parameters
        ----------
        plot_parameters_dict : dict[int, tuple[str, bool]]
            Mapping from variable index to ``(name, is_diagnostic)``.
        x : np.ndarray
            Input (analysis) field array.
        y_true : np.ndarray | None
            Target field array, or ``None`` for single-output tasks.
        y_pred : np.ndarray
            Predicted field array.

        Returns
        -------
        Figure
            Matplotlib figure with histogram panels.
        """
        return plot_histogram(
            plot_parameters_dict,
            x,
            y_true,
            y_pred,
            self.precip_and_related_fields,
            self.log_scale,
        )


class EnsemblePlotter(BaseSpatialPlotter):
    """Plots a post-processed ensemble sample: target, ensemble mean, spread and members.

    Parameters
    ----------
    accumulation_levels_plot : list[float]
        Accumulation levels used for precipitation colormaps.
    precip_and_related_fields : list[str] | None, optional
        Names of precipitation-like fields that use a special colormap.
    colormaps : dict[str, CustomColormap] | None, optional
        Variable-specific colormaps keyed by ``"default"``, ``"error"``, or
        variable name group.
    datashader : bool, optional
        Whether to use datashader for scatter rendering, by default True.
    projection_kind : str, optional
        Map projection kind, by default ``"equirectangular"``.
    focus_area : dict | None, optional
        Focus-area configuration forwarded to :class:`BaseSpatialPlotter`.
    """

    def __init__(
        self,
        accumulation_levels_plot: list[float],
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict[str, CustomColormap] | None = None,
        datashader: bool = True,
        projection_kind: str = "equirectangular",
        focus_area: dict | None = None,
    ) -> None:
        super().__init__(focus_area=focus_area)
        self.accumulation_levels_plot = accumulation_levels_plot
        self.precip_and_related_fields = precip_and_related_fields
        self.colormaps = colormaps
        self.datashader = datashader
        self.projection_kind = projection_kind

    def plot(
        self,
        plot_parameters_dict: dict[int, tuple[str, bool]],
        latlons: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Figure:
        """Plot one ensemble forecast sample.

        Parameters
        ----------
        plot_parameters_dict : dict[int, tuple[str, bool]]
            Mapping from variable index to ``(name, is_diagnostic)``.
        latlons : np.ndarray of shape (n_grid, 2)
            Latitude/longitude coordinates in degrees (already masked if needed).
        y_true : np.ndarray
            Target field array.
        y_pred : np.ndarray
            Ensemble prediction array (members x grid x variables).

        Returns
        -------
        Figure
            Matplotlib figure with target, mean, spread and per-member panels.
        """
        from anemoi.training.diagnostics.evaluation.plotting.ensemble import plot_predicted_ensemble

        return plot_predicted_ensemble(
            parameters=plot_parameters_dict,
            n_plots_per_sample=4,
            latlons=latlons,
            clevels=self.accumulation_levels_plot,
            y_true=y_true,
            y_pred=y_pred,
            datashader=self.datashader,
            precip_and_related_fields=self.precip_and_related_fields,
            colormaps=self.colormaps,
            projection_kind=self.projection_kind,
        )
