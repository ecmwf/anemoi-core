# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Re-export shim: import from the canonical submodules instead of this file."""

from anemoi.training.diagnostics.evaluation.plotting.ensemble import plot_ensemble_sample
from anemoi.training.diagnostics.evaluation.plotting.ensemble import plot_predicted_ensemble
from anemoi.training.diagnostics.evaluation.plotting.ensemble import plot_rank_histograms
from anemoi.training.diagnostics.evaluation.plotting.graph import edge_plot
from anemoi.training.diagnostics.evaluation.plotting.graph import plot_graph_edge_features
from anemoi.training.diagnostics.evaluation.plotting.graph import plot_graph_node_features
from anemoi.training.diagnostics.evaluation.plotting.histogram import plot_histogram
from anemoi.training.diagnostics.evaluation.plotting.loss import plot_loss
from anemoi.training.diagnostics.evaluation.plotting.sample import get_scatter_frame
from anemoi.training.diagnostics.evaluation.plotting.sample import plot_flat_sample
from anemoi.training.diagnostics.evaluation.plotting.sample import plot_predicted_multilevel_flat_sample
from anemoi.training.diagnostics.evaluation.plotting.sample import single_plot
from anemoi.training.diagnostics.evaluation.plotting.settings import LAYOUT
from anemoi.training.diagnostics.evaluation.plotting.settings import _hide_axes_ticks
from anemoi.training.diagnostics.evaluation.plotting.settings import argsort_variablename_variablelevel
from anemoi.training.diagnostics.evaluation.plotting.settings import init_plot_settings
from anemoi.training.diagnostics.evaluation.plotting.spectrum import compute_spectra
from anemoi.training.diagnostics.evaluation.plotting.spectrum import plot_power_spectrum

__all__ = [
    "LAYOUT",
    "_hide_axes_ticks",
    "argsort_variablename_variablelevel",
    "compute_spectra",
    "edge_plot",
    "get_scatter_frame",
    "init_plot_settings",
    "plot_ensemble_sample",
    "plot_flat_sample",
    "plot_graph_edge_features",
    "plot_graph_node_features",
    "plot_histogram",
    "plot_loss",
    "plot_power_spectrum",
    "plot_predicted_ensemble",
    "plot_predicted_multilevel_flat_sample",
    "plot_rank_histograms",
    "single_plot",
]
