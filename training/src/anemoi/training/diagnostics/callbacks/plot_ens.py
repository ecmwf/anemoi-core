# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from anemoi.training.diagnostics.callbacks.plot import PlotSample as _PlotSample

if TYPE_CHECKING:
    from typing import Any

    from matplotlib.figure import Figure
    from omegaconf import DictConfig


LOGGER = logging.getLogger(__name__)


class PlotEnsSample(_PlotSample):
    """Plots a post-processed ensemble sample: input, target and prediction.

    Uses the ensemble-aware plot adapter (via ``pl_module.plot_adapter``) and calls
    ``plot_predicted_ensemble`` to visualise multiple ensemble members side-by-side.
    """

    def __init__(
        self,
        config: DictConfig,
        sample_idx: int,
        parameters: list[str],
        accumulation_levels_plot: list[float],
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict[str] | None = None,
        per_sample: int = 6,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        members: list | None = None,
        focus_area: list[dict] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config,
            sample_idx,
            parameters,
            accumulation_levels_plot,
            precip_and_related_fields,
            colormaps,
            per_sample,
            every_n_batches,
            dataset_names,
            focus_area,
            **kwargs,
        )
        self.plot_members = members

    def _get_process_members(self) -> list | None:
        """Return configured ensemble members (None = all members)."""
        return self.plot_members

    def _make_figure(
        self,
        plot_parameters_dict: dict,
        latlons: np.ndarray,
        x: np.ndarray,  # noqa: ARG002
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Figure:
        """Create an ensemble figure with members, mean, spread and error."""
        from anemoi.training.diagnostics.plots import plot_predicted_ensemble

        return plot_predicted_ensemble(
            parameters=plot_parameters_dict,
            n_plots_per_sample=4,
            latlons=latlons,
            clevels=self.accumulation_levels_plot,
            y_true=np.asarray(y_true).squeeze(),
            y_pred=np.asarray(y_pred).squeeze(),
            datashader=self.datashader_plotting,
            precip_and_related_fields=self.precip_and_related_fields,
            colormaps=self.colormaps,
            projection_kind=self.projection_kind,
        )
