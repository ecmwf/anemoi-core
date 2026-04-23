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
from pytorch_lightning.utilities import rank_zero_only

from anemoi.training.diagnostics.callbacks.plot import PlotSample as _PlotSample

if TYPE_CHECKING:
    from typing import Any

    import pytorch_lightning as pl
    import torch
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

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        outputs: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
    ) -> None:
        from anemoi.training.diagnostics.plots import plot_predicted_ensemble

        logger = trainer.logger

        for dataset_name in dataset_names:

            # Build dictionary of indices and parameters to be plotted
            diagnostics = (
                []
                if self.config.data.datasets[dataset_name].diagnostic is None
                else self.config.data.datasets[dataset_name].diagnostic
            )
            plot_parameters_dict = {
                pl_module.data_indices[dataset_name].model.output.name_to_index[name]: (name, name in diagnostics)
                for name in self.parameters
            }

            data, output_tensor = self.process(
                pl_module,
                dataset_name,
                outputs,
                batch,
                members=self.plot_members,
            )

            # Apply spatial mask
            latlons, data, output_tensor = self.focus_mask.apply(
                pl_module.model.model._graph_data,
                self.latlons[dataset_name],
                data,
                output_tensor,
            )

            local_rank = pl_module.local_rank
            for _, y_true, y_pred, tag_suffix in pl_module.plot_adapter.iter_plot_samples(data, output_tensor):
                y_true = np.asarray(y_true).squeeze()
                y_pred = np.asarray(y_pred).squeeze()
                fig = plot_predicted_ensemble(
                    parameters=plot_parameters_dict,
                    n_plots_per_sample=4,
                    latlons=latlons,
                    clevels=self.accumulation_levels_plot,
                    y_true=y_true,
                    y_pred=y_pred,
                    datashader=self.datashader_plotting,
                    precip_and_related_fields=self.precip_and_related_fields,
                    colormaps=self.colormaps,
                    projection_kind=self.projection_kind,
                )
                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=(
                        f"pred_val_sample_{dataset_name}_{tag_suffix}_"
                        f"batch{batch_idx:04d}_rank{local_rank:01d}{self.focus_mask.tag}"
                    ),
                    exp_log_tag=(
                        f"pred_val_sample_{dataset_name}_{tag_suffix}_rank{local_rank:01d}{self.focus_mask.tag}"
                    ),
                )
