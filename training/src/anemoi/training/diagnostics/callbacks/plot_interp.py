# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only

from anemoi.training.diagnostics.callbacks.plot import PlotHistogram
from anemoi.training.diagnostics.callbacks.plot import PlotLoss
from anemoi.training.diagnostics.callbacks.plot import PlotSample
from anemoi.training.diagnostics.callbacks.plot import PlotSpectrum
from anemoi.training.diagnostics.plots import argsort_variablename_variablelevel
from anemoi.training.diagnostics.plots import plot_histogram
from anemoi.training.diagnostics.plots import plot_loss
from anemoi.training.diagnostics.plots import plot_power_spectrum
from anemoi.training.diagnostics.plots import plot_predicted_multilevel_flat_sample
from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


class PlotInterpLoss(PlotLoss):
    """Plots the unsqueezed loss over rollouts."""

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger
        _ = batch_idx

        parameter_names = list(pl_module.data_indices.model.output.name_to_index.keys())
        parameter_positions = list(pl_module.data_indices.model.output.name_to_index.values())
        # reorder parameter_names by position
        self.parameter_names = [parameter_names[i] for i in np.argsort(parameter_positions)]
        self.metadata_variables = pl_module.model.metadata["dataset"].get("variables_metadata")

        # Sort the list using the custom key
        argsort_indices = argsort_variablename_variablelevel(
            self.parameter_names,
            metadata_variables=self.metadata_variables,
        )
        self.parameter_names = [self.parameter_names[i] for i in argsort_indices]
        if not isinstance(self.loss, BaseLoss):
            LOGGER.warning(
                "Loss function must be a subclass of BaseLoss, or provide `squash`.",
                RuntimeWarning,
            )

        interpolator_times = len(self.config.training.explicit_times.target)

        if pl_module.rollout > 1:
            LOGGER.info("Time interpolator plots only currently work for rollout 1")
            return

        for interp in range(interpolator_times):
            y_hat = outputs[1][interp]
            y_true = batch[
                :,
                pl_module.multi_step + interp,
                ...,
                pl_module.data_indices.data.output.full,
            ]
            loss = self.loss(y_hat, y_true, squash=False).detach().cpu().numpy()

            sort_by_parameter_group, colors, xticks, legend_patches = self.sort_and_color_by_parameter_group
            loss = loss[argsort_indices]
            fig = plot_loss(loss[sort_by_parameter_group], colors, xticks, legend_patches)

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"loss_istep{interp:02d}_rank{pl_module.local_rank:01d}",
                exp_log_tag=f"loss_sample_istep{interp:02d}_rank{pl_module.local_rank:01d}",
            )


class PlotInterpSample(PlotSample):
    """Plots a post-processed interpolated sample: input, target and prediction."""

    def process(
        self,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().detach().cpu().numpy())

        target_times = len(pl_module.config.training.explicit_times.target)
        input_tensor = (
            batch[
                :,
                pl_module.multi_step - 1 : pl_module.multi_step + target_times + 1,
                ...,
                pl_module.data_indices.data.output.full,
            ]
            .detach()
            .cpu()
        )
        data = self.post_processors(input_tensor)[self.sample_idx]
        output_tensor = torch.cat(
            tuple(
                self.post_processors(x[:, ...].detach().cpu(), in_place=False)[self.sample_idx : self.sample_idx + 1]
                for x in outputs[1]
            ),
        )
        output_tensor = pl_module.output_mask.apply(output_tensor, dim=2, fill_value=np.nan).numpy()
        data[1:, ...] = pl_module.output_mask.apply(data[1:, ...], dim=2, fill_value=np.nan)
        data = data.numpy()

        return data, output_tensor

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger

        # Build dictionary of indices and parameters to be plotted
        diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: (
                name,
                name not in diagnostics,
            )
            for name in self.parameters
        }

        data, output_tensor = self.process(pl_module, outputs, batch)

        local_rank = pl_module.local_rank
        interpolator_times = len(self.config.training.explicit_times.target)

        if pl_module.rollout > 1:
            LOGGER.info("Time interpolator plots only currently work for rollout 1")
            return

        for interp in range(interpolator_times):
            fig = plot_predicted_multilevel_flat_sample(
                plot_parameters_dict,
                self.per_sample,
                self.latlons,
                self.accumulation_levels_plot,
                data[interp, ...].squeeze(),
                data[interp + 1, ...].squeeze(),
                output_tensor[interp, ...],
                datashader=self.datashader_plotting,
                precip_and_related_fields=self.precip_and_related_fields,
                colormaps=self.colormaps,
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"pred_val_sample_istep{interp+1:02d}_batch{batch_idx:04d}_rank{local_rank:01d}",
                exp_log_tag=f"val_pred_sample_istep{interp+1:02d}_rank{local_rank:01d}",
            )


class PlotInterpSpectrum(PlotSpectrum):
    """Plots TP related metric comparing target and prediction.

    The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.

    - Power Spectrum
    """

    def process(
        self,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().detach().cpu().numpy())

        target_times = len(pl_module.config.training.explicit_times.target)
        input_tensor = (
            batch[
                :,
                pl_module.multi_step - 1 : pl_module.multi_step + target_times + 1,
                ...,
                pl_module.data_indices.data.output.full,
            ]
            .detach()
            .cpu()
        )
        data = self.post_processors(input_tensor)[self.sample_idx]
        output_tensor = torch.cat(
            tuple(
                self.post_processors(x[:, ...].detach().cpu(), in_place=False)[self.sample_idx : self.sample_idx + 1]
                for x in outputs[1]
            ),
        )
        output_tensor = pl_module.output_mask.apply(output_tensor, dim=2, fill_value=np.nan).numpy()
        data[1:, ...] = pl_module.output_mask.apply(data[1:, ...], dim=2, fill_value=np.nan)
        data = data.numpy()

        return data, output_tensor

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger

        local_rank = pl_module.local_rank
        data, output_tensor = self.process(pl_module, outputs, batch)

        interpolator_times = len(self.config.training.explicit_times.target)

        if pl_module.rollout > 1:
            LOGGER.info("Time interpolator plots only currently work for rollout 1")
            return

        for interp in range(interpolator_times):
            # Build dictionary of indices and parameters to be plotted

            diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
            plot_parameters_dict_spectrum = {
                pl_module.data_indices.model.output.name_to_index[name]: (
                    name,
                    name not in diagnostics,
                )
                for name in self.parameters
            }

            fig = plot_power_spectrum(
                plot_parameters_dict_spectrum,
                self.latlons,
                data[interp, ...].squeeze(),
                data[interp + 1, ...].squeeze(),
                output_tensor[interp, ...],
                min_delta=self.min_delta,
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"pred_val_spec_istep_{interp:02d}_batch{batch_idx:04d}_rank{local_rank:01d}",
                exp_log_tag=f"pred_val_spec_istep_{interp:02d}_rank{local_rank:01d}",
            )


class PlotInterpHistogram(PlotHistogram):
    """Plots histograms comparing target and prediction.

    The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.
    """

    def process(
        self,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().detach().cpu().numpy())

        target_times = len(pl_module.config.training.explicit_times.target)
        input_tensor = (
            batch[
                :,
                pl_module.multi_step - 1 : pl_module.multi_step + target_times + 1,
                ...,
                pl_module.data_indices.data.output.full,
            ]
            .detach()
            .cpu()
        )
        data = self.post_processors(input_tensor)[self.sample_idx]
        output_tensor = torch.cat(
            tuple(
                self.post_processors(x[:, ...].detach().cpu(), in_place=False)[self.sample_idx : self.sample_idx + 1]
                for x in outputs[1]
            ),
        )
        output_tensor = pl_module.output_mask.apply(output_tensor, dim=2, fill_value=np.nan).numpy()
        data[1:, ...] = pl_module.output_mask.apply(data[1:, ...], dim=2, fill_value=np.nan)
        data = data.numpy()

        return data, output_tensor

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger

        local_rank = pl_module.local_rank
        data, output_tensor = self.process(pl_module, outputs, batch)

        interpolator_times = len(self.config.training.explicit_times.target)

        if pl_module.rollout > 1:
            LOGGER.info("Time interpolator plots only currently work for rollout 1")
            return

        for interp in range(interpolator_times):
            # Build dictionary of indices and parameters to be plotted
            diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic

            plot_parameters_dict_histogram = {
                pl_module.data_indices.model.output.name_to_index[name]: (
                    name,
                    name not in diagnostics,
                )
                for name in self.parameters
            }

            fig = plot_histogram(
                plot_parameters_dict_histogram,
                data[interp, ...].squeeze(),
                data[interp + 1, ...].squeeze(),
                output_tensor[interp, ...],
                self.precip_and_related_fields,
                self.log_scale,
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"pred_val_histo_istep_{interp:02d}_batch{batch_idx:04d}_rank{local_rank:01d}",
                exp_log_tag=f"pred_val_histo_istep_{interp:02d}_rank{local_rank:01d}",
            )
