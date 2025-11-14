# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import copy
import logging

import time
from contextlib import nullcontext

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib.colors import Colormap
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from anemoi.training.diagnostics.plots import argsort_variablename_variablelevel
from anemoi.training.diagnostics.plots import get_scatter_frame
from anemoi.training.diagnostics.plots import plot_histogram
from anemoi.training.diagnostics.plots import plot_loss
from anemoi.training.diagnostics.plots import plot_power_spectrum
from anemoi.training.diagnostics.plots import plot_predicted_multilevel_flat_sample
from anemoi.training.diagnostics.callbacks.plot import PlotLoss
from anemoi.training.diagnostics.callbacks.plot import LongRolloutPlots
from anemoi.training.diagnostics.callbacks.plot import PlotHistogram
from anemoi.training.diagnostics.callbacks.plot import PlotSample
from anemoi.training.diagnostics.callbacks.plot import PlotSpectrum
from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)



class LongInterpPlots(LongRolloutPlots):
    """Evaluates the model performance over a (longer) rollout window.

    This function allows evaluating the performance of the model over an extended number
    of rollout steps to observe long-term behavior.
    Add the callback to the configuration file as follows:

    Example::

        - _target_:  anemoi.training.diagnostics.callbacks.plot.LongRolloutPlots
            rollout:
            - ${dataloader.validation_rollout}
            video_rollout: ${dataloader.validation_rollout}
            every_n_epochs: 1
            sample_idx: ${diagnostics.plot.sample_idx}
            parameters: ${diagnostics.plot.parameters}

    The selected rollout steps for plots and video need to be lower or equal to dataloader.validation_rollout.
    Increasing dataloader.validation_rollout has no effect on the rollout steps during training.
    It ensures, that enough time steps are available for the plots and video in the validation batches.

    The runtime of creating one animation of one variable for 56 rollout steps is about 1 minute.
    Recommended use for video generation: Fork the run using fork_run_id for 1 additional epochs and enabled videos.

    """

    def __init__(
        self,
        config: OmegaConf,
        rollout: list[int],
        sample_idx: int,
        parameters: list[str],
        video_rollout: int = 0,
        accumulation_levels_plot: list[float] | None = None,
        colormaps: dict[str, Colormap] | None = None,
        per_sample: int = 6,
        every_n_epochs: int = 1,
        animation_interval: int = 400,
    ) -> None:
        """Initialise LongRolloutPlots callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        rollout : list[int]
            Rollout steps to plot at
        sample_idx : int
            Sample to plot
        parameters : list[str]
            Parameters to plot
        video_rollout : int, optional
            Number of rollout steps for video, by default 0 (no video)
        accumulation_levels_plot : list[float] | None
            Accumulation levels to plot, by default None
        colormaps : dict[str, Colormap] | None
            Dictionary of colormaps, by default None
        per_sample : int, optional
            Number of plots per sample, by default 6
        every_n_epochs : int, optional
            Epoch frequency to plot at, by default 1
        animation_interval : int, optional
            Delay between frames in the animation in milliseconds, by default 400
        """
        super().__init__(config)

        self.every_n_epochs = every_n_epochs

        self.rollout = rollout
        self.video_rollout = video_rollout
        self.max_rollout = 0
        if self.rollout:
            self.max_rollout = max(self.rollout)
        else:
            self.rollout = []
        if self.video_rollout:
            self.max_rollout = max(self.max_rollout, self.video_rollout)

        self.sample_idx = sample_idx
        self.accumulation_levels_plot = accumulation_levels_plot
        self.colormaps = colormaps
        self.per_sample = per_sample
        self.parameters = parameters
        self.animation_interval = animation_interval

        LOGGER.info(
            (
                "Setting up callback for plots with long rollout: rollout for plots = %s, ",
                "rollout for video = %s, frequency = every %d epoch.",
            ),
            self.rollout,
            self.video_rollout,
            every_n_epochs,
        )

        if self.config.diagnostics.plot.asynchronous and self.config.dataloader.read_group_size > 1:
            LOGGER.warning("Asynchronous plotting can result in NCCL timeouts with reader_group_size > 1.")

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        _ = output
        start_time = time.time()
        logger = trainer.logger

        # Initialize required variables for plotting
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: (
                name,
                name not in self.config.data.get("diagnostic", []),
            )
            for name in self.parameters
        }
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().detach().cpu().numpy())

        assert batch.shape[1] >= self.max_rollout + pl_module.multi_step, (
            "Batch length not sufficient for requested validation rollout length! "
            f"Set `dataloader.validation_rollout` to at least {max(self.rollout)}"
        )

        # prepare input tensor for plotting
        # the batch is already preprocessed in-place
        input_tensor_0 = (
            batch[
                :,
                pl_module.multi_step - 1,
                ...,
                pl_module.data_indices.data.output.full,
            ]
            .detach()
            .cpu()
        )
        data_0 = self.post_processors(input_tensor_0)[self.sample_idx]

        if self.video_rollout:
            data_over_time = []
            # collect min and max values for each variable for the colorbar
            vmin, vmax = (np.inf * np.ones(len(plot_parameters_dict)), -np.inf * np.ones(len(plot_parameters_dict)))

        # Plot for each rollout step
        with torch.no_grad():
            for rollout_step, (_, _, y_pred) in enumerate(
                pl_module.rollout_step(
                    batch,
                    rollout=self.max_rollout,
                    validation_mode=True,
                ),
            ):
                # plot only if the current rollout step is in the list of rollout steps
                if (rollout_step + 1) in self.rollout:
                    self._plot_rollout_step(
                        pl_module,
                        plot_parameters_dict,
                        batch,
                        data_0,
                        rollout_step,
                        y_pred,
                        batch_idx,
                        epoch,
                        logger,
                    )

                if self.video_rollout and rollout_step < self.video_rollout:
                    data_over_time, vmin, vmax = self._store_video_frame_data(
                        data_over_time,
                        y_pred,
                        plot_parameters_dict,
                        vmin,
                        vmax,
                    )

            # Generate and save video rollout animation if enabled
            if self.video_rollout:
                self._generate_video_rollout(
                    data_0,
                    data_over_time,
                    plot_parameters_dict,
                    vmin,
                    vmax,
                    self.video_rollout,
                    batch_idx,
                    epoch,
                    logger,
                    animation_interval=self.animation_interval,
                )

        LOGGER.info("Time taken to plot/animate samples for longer rollout: %d seconds", int(time.time() - start_time))

    @rank_zero_only
    def _plot_rollout_step(
        self,
        pl_module: pl.LightningModule,
        plot_parameters_dict: dict,
        input_batch: torch.Tensor,
        data_0: np.ndarray,
        rollout_step: int,
        y_pred: torch.Tensor,
        batch_idx: int,
        epoch: int,
        logger: pl.loggers.logger.Logger,
    ) -> None:
        """Plot the predicted output, input, true target and error plots for a given rollout step."""
        # prepare true output tensor for plotting
        input_tensor_rollout_step = (
            input_batch[
                :,
                pl_module.multi_step + rollout_step,  # (pl_module.multi_step - 1) + (rollout_step + 1)
                ...,
                pl_module.data_indices.data.output.full,
            ]
            .detach()
            .cpu()
        )
        data_rollout_step = self.post_processors(input_tensor_rollout_step)[self.sample_idx]
        # predicted output tensor
        output_tensor = self.post_processors(y_pred.detach().cpu())[self.sample_idx : self.sample_idx + 1]

        fig = plot_predicted_multilevel_flat_sample(
            plot_parameters_dict,
            self.per_sample,
            self.latlons,
            self.accumulation_levels_plot,
            data_0.squeeze(),
            data_rollout_step.squeeze(),
            output_tensor[0, 0, :, :],  # rolloutstep, first member
            colormaps=self.colormaps,
        )
        self._output_figure(
            logger,
            fig,
            epoch=epoch,
            tag=f"pred_val_sample_rstep{rollout_step + 1:03d}_batch{batch_idx:04d}_rank{pl_module.local_rank:01d}",
            exp_log_tag=f"pred_val_sample_rstep{rollout_step + 1:03d}_rank{pl_module.local_rank:01d}",
        )

    def _store_video_frame_data(
        self,
        data_over_time: list,
        y_pred: torch.Tensor,
        plot_parameters_dict: dict,
        vmin: np.ndarray,
        vmax: np.ndarray,
    ) -> tuple[list, np.ndarray, np.ndarray]:
        """Store the data for each frame of the video."""
        # prepare predicted output tensors for video
        output_tensor = self.post_processors(y_pred.detach().cpu())[self.sample_idx : self.sample_idx + 1]
        data_over_time.append(output_tensor[0, 0, :, np.array(list(plot_parameters_dict.keys()))])
        # update min and max values for each variable for the colorbar
        vmin[:] = np.minimum(vmin, np.nanmin(data_over_time[-1], axis=0))
        vmax[:] = np.maximum(vmax, np.nanmax(data_over_time[-1], axis=0))
        return data_over_time, vmin, vmax

    @rank_zero_only
    def _generate_video_rollout(
        self,
        data_0: np.ndarray,
        data_over_time: list,
        plot_parameters_dict: dict,
        vmin: np.ndarray,
        vmax: np.ndarray,
        rollout_step: int,
        batch_idx: int,
        epoch: int,
        logger: pl.loggers.logger.Logger,
        animation_interval: int = 400,
    ) -> None:
        """Generate the video animation for the rollout."""
        for idx, (variable_idx, (variable_name, _)) in enumerate(plot_parameters_dict.items()):
            # Create the animation and list to store the frames (artists)
            frames = []
            # Prepare the figure
            fig, ax = plt.subplots(figsize=(10, 6), dpi=72)
            cmap = "viridis"

            # Create initial data and colorbar
            ax, scatter_frame = get_scatter_frame(
                ax,
                data_0[0, :, variable_idx],
                self.latlons,
                cmap=cmap,
                vmin=vmin[idx],
                vmax=vmax[idx],
            )
            ax.set_title(f"{variable_name}")
            fig.colorbar(scatter_frame, ax=ax)
            frames.append([scatter_frame])

            # Loop through the data and create the scatter plot for each frame
            for frame_data in data_over_time:
                ax, scatter_frame = get_scatter_frame(
                    ax,
                    frame_data[:, idx],
                    self.latlons,
                    cmap=cmap,
                    vmin=vmin[idx],
                    vmax=vmax[idx],
                )
                frames.append([scatter_frame])  # Each frame contains a list of artists (images)

            # Create the animation using ArtistAnimation
            anim = animation.ArtistAnimation(fig, frames, interval=animation_interval, blit=True)
            self._output_gif(
                logger,
                fig,
                anim,
                epoch=epoch,
                tag=f"pred_val_animation_{variable_name}_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank0",
            )



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

        import ipdb; ipdb.set_trace()
        interpolator_times = len(self.config.training.explicit_times.target)

        if  pl_module.rollout > 1:
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

        if  pl_module.rollout > 1:
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

        if  pl_module.rollout > 1:
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

        if  pl_module.rollout > 1:
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