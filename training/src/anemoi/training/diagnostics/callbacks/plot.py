# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import asyncio
import copy
import logging
import threading
import time
import traceback
from abc import ABC
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import cached_property
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib.colors import Colormap
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.mapper import GraphEdgeMixin
from anemoi.training.diagnostics.plots import argsort_variablename_variablelevel
from anemoi.training.diagnostics.plots import get_scatter_frame
from anemoi.training.diagnostics.plots import init_plot_settings
from anemoi.training.diagnostics.plots import plot_graph_edge_features
from anemoi.training.diagnostics.plots import plot_graph_node_features
from anemoi.training.diagnostics.plots import plot_histogram
from anemoi.training.diagnostics.plots import plot_loss
from anemoi.training.diagnostics.plots import plot_power_spectrum
from anemoi.training.diagnostics.plots import plot_predicted_multilevel_flat_sample
from anemoi.training.losses.base import BaseLoss
from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class BasePlotCallback(Callback, ABC):
    """Factory for creating a callback that plots data to Experiment Logging."""

    def __init__(self, config: BaseSchema) -> None:
        """Initialise the BasePlotCallback abstract base class.

        Parameters
        ----------
        config : OmegaConf
            Config object

        """
        super().__init__()
        self.config = config
        self.save_basedir = config.hardware.paths.plots

        self.post_processors = None
        self.latlons = None
        init_plot_settings()

        self.plot = self._plot
        self._executor = None
        self._error: BaseException = None
        self.datashader_plotting = config.diagnostics.plot.datashader

        if self.config.diagnostics.plot.asynchronous:
            LOGGER.info("Setting up asynchronous plotting ...")
            self.plot = self._async_plot
            self._executor = ThreadPoolExecutor(max_workers=1)
            self.loop_thread = threading.Thread(target=self.start_event_loop, daemon=True)
            self.loop_thread.start()

    def start_event_loop(self) -> None:
        """Start the event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @rank_zero_only
    def _output_figure(
        self,
        logger: pl.loggers.logger.Logger,
        fig: plt.Figure,
        epoch: int,
        tag: str = "gnn",
        exp_log_tag: str = "val_pred_sample",
    ) -> None:
        """Figure output: save to file and/or display in notebook."""
        if self.save_basedir is not None and fig is not None:
            save_path = Path(
                self.save_basedir,
                "plots",
                f"{tag}_epoch{epoch:03d}.jpg",
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.canvas.draw()
            image_array = np.array(fig.canvas.renderer.buffer_rgba())
            plt.imsave(save_path, image_array, dpi=100)
            if self.config.diagnostics.log.wandb.enabled:
                import wandb

                logger.experiment.log({exp_log_tag: wandb.Image(fig)})
            if self.config.diagnostics.log.mlflow.enabled:
                run_id = logger.run_id
                logger.experiment.log_artifact(run_id, str(save_path))

        plt.close(fig)  # cleanup

    @rank_zero_only
    def _output_gif(
        self,
        logger: pl.loggers.logger.Logger,
        fig: plt.Figure,
        anim: animation.ArtistAnimation,
        epoch: int,
        tag: str = "gnn",
    ) -> None:
        """Animation output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = Path(
                self.save_basedir,
                "plots",
                f"{tag}_epoch{epoch:03d}.gif",
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            anim.save(save_path, writer="pillow", fps=8)

            if self.config.diagnostics.log.wandb.enabled:
                LOGGER.warning("Saving gif animations not tested for wandb.")

            if self.config.diagnostics.log.mlflow.enabled:
                run_id = logger.run_id
                logger.experiment.log_artifact(run_id, str(save_path))

        plt.close(fig)  # cleanup

    @rank_zero_only
    def _plot_with_error_catching(self, trainer: pl.Trainer, args: Any, kwargs: Any) -> None:
        """To execute the plot function but ensuring we catch any errors."""
        try:
            self._plot(trainer, *args, **kwargs)
        except BaseException:
            import os

            LOGGER.exception(traceback.format_exc())
            os._exit(1)  # to force exit when sanity val steps are used

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Teardown the callback."""
        del trainer, pl_module, stage  # unused
        LOGGER.info("Teardown of the Plot Callback ...")

        if self._executor is not None:
            LOGGER.info("waiting and shutting down the executor ...")
            self._executor.shutdown(wait=False, cancel_futures=True)

            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join()
            # Step 3: Close the asyncio event loop
            self.loop_thread._stop()
            self.loop_thread._delete()

    def apply_output_mask(self, pl_module: pl.LightningModule, data: torch.Tensor) -> torch.Tensor:
        if hasattr(pl_module, "output_mask") and pl_module.output_mask is not None:
            # Fill with NaNs values where the mask is False
            data[:, :, ~pl_module.output_mask, :] = np.nan
        return data

    @abstractmethod
    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Plotting function to be implemented by subclasses."""

    # Async function to run the plot function in the background thread
    async def submit_plot(self, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        """Async function or coroutine to schedule the plot function."""
        loop = asyncio.get_running_loop()
        # run_in_executor doesn't support keyword arguments,
        await loop.run_in_executor(
            self._executor,
            self._plot_with_error_catching,
            trainer,
            args,
            kwargs,
        )  # because loop.run_in_executor expects positional arguments, not keyword arguments

    @rank_zero_only
    def _async_plot(
        self,
        trainer: pl.Trainer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run the plot function asynchronously.

        This is the function that is called by the callback. It schedules the plot
        function to run in the background thread. Since we have an event loop running in
        the background thread, we need to schedule the plot function to run in that
        loop.
        """
        asyncio.run_coroutine_threadsafe(self.submit_plot(trainer, *args, **kwargs), self.loop)


class BasePerBatchPlotCallback(BasePlotCallback):
    """Base Callback for plotting at the end of each batch."""

    def __init__(self, config: OmegaConf, every_n_batches: int | None = None):
        """Initialise the BasePerBatchPlotCallback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        every_n_batches : int, optional
            Batch Frequency to plot at, by default None
            If not given, uses default from config at `diagnostics.plot.frequency.batch`

        """
        super().__init__(config)
        self.every_n_batches = every_n_batches or self.config.diagnostics.plot.frequency.batch

        if self.config.diagnostics.plot.asynchronous and self.config.dataloader.read_group_size > 1:
            LOGGER.warning("Asynchronous plotting can result in NCCL timeouts with reader_group_size > 1.")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        **kwargs,
    ) -> None:
        if batch_idx % self.every_n_batches == 0:
            # gather tensors if necessary
            batch = pl_module.allgather_batch(batch)
            # output: [loss, [pred1, pred2, ...]], gather predictions for plotting
            output = [output[0], [pl_module.allgather_batch(pred) for pred in output[1]]]

            # When running in Async mode, it might happen that in the last epoch these tensors
            # have been moved to the cpu (and then the denormalising would fail as the 'input_tensor' would be on CUDA
            # but internal ones would be on the cpu), The lines below allow to address this problem
            self.post_processors = copy.deepcopy(pl_module.model.post_processors)
            for post_processor in self.post_processors.processors.values():
                if hasattr(post_processor, "nan_locations"):
                    post_processor.nan_locations = pl_module.allgather_batch(post_processor.nan_locations)
            self.post_processors = self.post_processors.cpu()

            self.plot(
                trainer,
                pl_module,
                output,
                batch,
                batch_idx,
                epoch=trainer.current_epoch,
                **kwargs,
            )


class BasePerEpochPlotCallback(BasePlotCallback):
    """Base Callback for plotting at the end of each epoch."""

    def __init__(self, config: OmegaConf, every_n_epochs: int | None = None):
        """Initialise the BasePerEpochPlotCallback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        every_n_epochs : int, optional
            Epoch frequency to plot at, by default None
            If not given, uses default from config at `diagnostics.plot.frequency.epoch`
        """
        super().__init__(config)
        self.every_n_epochs = every_n_epochs or self.config.diagnostics.plot.frequency.epoch

    @rank_zero_only
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        **kwargs,
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.plot(trainer, pl_module, epoch=trainer.current_epoch, **kwargs)


class LongRolloutPlots(BasePlotCallback):
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

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if (batch_idx) == 0 and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            batch = pl_module.allgather_batch(batch)
            output = [output[0], [pl_module.allgather_batch(pred) for pred in output[1]]]

            self.post_processors = copy.deepcopy(pl_module.model.post_processors)
            for post_processor in self.post_processors.processors.values():
                if hasattr(post_processor, "nan_locations"):
                    post_processor.nan_locations = pl_module.allgather_batch(post_processor.nan_locations)
            self.post_processors = self.post_processors.cpu()

            precision_mapping = {
                "16-mixed": torch.float16,
                "bf16-mixed": torch.bfloat16,
            }
            prec = trainer.precision
            dtype = precision_mapping.get(prec)
            context = torch.autocast(device_type=batch.device.type, dtype=dtype) if dtype is not None else nullcontext()

            if self.config.diagnostics.plot.asynchronous:
                LOGGER.warning("Asynchronous plotting not supported for long rollout plots.")

            with context:
                # Issue with running asyncronously, so call the plot function directly
                self._plot(trainer, pl_module, output, batch, batch_idx, trainer.current_epoch)


class GraphTrainableFeaturesPlot(BasePerEpochPlotCallback):
    """Visualize the node & edge trainable features defined."""

    def __init__(self, config: OmegaConf, every_n_epochs: int | None = None) -> None:
        """Initialise the GraphTrainableFeaturesPlot callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        every_n_epochs: int | None, optional
            Override for frequency to plot at, by default None
        """
        super().__init__(config, every_n_epochs=every_n_epochs)
        self.q_extreme_limit = config.get("quantile_edges_to_represent", 0.05)

    def get_node_trainable_tensors(self, node_attributes: NamedNodesAttributes) -> dict[str, torch.Tensor]:
        return {
            name: tt.trainable for name, tt in node_attributes.trainable_tensors.items() if tt.trainable is not None
        }

    def get_edge_trainable_modules(self, model: torch.nn.Module) -> dict[tuple[str, str], torch.Tensor]:
        trainable_modules = {
            (model._graph_name_data, model._graph_name_hidden): model.encoder,
            (model._graph_name_hidden, model._graph_name_data): model.decoder,
        }

        if isinstance(model.processor, GraphEdgeMixin):
            trainable_modules[model._graph_name_hidden, model._graph_name_hidden] = model.processor

        return {name: module for name, module in trainable_modules.items() if module.trainable.trainable is not None}

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int,
    ) -> None:
        _ = epoch
        model = pl_module.model.module.model if hasattr(pl_module.model, "module") else pl_module.model.model

        if len(node_trainable_tensors := self.get_node_trainable_tensors(model.node_attributes)):
            fig = plot_graph_node_features(model, node_trainable_tensors, datashader=self.datashader_plotting)

            self._output_figure(
                trainer.logger,
                fig,
                epoch=trainer.current_epoch,
                tag="node_trainable_params",
                exp_log_tag="node_trainable_params",
            )
        else:
            LOGGER.warning("There are no trainable node attributes to plot.")

        if len(edge_trainable_modules := self.get_edge_trainable_modules(model)):
            fig = plot_graph_edge_features(model, edge_trainable_modules, q_extreme_limit=self.q_extreme_limit)

            self._output_figure(
                trainer.logger,
                fig,
                epoch=trainer.current_epoch,
                tag="edge_trainable_params",
                exp_log_tag="edge_trainable_params",
            )
        else:
            LOGGER.warning("There are no trainable edge attributes to plot.")


class PlotLoss(BasePerBatchPlotCallback):
    """Plots the unsqueezed loss over rollouts."""

    def __init__(
        self,
        config: OmegaConf,
        parameter_groups: dict[dict[str, list[str]]],
        every_n_batches: int | None = None,
    ) -> None:
        """Initialise the PlotLoss callback.

        Parameters
        ----------
        config : OmegaConf
            Object with configuration settings
        parameter_groups : dict
            Dictionary with parameter groups with parameter names as keys
        every_n_batches : int, optional
            Override for batch frequency, by default None

        """
        super().__init__(config, every_n_batches=every_n_batches)
        self.parameter_names = None
        self.parameter_groups = parameter_groups
        if self.parameter_groups is None:
            self.parameter_groups = {}

    @cached_property
    def sort_and_color_by_parameter_group(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict, list]:
        """Sort parameters by group and prepare colors."""

        def automatically_determine_group(name: str) -> str:
            # first prefix of parameter name is group name
            parts = name.split("_")
            if len(parts) == 1:
                # if no underscore is present, return full name
                return parts[0]
            # else remove last part of name
            return name[: -len(parts[-1]) - 1]

        # group parameters by their determined group name for > 15 parameters
        if len(self.parameter_names) <= 15:
            # for <= 15 parameters, keep the full name of parameters
            parameters_to_groups = np.array(self.parameter_names)
            sort_by_parameter_group = np.arange(len(self.parameter_names), dtype=int)
        else:
            parameters_to_groups = np.array(
                [
                    next(
                        (
                            group_name
                            for group_name, group_parameters in self.parameter_groups.items()
                            if name in group_parameters
                        ),
                        automatically_determine_group(name),
                    )
                    for name in self.parameter_names
                ],
            )

            unique_group_list, group_inverse, group_counts = np.unique(
                parameters_to_groups,
                return_inverse=True,
                return_counts=True,
            )

            # join parameter groups that appear only once and are not given in config-file
            unique_group_list = np.array(
                [
                    (unique_group_list[tn] if count > 1 or unique_group_list[tn] in self.parameter_groups else "other")
                    for tn, count in enumerate(group_counts)
                ],
            )
            parameters_to_groups = unique_group_list[group_inverse]
            unique_group_list, group_inverse = np.unique(parameters_to_groups, return_inverse=True)

            # sort parameters by groups
            sort_by_parameter_group = np.argsort(group_inverse, kind="stable")

        # apply new order to parameters
        sorted_parameter_names = np.array(self.parameter_names)[sort_by_parameter_group]
        parameters_to_groups = parameters_to_groups[sort_by_parameter_group]
        unique_group_list, group_inverse, group_counts = np.unique(
            parameters_to_groups,
            return_inverse=True,
            return_counts=True,
        )

        # get a color per group and project to parameter list
        cmap = "tab10" if len(unique_group_list) <= 10 else "tab20"
        if len(unique_group_list) > 20:
            LOGGER.warning("More than 20 groups detected, but colormap has only 20 colors.")
        # if all groups have count 1 use black color
        bar_color_per_group = (
            np.tile("k", len(group_counts))
            if not np.any(group_counts - 1)
            else plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_group_list)))
        )

        # set x-ticks
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
                    # linebreak after 50 characters
                    text_label += "\n"
                    string_length = 0
            legend_patches.append(mpatches.Patch(color=bar_color_per_group[group_idx], label=text_label[:-2]))

        return (
            sort_by_parameter_group,
            bar_color_per_group[group_inverse],
            xticks,
            legend_patches,
        )

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

        rollout = getattr(pl_module, "rollout", 0)

        for rollout_step in range(rollout):
            y_hat = outputs[1][rollout_step]
            y_true = batch[
                :,
                pl_module.multi_step + rollout_step,
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
                tag=f"loss_rstep_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
                exp_log_tag=f"loss_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:

        if batch_idx % self.every_n_batches == 0:

            self.loss = copy.deepcopy(pl_module.loss)

            # gather nan-mask weight shards, don't gather if constant in grid dimension (broadcastable)
            if (
                hasattr(self.loss.scaler, "nan_mask_weights")
                and self.loss.scaler.nan_mask_weights.shape[pl_module.grid_dim] != 1
            ):
                self.loss.scaler.nan_mask_weights = pl_module.allgather_batch(self.loss.scaler.nan_mask_weights)

            super().on_validation_batch_end(
                trainer,
                pl_module,
                output,
                batch,
                batch_idx,
            )


class BasePlotAdditionalMetrics(BasePerBatchPlotCallback):
    """Base processing class for additional metrics."""

    def process(
        self,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().detach().cpu().numpy())

        input_tensor = (
            batch[
                :,
                pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
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


class PlotSample(BasePlotAdditionalMetrics):
    """Plots a post-processed sample: input, target and prediction."""

    def __init__(
        self,
        config: OmegaConf,
        sample_idx: int,
        parameters: list[str],
        accumulation_levels_plot: list[float],
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict[str, Colormap] | None = None,
        per_sample: int = 6,
        every_n_batches: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the PlotSample callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        sample_idx : int
            Sample to plot
        parameters : list[str]
            Parameters to plot
        accumulation_levels_plot : list[float]
            Accumulation levels to plot
        precip_and_related_fields : list[str] | None, optional
            Precip variable names, by default None
        colormaps : dict[str, Colormap] | None, optional
            Dictionary of colormaps, by default None
        per_sample : int, optional
            Number of plots per sample, by default 6
        every_n_batches : int, optional
            Batch frequency to plot at, by default None
        """
        del kwargs
        super().__init__(config, every_n_batches=every_n_batches)
        self.sample_idx = sample_idx
        self.parameters = parameters

        self.precip_and_related_fields = precip_and_related_fields
        self.accumulation_levels_plot = accumulation_levels_plot
        self.per_sample = per_sample
        self.colormaps = colormaps

        LOGGER.info(
            "Using defined accumulation colormap for fields: %s",
            self.precip_and_related_fields,
        )

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
        rollout = getattr(pl_module, "rollout", 0)
        for rollout_step in range(rollout):
            fig = plot_predicted_multilevel_flat_sample(
                plot_parameters_dict,
                self.per_sample,
                self.latlons,
                self.accumulation_levels_plot,
                data[0, ...].squeeze(),
                data[rollout_step + 1, ...].squeeze(),
                output_tensor[rollout_step, ...],
                datashader=self.datashader_plotting,
                precip_and_related_fields=self.precip_and_related_fields,
                colormaps=self.colormaps,
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"pred_val_sample_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank{local_rank:01d}",
                exp_log_tag=f"val_pred_sample_rstep{rollout_step:02d}_rank{local_rank:01d}",
            )


class PlotSpectrum(BasePlotAdditionalMetrics):
    """Plots TP related metric comparing target and prediction.

    The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.

    - Power Spectrum
    """

    def __init__(
        self,
        config: OmegaConf,
        sample_idx: int,
        parameters: list[str],
        min_delta: float | None = None,
        every_n_batches: int | None = None,
    ) -> None:
        """Initialise the PlotSpectrum callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        sample_idx : int
            Sample to plot
        parameters : list[str]
            Parameters to plot
        every_n_batches : int | None, optional
            Override for batch frequency, by default None
        """
        super().__init__(config, every_n_batches=every_n_batches)
        self.sample_idx = sample_idx
        self.parameters = parameters
        self.min_delta = min_delta

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

        rollout = getattr(pl_module, "rollout", 0)
        for rollout_step in range(rollout):
            # Build dictionary of inidicies and parameters to be plotted

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
                data[0, ...].squeeze(),
                data[rollout_step + 1, ...].squeeze(),
                output_tensor[rollout_step, ...],
                min_delta=self.min_delta,
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"pred_val_spec_rstep_{rollout_step:02d}_batch{batch_idx:04d}_rank{local_rank:01d}",
                exp_log_tag=f"pred_val_spec_rstep_{rollout_step:02d}_rank{local_rank:01d}",
            )


class PlotHistogram(BasePlotAdditionalMetrics):
    """Plots histograms comparing target and prediction.

    The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.
    """

    def __init__(
        self,
        config: OmegaConf,
        sample_idx: int,
        parameters: list[str],
        precip_and_related_fields: list[str] | None = None,
        log_scale: bool = False,
        every_n_batches: int | None = None,
    ) -> None:
        """Initialise the PlotHistogram callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        sample_idx : int
            Sample to plot
        parameters : list[str]
            Parameters to plot
        precip_and_related_fields : list[str] | None, optional
            Precip variable names, by default None
        every_n_batches : int | None, optional
            Override for batch frequency, by default None
        """
        super().__init__(config, every_n_batches=every_n_batches)
        self.sample_idx = sample_idx
        self.parameters = parameters
        self.precip_and_related_fields = precip_and_related_fields
        self.log_scale = log_scale
        LOGGER.info(
            "Using precip histogram plotting method for fields: %s.",
            self.precip_and_related_fields,
        )

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

        rollout = getattr(pl_module, "rollout", 0)

        for rollout_step in range(rollout):

            # Build dictionary of inidicies and parameters to be plotted
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
                data[0, ...].squeeze(),
                data[rollout_step + 1, ...].squeeze(),
                output_tensor[rollout_step, ...],
                self.precip_and_related_fields,
                self.log_scale,
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"pred_val_histo_rstep_{rollout_step:02d}_batch{batch_idx:04d}_rank{local_rank:01d}",
                exp_log_tag=f"pred_val_histo_rstep_{rollout_step:02d}_rank{local_rank:01d}",
            )
