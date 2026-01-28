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
from anemoi.training.diagnostics.focus_area import build_spatial_mask
from anemoi.training.diagnostics.focus_area  import NoOpSpatialMask
from anemoi.training.diagnostics.focus_area  import SpatialMask
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
from anemoi.training.losses.utils import reduce_to_last_dim
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.tasks import GraphInterpolator

LOGGER = logging.getLogger(__name__)


class BasePlotCallback(Callback, ABC):
    """Factory for creating a callback that plots data to Experiment Logging."""

    def __init__(
        self,
        config: BaseSchema,
        dataset_names: list[str] | None = None,
        focus_areas: list[dict] | None = None,
    ) -> None:
        """Initialise the BasePlotCallback abstract base class.

        Parameters
        ----------
        config : OmegaConf
            Config object
        focus_areas : list[dict] | None, optional
            Area or point indices to focus the plot on.
            Can be:
            - {"mask_attr_name": str}
            - {"latlon_bbox": [lat_min, lon_min, lat_max, lon_max]}
        """
        super().__init__()
        self.config = config
        self.save_basedir = config.system.output.plots
        self.dataset_names = dataset_names if dataset_names is not None else ["data"]

        self.post_processors = None
        self.latlons = None

        if focus_areas is None:
            focus_areas = [None] * len(self.dataset_names)

        # Focus area for plotting
        self.focus_masks = [
            build_spatial_mask(
                node_attribute_name=fa.get("mask_attr_name", None) if fa is not None else None,
                latlon_bbox=fa.get("latlon_bbox", None) if fa is not None else None,
            )
            for fa in focus_areas
        ]

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

    def _get_init_step(self, rollout_step: int, mode: tuple) -> int:
        """Return index of initial step for plotting."""
        return rollout_step if mode == "time_interp" else 0

    def _get_output_times(self, config: BaseSchema, pl_module: pl.LightningModule) -> tuple:
        """Return times outputted by the model."""
        if isinstance(pl_module, GraphInterpolator):
            output_times = (len(config.training.explicit_times.target), "time_interp")
        else:
            output_times = (getattr(pl_module, "rollout", 0), "forecast")
        return output_times

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
        dataset_names: list[str],
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

    def __init__(
        self,
        config: OmegaConf,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_areas: list[dict] | None = None,
    ):
        """Initialise the BasePerBatchPlotCallback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        every_n_batches : int, optional
            Batch Frequency to plot at, by default None
            If not given, uses default from config at `diagnostics.plot.frequency.batch`

        """
        super().__init__(config, dataset_names=dataset_names, focus_areas=focus_areas)
        self.every_n_batches = every_n_batches or self.config.diagnostics.plot.frequency.batch

        if self.config.diagnostics.plot.asynchronous and self.config.dataloader.read_group_size > 1:
            LOGGER.warning("Asynchronous plotting can result in NCCL timeouts with reader_group_size > 1.")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        **kwargs,
    ) -> None:
        if batch_idx % self.every_n_batches == 0:
            # gather tensors if necessary
            batch = {
                dataset_name: pl_module.allgather_batch(
                    dataset_tensor,
                    pl_module.grid_indices[dataset_name],
                    pl_module.grid_dim,
                )
                for dataset_name, dataset_tensor in batch.items()
            }
            # output: [loss, [pred_dict1, pred_dict2, ...]], gather predictions for plotting
            output = [
                output[0],
                [
                    {
                        dataset_name: pl_module.allgather_batch(
                            dataset_pred,
                            pl_module.grid_indices[dataset_name],
                            pl_module.grid_dim,
                        )
                        for dataset_name, dataset_pred in pred.items()
                    }
                    for pred in output[1]
                ],
            ]

            # When running in Async mode, it might happen that in the last epoch these tensors
            # have been moved to the cpu (and then the denormalising would fail as the 'input_tensor' would be on CUDA
            # but internal ones would be on the cpu), The lines below allow to address this problem
            self.post_processors = copy.deepcopy(pl_module.model.post_processors)
            for dataset_name in self.post_processors:
                for post_processor in self.post_processors[dataset_name].processors.values():
                    if hasattr(post_processor, "nan_locations"):
                        post_processor.nan_locations = pl_module.allgather_batch(
                            post_processor.nan_locations,
                            pl_module.grid_indices[dataset_name],
                            pl_module.grid_dim,
                        )
                self.post_processors[dataset_name] = self.post_processors[dataset_name].cpu()

            output_times = self._get_output_times(self.config, pl_module)

            self.plot(
                trainer,
                pl_module,
                self.dataset_names,
                output,
                batch,
                batch_idx,
                epoch=trainer.current_epoch,
                output_times=output_times,
                **kwargs,
            )


class BasePerEpochPlotCallback(BasePlotCallback):
    """Base Callback for plotting at the end of each epoch."""

    def __init__(
        self,
        config: OmegaConf,
        every_n_epochs: int | None = None,
        dataset_names: list[str] | None = None,
        focus_areas: list[dict] | None = None,
    ):
        """Initialise the BasePerEpochPlotCallback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        every_n_epochs : int, optional
            Epoch frequency to plot at, by default None
            If not given, uses default from config at `diagnostics.plot.frequency.epoch`
        """
        super().__init__(config, dataset_names=dataset_names, focus_areas=focus_areas)
        self.every_n_epochs = every_n_epochs or self.config.diagnostics.plot.frequency.epoch

    @rank_zero_only
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        **kwargs,
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:

            output_times = self._get_output_times(self.config, pl_module)

            self.plot(
                trainer,
                pl_module,
                self.dataset_names,
                epoch=trainer.current_epoch,
                output_times=output_times,
                **kwargs,
            )


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
        focus_area : dict | None, optional
            Area or point indices to focus the plot on.
            Can be:
            - {"mask_attr_name": str}
            - {"latlon_bbox": [lat_min, lon_min, lat_max, lon_max]}
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
            self.latlons = pl_module.model.model._graph_data[pl_module.model.model._graph_name_data].x.detach()
            self.latlons = np.rad2deg(self.latlons.cpu().numpy())

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

    def __init__(
        self,
        config: OmegaConf,
        dataset_names: list[str] | None = None,
        focus_areas: list[dict] | None = None,
        every_n_epochs: int | None = None,
    ) -> None:
        """Initialise the GraphTrainableFeaturesPlot callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        every_n_epochs: int | None, optional
            Override for frequency to plot at, by default None
        """
        super().__init__(config, dataset_names=dataset_names, focus_areas=focus_areas, every_n_epochs=every_n_epochs)
        self.q_extreme_limit = config.get("quantile_edges_to_represent", 0.05)

    def get_node_trainable_tensors(
        self,
        node_attributes: NamedNodesAttributes,
        dataset_name: str,
    ) -> dict[str, torch.Tensor]:
        return {
            name: tt.trainable
            for name, tt in node_attributes[dataset_name].trainable_tensors.items()
            if tt.trainable is not None
        }

    def get_edge_trainable_modules(
        self,
        model: torch.nn.Module,
        dataset_name: str,
    ) -> dict[tuple[str, str], torch.Tensor]:
        # `_graph_name_data` and `_graph_name_hidden` above are the keys for different
        # layers of nodes in the graphs obtained from the config (e.g., "data", "hidden").
        # They are not themselves dictionaries; but the identifiers of the dictionaries
        # of graphs. That is, the “dictionarification” happens one level down.
        # Here, they are used as keys to track and label different parts of the model
        # in the plots for one dataset.
        # Therefore, we don't select `dataset_name` for the `_graph_name_xy`,
        # but only for the modules (encoder/processor/decoder).
        trainable_modules = {}

        # Check encoder
        if (
            hasattr(model, "encoder_graph_provider")
            and dataset_name in model.encoder_graph_provider
            and hasattr(model.encoder_graph_provider[dataset_name], "trainable")
            and model.encoder_graph_provider[dataset_name].trainable is not None
            and model.encoder_graph_provider[dataset_name].trainable.trainable is not None
        ):
            trainable_modules[(model._graph_name_data, model._graph_name_hidden)] = model.encoder_graph_provider[
                dataset_name
            ]

        # Check decoder
        if (
            hasattr(model, "decoder_graph_provider")
            and dataset_name in model.decoder_graph_provider
            and hasattr(model.decoder_graph_provider[dataset_name], "trainable")
            and model.decoder_graph_provider[dataset_name].trainable is not None
            and model.decoder_graph_provider[dataset_name].trainable.trainable is not None
        ):
            trainable_modules[(model._graph_name_hidden, model._graph_name_data)] = model.decoder_graph_provider[
                dataset_name
            ]

        # Check processor
        if (
            hasattr(model, "processor_graph_provider")
            and model.processor_graph_provider.trainable is not None
            and model.processor_graph_provider.trainable.trainable is not None
        ):
            trainable_modules[(model._graph_name_hidden, model._graph_name_hidden)] = model.processor_graph_provider

        return trainable_modules

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        epoch: int,
    ) -> None:
        _ = epoch
        model = pl_module.model.module.model if hasattr(pl_module.model, "module") else pl_module.model.model
        for dataset_name in dataset_names:
            if len(node_trainable_tensors := self.get_node_trainable_tensors(model.node_attributes, dataset_name)):
                fig = plot_graph_node_features(
                    model.node_attributes[dataset_name],
                    node_trainable_tensors,
                    datashader=self.datashader_plotting,
                )

                self._output_figure(
                    trainer.logger,
                    fig,
                    epoch=trainer.current_epoch,
                    tag=f"node_trainable_params_{dataset_name}",
                    exp_log_tag=f"node_trainable_params_{dataset_name}",
                )
            else:
                LOGGER.warning("There are no trainable node attributes to plot.")

            from anemoi.models.models import AnemoiModelEncProcDecHierarchical

            if isinstance(model, AnemoiModelEncProcDecHierarchical):
                LOGGER.warning(
                    "Edge trainable features are not supported for Hierarchical models, skipping plot generation.",
                )
            elif len(edge_trainable_modules := self.get_edge_trainable_modules(model, dataset_name)):
                fig = plot_graph_edge_features(
                    model.node_attributes[dataset_name],
                    edge_trainable_modules,
                    q_extreme_limit=self.q_extreme_limit,
                )

                self._output_figure(
                    trainer.logger,
                    fig,
                    epoch=trainer.current_epoch,
                    tag=f"edge_trainable_params_{dataset_name}",
                    exp_log_tag=f"edge_trainable_params_{dataset_name}",
                )
            else:
                LOGGER.warning("There are no trainable edge attributes to plot.")

    @rank_zero_only
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        **kwargs,
    ) -> None:

        self.plot(trainer, pl_module, self.dataset_names, epoch=trainer.current_epoch, **kwargs)


class PlotLoss(BasePerBatchPlotCallback):
    """Plots the unsqueezed loss over rollouts."""

    def __init__(
        self,
        config: OmegaConf,
        parameter_groups: dict[dict[str, list[str]]],
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_areas: list[dict] | None = None,
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
        focus_areas : list[dict] | None, optional
            Area or point indices to focus the plot on. Can be:
            - {"mask_attr_name": str}
            - {"latlon_bbox": [lat_min, lon_min, lat_max, lon_max]}
        """
        super().__init__(config, dataset_names=dataset_names, every_n_batches=every_n_batches, focus_areas=focus_areas)
        self.parameter_groups = parameter_groups
        self.dataset_names = dataset_names if dataset_names is not None else ["data"]
        if self.parameter_groups is None:
            self.parameter_groups = {}

    def sort_and_color_by_parameter_group(
        self,
        parameter_names: list[str],
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
        if len(parameter_names) <= 15:
            # for <= 15 parameters, keep the full name of parameters
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
                        automatically_determine_group(name),
                    )
                    for name in parameter_names
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
        sorted_parameter_names = np.array(parameter_names)[sort_by_parameter_group]
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
        dataset_names: list[str],
        outputs: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
        output_times: tuple,
    ) -> None:
        logger = trainer.logger
        _ = batch_idx

        for dataset_name, focus_mask in zip(dataset_names, self.focus_masks, strict=True):
            data_indices = pl_module.data_indices[dataset_name]
            parameter_names = list(data_indices.model.output.name_to_index.keys())
            parameter_positions = list(data_indices.model.output.name_to_index.values())
            # reorder parameter_names by position
            parameter_names = [parameter_names[i] for i in np.argsort(parameter_positions)]
            metadata_variables = pl_module.model.metadata["dataset"].get("variables_metadata")

            # Sort the list using the custom key
            argsort_indices = argsort_variablename_variablelevel(
                parameter_names,
                metadata_variables=metadata_variables,
            )
            parameter_names = [parameter_names[i] for i in argsort_indices]
            if not isinstance(self.loss[dataset_name], BaseLoss):
                LOGGER.warning(
                    "Loss function must be a subclass of BaseLoss, or provide `squash`.",
                    RuntimeWarning,
                )

            for rollout_step in range(output_times[0]):
                y_hat = outputs[1][rollout_step][dataset_name]
                y_true = batch[dataset_name][
                    :,
                    0,#pl_module.multi_step + rollout_step, # !TODO
                    ...,
                    data_indices.data.output.full,
                ]
                if self.latlons is None:
                    self.latlons = pl_module.model.model._graph_data[dataset_name][pl_module.model.model._graph_name_data].x.detach()
                    self.latlons = {dataset_name:np.rad2deg(self.latlons.cpu().numpy())}

                _, y_hat, y_true = focus_mask.apply(
                    pl_module.model.model._graph_data,
                    self.latlons[dataset_name],
                    y_hat,
                    y_true,
                )
                tag = focus_mask.tag

                loss = reduce_to_last_dim(self.loss[dataset_name](y_hat, y_true, squash=False).detach().cpu().numpy())

                sort_by_parameter_group, colors, xticks, legend_patches = self.sort_and_color_by_parameter_group(
                    parameter_names,
                )
                loss = loss[argsort_indices]
                fig = plot_loss(loss[sort_by_parameter_group], colors, xticks, legend_patches)

                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=f"loss_{dataset_name}_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}{tag}",
                    exp_log_tag=f"loss_sample_{dataset_name}_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}{tag}",
                )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:

        if batch_idx % self.every_n_batches == 0:

            self.loss = copy.deepcopy(pl_module.loss)

            # gather nan-mask weight shards, don't gather if constant in grid dimension (broadcastable)
            for dataset in self.loss:
                if (
                    hasattr(self.loss[dataset].scaler, "nan_mask_weights")
                    and self.loss[dataset].scaler.nan_mask_weights.shape[pl_module.grid_dim] != 1
                ):
                    self.loss[dataset].scaler.nan_mask_weights = pl_module.allgather_batch(
                        self.loss[dataset].scaler.nan_mask_weights,
                        pl_module.grid_indices[dataset],
                        pl_module.grid_dim,
                    )

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
        dataset_name: str,
        outputs: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
        batch: dict[str, torch.Tensor],
        output_times: tuple,
        focus_mask: SpatialMask | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.latlons is None:
            self.latlons = {}

        if dataset_name not in self.latlons:
            self.latlons[dataset_name] = pl_module.model.model._graph_data[dataset_name][
                pl_module.model.model._graph_name_data
            ].x.detach()
            self.latlons[dataset_name] = np.rad2deg(self.latlons[dataset_name].cpu().numpy())

        # prepare input and output tensors for plotting one dataset specified by dataset_name
        input_tensor = (
            batch[dataset_name][
                :,
                pl_module.multi_step - 1 : pl_module.multi_step + output_times[0] + 1,
                ...,
                pl_module.data_indices[dataset_name].data.output.full,
            ]
            .detach()
            .cpu()
        )
        data = self.post_processors[dataset_name](input_tensor)[self.sample_idx]
        output_tensor = torch.cat(
            tuple(
                self.post_processors[dataset_name](x[dataset_name][:, ...].detach().cpu(), in_place=False)[
                    self.sample_idx : self.sample_idx + 1
                ]
                for x in outputs[1]
            ),
        )
        output_tensor = pl_module.output_mask[dataset_name].apply(output_tensor, dim=2, fill_value=np.nan).numpy()
        data[1:, ...] = pl_module.output_mask[dataset_name].apply(data[1:, ...], dim=2, fill_value=np.nan)
        data = data.numpy()

        focus_mask = focus_mask or NoOpSpatialMask()
        # Apply spatial mask

        latlons, data, output_tensor = focus_mask.apply(
            pl_module.model.model._graph_data,
            self.latlons[dataset_name],
            data,
            output_tensor,
        )

        return data, output_tensor, latlons


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
        focus_areas: list[dict] | None = None,
        dataset_names: list[str] | None = None,
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
        focus_areas : list[dict] | None, optional
            Area or point indices to focus the plot on. Can be:
            - {"mask_attr_name": str}
            - {"latlon_bbox": [lat_min, lon_min, lat_max, lon_max]}
        """
        del kwargs
        super().__init__(config, dataset_names=dataset_names, every_n_batches=every_n_batches, focus_areas=focus_areas)
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
        dataset_names: list[str],
        outputs: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
        output_times: tuple,
    ) -> None:
        logger = trainer.logger

        for dataset_name, focus_mask in zip(dataset_names, self.focus_masks, strict=True):
            # Build dictionary of indices and parameters to be plotted
            diagnostics = (
                []
                if self.config.data.datasets[dataset_name].diagnostic is None
                else self.config.data.datasets[dataset_name].diagnostic
            )
            plot_parameters_dict = {
                pl_module.data_indices[dataset_name].model.output.name_to_index[name]: (
                    name,
                    name not in diagnostics,
                )
                for name in self.parameters
            }
            data, output_tensor, latlons = self.process(pl_module, dataset_name, outputs, batch, output_times, focus_mask)
            tag = focus_mask.tag
            local_rank = pl_module.local_rank


            for rollout_step in range(output_times[0]):
                init_step = self._get_init_step(rollout_step, output_times[1])

                y_true = data[rollout_step + 1, ...].squeeze() if data.ndim==2 else None
                per_sample = self.per_sample if data.ndim ==2 else 3

                fig = plot_predicted_multilevel_flat_sample(
                    plot_parameters_dict,
                    per_sample,
                    latlons,
                    self.accumulation_levels_plot,
                    data[init_step, ...].squeeze(),
                    y_true,
                    output_tensor[rollout_step, ...],
                    datashader=self.datashader_plotting,
                    precip_and_related_fields=self.precip_and_related_fields,
                    colormaps=self.colormaps,
                )

                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=f"pred_val_sample_{dataset_name}_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank{local_rank:01d}{tag}",
                    exp_log_tag=f"val_pred_sample_{dataset_name}_rstep{rollout_step:02d}_rank{local_rank:01d}{tag}",
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
        dataset_names: list[str] | None = None,
        focus_areas: list[dict] | None = None,
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
        focus_areas : list[dict] | None, optional
            Area or point indices to focus the plot on. Can be:
            - {"mask_attr_name": str}
            - {"latlon_bbox": [lat_min, lon_min, lat_max, lon_max]}
        """
        super().__init__(config, dataset_names=dataset_names, every_n_batches=every_n_batches, focus_areas=focus_areas)
        self.sample_idx = sample_idx
        self.parameters = parameters
        self.min_delta = min_delta

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
        output_times: tuple,
    ) -> None:
        logger = trainer.logger

        local_rank = pl_module.local_rank
        for dataset_name, focus_mask in zip(dataset_names, self.focus_masks, strict=True):
            data, output_tensor, latlons = self.process(pl_module, dataset_name, outputs, batch, output_times,focus_mask)
            tag = focus_mask.tag

            for rollout_step in range(output_times[0]):
                # Build dictionary of inidicies and parameters to be plotted
                diagnostics = (
                    []
                    if self.config.data.datasets[dataset_name].diagnostic is None
                    else self.config.data.datasets[dataset_name].diagnostic
                )
                plot_parameters_dict_spectrum = {
                    pl_module.data_indices[dataset_name].model.output.name_to_index[name]: (
                        name,
                        name not in diagnostics,
                    )
                    for name in self.parameters
                }

                init_step = self._get_init_step(rollout_step, output_times[1])
                y_true = data[rollout_step + 1, ...].squeeze() if data.ndim==2 else None

                fig = plot_power_spectrum(
                    parameters=plot_parameters_dict_spectrum,
                    latlons=latlons,
                    x=data[init_step, ...].squeeze(),
                    y_true=y_true,
                    y_pred=output_tensor[rollout_step, ...].squeeze(),
                    min_delta=self.min_delta,
                )

                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=f"pred_val_spec_{dataset_name}_rstep_{rollout_step:02d}_batch{batch_idx:04d}_rank{local_rank:01d}{tag}",
                    exp_log_tag=f"pred_val_spec_{dataset_name}_rstep_{rollout_step:02d}_rank{local_rank:01d}{tag}",
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
        dataset_names: list[str] | None = None,
        focus_areas: list[dict] | None = None,
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
        focus_areas : list[dict] | None, optional
            Area or point indices to focus the plot on. Can be:
            - {"mask_attr_name": str}
            - {"latlon_bbox": [lat_min, lon_min, lat_max, lon_max]}
        """
        super().__init__(config, dataset_names=dataset_names, every_n_batches=every_n_batches, focus_areas=focus_areas)
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
        dataset_names: list[str],
        outputs: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
        output_times: tuple,
    ) -> None:
        logger = trainer.logger

        local_rank = pl_module.local_rank

        for dataset_name, focus_mask in zip(dataset_names, self.focus_masks, strict=True):
            data, output_tensor, _ = self.process(pl_module, dataset_name, outputs, batch, output_times, focus_mask)
            tag = focus_mask.tag


            for rollout_step in range(output_times[0]):

                # Build dictionary of inidicies and parameters to be plotted
                diagnostics = (
                    []
                    if self.config.data.datasets[dataset_name].diagnostic is None
                    else self.config.data.datasets[dataset_name].diagnostic
                )

                plot_parameters_dict_histogram = {
                    pl_module.data_indices[dataset_name].model.output.name_to_index[name]: (
                        name,
                        name not in diagnostics,
                    )
                    for name in self.parameters
                }

                init_step = self._get_init_step(rollout_step, output_times[1])
                y_true = data[rollout_step + 1, ...].squeeze() if data.ndim==2 else None

                fig = plot_histogram(
                    plot_parameters_dict_histogram,
                    data[init_step, ...].squeeze(),
                    y_true,
                    output_tensor[rollout_step, ...],
                    self.precip_and_related_fields,
                    self.log_scale,
                )

                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=f"pred_val_histo_{dataset_name}_rstep_{rollout_step:02d}_batch{batch_idx:04d}_rank{local_rank:01d}{tag}",
                    exp_log_tag=f"pred_val_histo_{dataset_name}_rstep_{rollout_step:02d}_rank{local_rank:01d}{tag}",
                )
