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
from anemoi.training.diagnostics.plots import argsort_variablename_variablelevel
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

LOGGER = logging.getLogger(__name__)


class BasePlotCallback(Callback, ABC):
    """Factory for creating a callback that plots data to Experiment Logging."""

    def __init__(
        self,
        config: BaseSchema,
        dataset_names: list[str] | None = None,
    ) -> None:
        """Initialise the BasePlotCallback abstract base class.

        Parameters
        ----------
        config : OmegaConf
            Config object

        """
        super().__init__()
        self.config = config
        self.save_basedir = config.system.output.plots
        self.dataset_names = dataset_names if dataset_names is not None else ["data"]

        self.post_processors = None
        self.latlons = None

        init_plot_settings()

        self.plot = self._plot
        self._executor = None
        self._error: BaseException = None
        self.datashader_plotting = config.diagnostics.plot.datashader
        self.projection_kind = getattr(config.diagnostics.plot, "projection_kind", "equirectangular")

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
            if logger and logger.logger_name == "wandb":
                import wandb

                logger.experiment.log({exp_log_tag: wandb.Image(fig)})
            elif logger and logger.logger_name == "mlflow":
                run_id = logger.run_id
                logger.experiment.log_artifact(run_id, str(save_path), artifact_path=self.artifact_subfolder)

        plt.close(fig)  # cleanup

    @rank_zero_only
    def _output_gif(
        self,
        logger: pl.loggers.logger.Logger,
        fig: plt.Figure,
        anim: animation.Animation,
        epoch: int,
        tag: str = "gnn",
        fps: int = 8,
    ) -> None:
        """Animation output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = Path(
                self.save_basedir,
                "plots",
                f"{tag}_epoch{epoch:03d}.gif",
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            anim.save(save_path, writer="pillow", fps=fps)

            if self.config.diagnostics.log.wandb.enabled:
                LOGGER.warning("Saving gif animations not tested for wandb.")

            if self.config.diagnostics.log.mlflow.enabled:
                run_id = logger.run_id
                logger.experiment.log_artifact(run_id, str(save_path), artifact_path=self.artifact_subfolder)

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
            data[:, :, :, ~pl_module.output_mask, :] = np.nan
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
        super().__init__(config, dataset_names=dataset_names)
        self.every_n_batches = every_n_batches or self.config.diagnostics.plot.frequency.batch

        if self.config.diagnostics.plot.asynchronous and self.config.dataloader.read_group_size > 1:
            LOGGER.warning("Asynchronous plotting can result in NCCL timeouts with reader_group_size > 1.")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: tuple[torch.Tensor, list[dict[str, torch.Tensor]] | dict[str, torch.Tensor]],
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
            preds = output[1]
            if isinstance(preds, dict):
                preds = [preds]
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
                    for pred in preds
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

            self.plot(
                trainer,
                pl_module,
                self.dataset_names,
                output,
                batch,
                batch_idx,
                epoch=trainer.current_epoch,
                output_times=pl_module.output_times,
                **kwargs,
            )


class BasePerEpochPlotCallback(BasePlotCallback):
    """Base Callback for plotting at the end of each epoch."""

    def __init__(
        self,
        config: OmegaConf,
        every_n_epochs: int | None = None,
        dataset_names: list[str] | None = None,
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
        super().__init__(config, dataset_names=dataset_names)
        self.every_n_epochs = every_n_epochs or self.config.diagnostics.plot.frequency.epoch

    @rank_zero_only
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        **kwargs,
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:

            self.plot(
                trainer,
                pl_module,
                self.dataset_names,
                epoch=trainer.current_epoch,
                output_times=pl_module.output_times,
                **kwargs,
            )


class LongRolloutPlots(BasePlotCallback):
    """Evaluates the model performance over a (longer) rollout window.

    Updated to support multi-dataset batches:
      batch: dict[dataset_name, tensor]
      rollout_step yields y_pred as either:
        - dict[dataset_name, tensor] (multi-dataset)
        - tensor (single-dataset legacy)
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
        dataset_names: list[str] | None = None,
    ) -> None:
        super().__init__(config, dataset_names=dataset_names)

        self.every_n_epochs = every_n_epochs

        self.rollout = rollout or []
        self.video_rollout = video_rollout
        self.max_rollout = max(self.rollout) if self.rollout else 0
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
                "Setting up callback for plots with long rollout: rollout for plots = %s, "
                "rollout for video = %s, frequency = every %d epoch."
            ),
            self.rollout,
            self.video_rollout,
            every_n_epochs,
        )

        if self.config.diagnostics.plot.asynchronous and self.config.dataloader.read_group_size > 1:
            LOGGER.warning("Asynchronous plotting can result in NCCL timeouts with reader_group_size > 1.")

    @staticmethod
    def _select_pred(y_pred: Any, dataset_name: str) -> torch.Tensor:
        """Select the per-dataset prediction tensor from y_pred."""
        if isinstance(y_pred, dict):
            return y_pred[dataset_name]
        return y_pred

    def _get_diagnostics(self, dataset_name: str) -> list[str]:
        try:
            diag = self.config.data.datasets[dataset_name].diagnostic
        except Exception:
            diag = None
        return [] if diag is None else list(diag)

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        output: list[torch.Tensor],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
        **_: Any,
    ) -> None:
        _ = output
        start_time = time.time()
        logger = trainer.logger

        if self.latlons is None:
            self.latlons = {}

        # Ensure dataset_names only includes available keys
        ds_names = [d for d in dataset_names if d in batch]
        if not ds_names:
            LOGGER.warning("LongRolloutPlots: no matching dataset_names in batch. dataset_names=%s batch_keys=%s", dataset_names, list(batch.keys()))
            return

        # Prepare per-dataset plotting metadata once
        plot_parameters = {}
        init_data0 = {}

        for dataset_name in ds_names:
            diagnostics = self._get_diagnostics(dataset_name)
            di = pl_module.data_indices[dataset_name]

            plot_parameters_dict = {
                di.model.output.name_to_index[name]: (name, name not in diagnostics)
                for name in self.parameters
                if name in di.model.output.name_to_index
            }
            if not plot_parameters_dict:
                LOGGER.warning("LongRolloutPlots: no parameters matched for dataset '%s' (requested=%s).", dataset_name, self.parameters)
                continue

            plot_parameters[dataset_name] = plot_parameters_dict

            if dataset_name not in self.latlons:
                self.latlons[dataset_name] = pl_module.model.model._graph_data[dataset_name].x.detach()
                self.latlons[dataset_name] = np.rad2deg(self.latlons[dataset_name].cpu().numpy())

            # input tensor at init time (already preprocessed in-place)
            input_tensor_0 = (
                batch[dataset_name][
                    :,
                    pl_module.n_step_input - 1,
                    ...,
                    di.data.output.full,
                ]
                .detach()
                .cpu()
            )
            init_data0[dataset_name] = self.post_processors[dataset_name](input_tensor_0)[self.sample_idx]

        # Drop datasets that failed parameter matching
        ds_names = [d for d in ds_names if d in plot_parameters and d in init_data0]
        if not ds_names:
            return

        # Video buffers per dataset
        video_data_over_time: dict[str, list] = {}
        vmin: dict[str, np.ndarray] = {}
        vmax: dict[str, np.ndarray] = {}
        if self.video_rollout:
            for dataset_name in ds_names:
                nvars = len(plot_parameters[dataset_name])
                video_data_over_time[dataset_name] = []
                vmin[dataset_name] = np.inf * np.ones(nvars, dtype=float)
                vmax[dataset_name] = -np.inf * np.ones(nvars, dtype=float)

        with torch.no_grad():
            for rollout_step, (_, _, y_pred) in enumerate(
                pl_module._rollout_step(
                    batch,  # dict[str, tensor]
                    rollout=self.max_rollout,
                    validation_mode=True,
                ),
            ):
                # Plot requested rollout steps (1-indexed in config)
                if (rollout_step + 1) in self.rollout:
                    for dataset_name in ds_names:
                        y_pred_ds = self._select_pred(y_pred, dataset_name)
                        self._plot_rollout_step(
                            pl_module=pl_module,
                            dataset_name=dataset_name,
                            plot_parameters_dict=plot_parameters[dataset_name],
                            input_batch=batch[dataset_name],
                            data_0=init_data0[dataset_name],
                            rollout_step=rollout_step,
                            y_pred=y_pred_ds,
                            batch_idx=batch_idx,
                            epoch=epoch,
                            logger=logger,
                        )

                # Store video frames
                if self.video_rollout and rollout_step < self.video_rollout:
                    for dataset_name in ds_names:
                        y_pred_ds = self._select_pred(y_pred, dataset_name)
                        video_data_over_time[dataset_name], vmin[dataset_name], vmax[dataset_name] = self._store_video_frame_data(
                            data_over_time=video_data_over_time[dataset_name],
                            y_pred=y_pred_ds,
                            dataset_name=dataset_name,
                            plot_parameters_dict=plot_parameters[dataset_name],
                            vmin=vmin[dataset_name],
                            vmax=vmax[dataset_name],
                        )

            # Generate videos
            if self.video_rollout:
                for dataset_name in ds_names:
                    self._generate_video_rollout(
                        dataset_name=dataset_name,
                        data_0=init_data0[dataset_name],
                        data_over_time=video_data_over_time[dataset_name],
                        plot_parameters_dict=plot_parameters[dataset_name],
                        vmin=vmin[dataset_name],
                        vmax=vmax[dataset_name],
                        rollout_step=self.video_rollout,
                        batch_idx=batch_idx,
                        epoch=epoch,
                        logger=logger,
                        animation_interval=self.animation_interval,
                    )

        LOGGER.info(
            "Time taken to plot/animate samples for longer rollout: %d seconds",
            int(time.time() - start_time),
        )

    @rank_zero_only
    def _plot_rollout_step(
        self,
        pl_module: pl.LightningModule,
        dataset_name: str,
        plot_parameters_dict: dict[int, tuple[str, bool]],
        input_batch: torch.Tensor,
        data_0: np.ndarray,
        rollout_step: int,
        y_pred: torch.Tensor,
        batch_idx: int,
        epoch: int,
        logger: pl.loggers.logger.Logger,
    ) -> None:
        di = pl_module.data_indices[dataset_name]

        # true output at this rollout step
        input_tensor_rollout_step = (
            input_batch[
                :,
                pl_module.n_step_input + rollout_step,  # (n_step_input - 1) + (rollout_step + 1)
                ...,
                di.data.output.full,
            ]
            .detach()
            .cpu()
        )
        data_rollout_step = self.post_processors[dataset_name](input_tensor_rollout_step)[self.sample_idx]

        # predicted output tensor
        output_tensor = self.post_processors[dataset_name](y_pred.detach().cpu(), in_place=False)[
            self.sample_idx : self.sample_idx + 1
        ]

        fig = plot_predicted_multilevel_flat_sample(
            plot_parameters_dict,
            self.per_sample,
            self.latlons[dataset_name],
            self.accumulation_levels_plot,
            data_0.squeeze(),
            data_rollout_step.squeeze(),
            output_tensor[0, 0, :, :],  # rolloutstep, first member
            colormaps=self.colormaps,
            datashader=self.datashader_plotting,
        )
        self._output_figure(
            logger,
            fig,
            epoch=epoch,
            tag=(
                f"pred_val_sample_{dataset_name}_rstep{rollout_step + 1:03d}_"
                f"batch{batch_idx:04d}_rank{pl_module.local_rank:01d}"
            ),
            exp_log_tag=f"pred_val_sample_{dataset_name}_rstep{rollout_step + 1:03d}_rank{pl_module.local_rank:01d}",
        )

    def _store_video_frame_data(
        self,
        data_over_time: list,
        y_pred: torch.Tensor,
        dataset_name: str,
        plot_parameters_dict: dict[int, tuple[str, bool]],
        vmin: np.ndarray,
        vmax: np.ndarray,
    ) -> tuple[list, np.ndarray, np.ndarray]:
        # predicted output tensors for video
        output_tensor = self.post_processors[dataset_name](y_pred.detach().cpu(), in_place=False)[
            self.sample_idx : self.sample_idx + 1
        ]
        var_indices = np.array(list(plot_parameters_dict.keys()), dtype=int)
        frame = output_tensor[0, 0, :, var_indices].numpy()  # [nodes, nvars]
        data_over_time.append(frame)

        # update per-variable min/max for colorbar
        frame_min = np.nanmin(frame, axis=0)
        frame_max = np.nanmax(frame, axis=0)
        vmin[:] = np.where(np.isfinite(frame_min), np.minimum(vmin, frame_min), vmin)
        vmax[:] = np.where(np.isfinite(frame_max), np.maximum(vmax, frame_max), vmax)
        return data_over_time, vmin, vmax

    @rank_zero_only
    def _generate_video_rollout(
        self,
        dataset_name: str,
        data_0: np.ndarray,
        data_over_time: list,
        plot_parameters_dict: dict[int, tuple[str, bool]],
        vmin: np.ndarray,
        vmax: np.ndarray,
        rollout_step: int,
        batch_idx: int,
        epoch: int,
        logger: pl.loggers.logger.Logger,
        animation_interval: int = 400,
    ) -> None:
        # data_0 is a single timestep sample: typically [ens?, nodes, vars] or [nodes, vars] depending on postproc
        # Normalize to [nodes, vars] for scatter
        if data_0.ndim == 3:
            data0_nodes_vars = data_0[0, :, :]
        else:
            data0_nodes_vars = data_0

        if not data_over_time:
            LOGGER.warning("No rollout frames available for video generation (dataset=%s).", dataset_name)
            return

        for idx, (variable_idx, (variable_name, _)) in enumerate(plot_parameters_dict.items()):
            fig, ax = plt.subplots(figsize=(10, 6), dpi=72)
            cmap = "viridis"

            ax, scatter_frame = get_scatter_frame(
                ax,
                data0_nodes_vars[:, variable_idx],
                self.latlons[dataset_name],
                cmap=cmap,
                vmin=vmin[idx],
                vmax=vmax[idx],
            )
            ax.set_title(f"{dataset_name}: {variable_name}")
            fig.colorbar(scatter_frame, ax=ax)

            frame_values = [data0_nodes_vars[:, variable_idx]] + [frame_data[:, idx] for frame_data in data_over_time]

            def _update(frame_id: int):
                scatter_frame.set_array(frame_values[frame_id])
                return (scatter_frame,)

            anim = animation.FuncAnimation(
                fig,
                _update,
                frames=len(frame_values),
                interval=animation_interval,
                blit=True,
            )
            self._output_gif(
                logger,
                fig,
                anim,
                epoch=epoch,
                tag=(
                    f"pred_val_animation_{dataset_name}_{variable_name}_"
                    f"rstep{rollout_step:02d}_batch{batch_idx:04d}_rank0"
                ),
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: list[torch.Tensor],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx != 0 or (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        if self.config.diagnostics.plot.asynchronous:
            LOGGER.warning("Asynchronous plotting not supported for long rollout plots; running synchronously.")

        # Gather tensors per dataset (matches BasePerBatchPlotCallback behavior)
        batch = {
            dataset_name: pl_module.allgather_batch(dataset_tensor, dataset_name)
            for dataset_name, dataset_tensor in batch.items()
        }

        # (loss, [pred_dict1, pred_dict2, ...]) expected across the codebase
        preds = output[1]
        if not isinstance(preds, list):
            preds = [preds]

        output = [
            output[0],
            [
                {
                    dataset_name: pl_module.allgather_batch(dataset_pred, dataset_name)
                    for dataset_name, dataset_pred in pred.items()
                }
                for pred in preds
            ],
        ]

        self.post_processors = copy.deepcopy(pl_module.model.post_processors)
        for dataset_name in list(self.post_processors.keys()):
            for post_processor in self.post_processors[dataset_name].processors.values():
                if hasattr(post_processor, "nan_locations"):
                    post_processor.nan_locations = pl_module.allgather_batch(post_processor.nan_locations, dataset_name)
            self.post_processors[dataset_name] = self.post_processors[dataset_name].cpu()

        precision_mapping = {"16-mixed": torch.float16, "bf16-mixed": torch.bfloat16}
        dtype = precision_mapping.get(trainer.precision)
        context = torch.autocast(device_type=list(batch.values())[0].device.type, dtype=dtype) if dtype else nullcontext()
        with context:
            self._plot(
                trainer,
                pl_module,
                self.dataset_names,
                output,
                batch,
                batch_idx,
                trainer.current_epoch,
            )

class GraphTrainableFeaturesPlot(BasePerEpochPlotCallback):
    """Visualize the node & edge trainable features defined."""

    def __init__(
        self,
        config: OmegaConf,
        dataset_names: list[str] | None = None,
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
        super().__init__(config, dataset_names=dataset_names, every_n_epochs=every_n_epochs)
        self.q_extreme_limit = config.get("quantile_edges_to_represent", 0.05)

    def get_node_trainable_tensors(self, node_attributes: NamedNodesAttributes) -> dict[str, torch.Tensor]:
        return {
            name: tt.trainable for name, tt in node_attributes.trainable_tensors.items() if tt.trainable is not None
        }

    @staticmethod
    def _resolve_edge_provider(provider: Any, dataset_name: str) -> Any:
        if provider is None:
            return None
        if isinstance(provider, (dict, torch.nn.ModuleDict)):
            if dataset_name in provider:
                return provider[dataset_name]
            return None
        return provider

    @staticmethod
    def _has_trainable_edge_params(provider: Any) -> bool:
        if provider is None:
            return False
        trainable_module = getattr(provider, "trainable", None)
        if trainable_module is None:
            return False
        # Graph providers has TrainableTensor -> .trainable;
        # parameter is nested as .trainable.trainable.
        trainable_parameter = getattr(trainable_module, "trainable", None)
        return trainable_parameter is not None

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
        node_trainable_tensors = self.get_node_trainable_tensors(model.node_attributes)

        for dataset_name in dataset_names:
            if dataset_name in node_trainable_tensors and node_trainable_tensors[dataset_name] is not None:
                fig = plot_graph_node_features(
                    model.node_attributes,
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
                    model.node_attributes,
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
        super().__init__(config, every_n_batches=every_n_batches, dataset_names=dataset_names)
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
        output_times: int,
    ) -> None:
        logger = trainer.logger
        _ = batch_idx

        if self.latlons is None:
            self.latlons = {}

        for dataset_name in dataset_names:

            data_indices = pl_module.data_indices[dataset_name]
            parameter_names = list[str](data_indices.model.output.name_to_index.keys())
            parameter_positions = list[int](data_indices.model.output.name_to_index.values())
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

            if pl_module.task_type != "forecaster":
                output_times = 1

            for rollout_step in range(output_times):
                y_hat = outputs[1][rollout_step][dataset_name]
                start = pl_module.n_step_input + rollout_step * pl_module.n_step_output
                y_time = batch[dataset_name].narrow(1, start, pl_module.n_step_output)
                var_idx = data_indices.data.output.full.to(device=batch[dataset_name].device)
                y_true = y_time.index_select(-1, var_idx)
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
                    tag=f"loss_{dataset_name}_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
                    exp_log_tag=f"loss_sample_{dataset_name}_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
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
                        dataset,
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

    def __init__(
        self,
        config: BaseSchema,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
    ) -> None:

        super().__init__(config, every_n_batches=every_n_batches, dataset_names=dataset_names)

        # Build focus mask
        self.focus_mask = build_spatial_mask(
            node_attribute_name=focus_area.get("mask_attr_name", None) if focus_area is not None else None,
            latlon_bbox=focus_area.get("latlon_bbox", None) if focus_area is not None else None,
            name=focus_area.get("name", None) if focus_area is not None else None,
        )

    def process(
        self,
        pl_module: pl.LightningModule,
        dataset_name: str,
        outputs: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
        batch: dict[str, torch.Tensor],
        output_times: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process the data and output tensors for plotting one dataset specified by dataset_name.

        Parameters
        ----------
        pl_module : pl.LightningModule
            The LightningModule instance
        dataset_name : str
            The name of the dataset to process
        outputs : tuple[torch.Tensor, list[dict[str, torch.Tensor]]]
            The outputs from the model. The second element must be a list of dicts
            (one per outer step). Tasks with a single step (e.g. diffusion, multi-out
            interpolator) must return [y_pred] so that ``for x in outputs[1]``
            iterates over steps; if they return the dict directly, iteration would
            be over dataset names and indexing would fail.
        batch : dict[str, torch.Tensor]
            The batch of data
        output_times : int
            Number of outer steps for plotting (rollout steps for forecaster, interp times for interpolator).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The data and output tensors for plotting
        """
        if self.latlons is None:
            self.latlons = {}

        if dataset_name not in self.latlons:
            self.latlons[dataset_name] = pl_module.model.model._graph_data[dataset_name].x.detach()
            self.latlons[dataset_name] = np.rad2deg(self.latlons[dataset_name].cpu().numpy())

        # All tasks return (loss, metrics, list of per-step dicts) from _step; on_validation_batch_end enforces list.
        assert isinstance(
            outputs[1],
            list,
        ), "outputs[1] must be a list of per-step dicts."

        # prepare input and output tensors for plotting one dataset specified by dataset_name
        # total_targets: forecaster has n_step_output per rollout step; interpolator has 1 per step
        total_targets = output_times
        if pl_module.task_type == "forecaster":
            total_targets *= pl_module.n_step_output

        input_tensor = (
            batch[dataset_name][
                :,
                pl_module.n_step_input - 1 : pl_module.n_step_input + total_targets + 1,
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

        if pl_module.task_type == "time-interpolator" and output_tensor.ndim == 5 and output_tensor.shape[0] == 1:
            # Multi-out interpolator: rollouts are packed in the time dimension.
            output_tensor = output_tensor.squeeze(0)
        output_tensor = (
            pl_module.output_mask[dataset_name].apply(output_tensor, dim=pl_module.grid_dim, fill_value=np.nan).numpy()
        )
        data[1:, ...] = pl_module.output_mask[dataset_name].apply(
            data[1:, ...],
            dim=pl_module.grid_dim,
            fill_value=np.nan,
        )
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
        output_steps: int,
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict[str, Colormap] | None = None,
        per_sample: int = 6,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
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
        output_steps : int
            Max number of output steps to plot per rollout in forecast mode
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
        super().__init__(config, dataset_names=dataset_names, every_n_batches=every_n_batches, focus_area=focus_area)
        self.sample_idx = sample_idx
        self.parameters = parameters

        self.precip_and_related_fields = precip_and_related_fields
        self.accumulation_levels_plot = accumulation_levels_plot
        self.output_steps = output_steps
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
        output_times: int,
    ) -> None:
        logger = trainer.logger

        for dataset_name in dataset_names:
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

            data, output_tensor = self.process(pl_module, dataset_name, outputs, batch, output_times)

            local_rank = pl_module.local_rank

            # Apply spatial mask
            latlons, data, output_tensor = self.focus_mask.apply(
                pl_module.model.model._graph_data,
                self.latlons[dataset_name],
                data,
                output_tensor,
            )

            if pl_module.task_type == "forecaster" and output_tensor.ndim == 5 and output_tensor.shape[0] == 1:
                max_out_steps = min(pl_module.n_step_output, self.output_steps)
                for rollout_step in range(output_times):
                    init_step = pl_module.get_init_step(rollout_step)
                    for out_step in range(max_out_steps):
                        truth_idx = rollout_step * pl_module.n_step_output + out_step + 1
                        fig = plot_predicted_multilevel_flat_sample(
                            plot_parameters_dict,
                            self.per_sample,
                            latlons,
                            self.accumulation_levels_plot,
                            data[init_step, ...].squeeze(),
                            data[truth_idx, ...].squeeze(),
                            output_tensor[rollout_step, out_step, ...],
                            datashader=self.datashader_plotting,
                            precip_and_related_fields=self.precip_and_related_fields,
                            colormaps=self.colormaps,
                        )

                        self._output_figure(
                            logger,
                            fig,
                            epoch=epoch,
                            tag=(
                                "pred_val_sample_"
                                f"{dataset_name}_rstep{rollout_step:02d}_out{out_step:02d}_"
                                f"batch{batch_idx:04d}_rank{local_rank:01d}{self.focus_mask.tag}"
                            ),
                            exp_log_tag=(
                                "val_pred_sample_"
                                f"{dataset_name}_rstep{rollout_step:02d}_out{out_step:02d}_"
                                f"rank{local_rank:01d}{self.focus_mask.tag}"
                            ),
                        )
            else:
                for rollout_step in range(output_times):
                    interp_step = rollout_step + 1
                    init_step = pl_module.get_init_step(rollout_step)
                    fig = plot_predicted_multilevel_flat_sample(
                        plot_parameters_dict,
                        self.per_sample,
                        latlons,
                        self.accumulation_levels_plot,
                        data[init_step, ...].squeeze(),
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
                        tag=f"pred_val_sample_{dataset_name}_istep{interp_step:02d}_batch{batch_idx:04d}_rank{local_rank:01d}{self.focus_mask.tag}",
                        exp_log_tag=f"val_pred_sample_{dataset_name}_istep{interp_step:02d}_rank{local_rank:01d}{self.focus_mask.tag}",
                    )


class MultiStepPlot(BasePlotAdditionalMetrics):
    """Plots/animates all available output steps from one forward pass (no rollout)."""

    def __init__(
        self,
        config: OmegaConf,
        sample_idx: int,
        parameters: list[str],
        accumulation_levels_plot: list[float],
        output_steps: int | None = None,
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict[str, Colormap] | None = None,
        per_sample: int = 6,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
        video: bool = True,
        animation_interval: int = 350,
        video_fps: int = 8,
        frame_dpi: int = 140,
        save_frames: bool = False,
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(config, dataset_names=dataset_names, every_n_batches=every_n_batches, focus_area=focus_area)
        self.sample_idx = sample_idx
        self.parameters = parameters
        self.accumulation_levels_plot = accumulation_levels_plot
        self.precip_and_related_fields = precip_and_related_fields
        self.colormaps = colormaps
        self.per_sample = per_sample
        self.video = video
        self.animation_interval = animation_interval
        self.video_fps = video_fps
        self.frame_dpi = frame_dpi
        self.save_frames = save_frames
        self.max_steps = max_steps if max_steps is not None else output_steps

    @staticmethod
    def _flatten_step_tensor(output_tensor: np.ndarray) -> np.ndarray:
        arr = np.asarray(output_tensor)
        if arr.ndim < 3:
            msg = f"Expected at least 3 dimensions for output_tensor, got shape {arr.shape}."
            raise ValueError(msg)
        if arr.ndim == 3:
            return arr
        return arr.reshape(-1, arr.shape[-2], arr.shape[-1])

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
        output_times: int,
    ) -> None:
        logger = trainer.logger
        local_rank = pl_module.local_rank

        for dataset_name in dataset_names:
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
                if name in pl_module.data_indices[dataset_name].model.output.name_to_index
            }
            if not plot_parameters_dict:
                LOGGER.warning("MultiStepPlot: no parameters matched for dataset '%s'.", dataset_name)
                continue

            data, output_tensor = self.process(pl_module, dataset_name, outputs, batch, output_times)

            latlons, data, output_tensor = self.focus_mask.apply(
                pl_module.model.model._graph_data,
                self.latlons[dataset_name],
                data,
                output_tensor,
            )

            pred_steps = self._flatten_step_tensor(output_tensor)
            max_by_truth = max(data.shape[0] - 1, 0)
            n_steps = min(pred_steps.shape[0], max_by_truth)
            if self.max_steps is not None:
                n_steps = min(n_steps, self.max_steps)
            if n_steps <= 0:
                LOGGER.warning("MultiStepPlot: no valid steps to plot for dataset '%s'.", dataset_name)
                continue

            frame_images = []
            for step in range(n_steps):
                fig_step = plot_predicted_multilevel_flat_sample(
                    plot_parameters_dict,
                    self.per_sample,
                    latlons,
                    self.accumulation_levels_plot,
                    data[0, ...].squeeze(),
                    data[step + 1, ...].squeeze(),
                    pred_steps[step, ...],
                    datashader=self.datashader_plotting,
                    precip_and_related_fields=self.precip_and_related_fields,
                    colormaps=self.colormaps,
                    dpi=self.frame_dpi,
                )

                if self.save_frames:
                    self._output_figure(
                        logger,
                        fig_step,
                        epoch=epoch,
                        tag=(
                            "pred_val_multistep_"
                            f"{dataset_name}_step{step + 1:02d}_batch{batch_idx:04d}_"
                            f"rank{local_rank:01d}{self.focus_mask.tag}"
                        ),
                        exp_log_tag=(
                            "val_pred_multistep_"
                            f"{dataset_name}_step{step + 1:02d}_rank{local_rank:01d}{self.focus_mask.tag}"
                        ),
                    )
                elif self.video:
                    fig_step.canvas.draw()
                    frame_images.append(np.array(fig_step.canvas.renderer.buffer_rgba()))
                    plt.close(fig_step)
                else:
                    plt.close(fig_step)

            if self.video and frame_images:
                frame_h, frame_w = frame_images[0].shape[:2]
                dpi = self.frame_dpi
                fig_anim, ax = plt.subplots(figsize=(frame_w / dpi, frame_h / dpi), dpi=dpi)
                ax.axis("off")
                ax.set_position([0.0, 0.0, 1.0, 1.0])
                im = ax.imshow(frame_images[0], animated=True, aspect="auto", interpolation="nearest")

                def _update(frame_id: int):
                    im.set_data(frame_images[frame_id])
                    return (im,)

                anim = animation.FuncAnimation(
                    fig_anim,
                    _update,
                    frames=len(frame_images),
                    interval=self.animation_interval,
                    blit=True,
                )
                self._output_gif(
                    logger,
                    fig_anim,
                    anim,
                    epoch=epoch,
                    tag=(
                        "pred_val_multistep_animation_"
                        f"{dataset_name}_steps{n_steps:02d}_batch{batch_idx:04d}_"
                        f"rank{local_rank:01d}{self.focus_mask.tag}"
                    ),
                    fps=self.video_fps,
                )


class TimeInterpolatorMultiStepPlot(MultiStepPlot):
    """Backward-compatible alias of MultiStepPlot."""


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
        output_steps: int,
        min_delta: float | None = None,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
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
        output_steps : int
            Max number of output steps to plot per rollout in forecast mode
        every_n_batches : int | None, optional
            Override for batch frequency, by default None
        """
        super().__init__(config, dataset_names=dataset_names, every_n_batches=every_n_batches, focus_area=focus_area)
        self.sample_idx = sample_idx
        self.parameters = parameters
        self.output_steps = output_steps
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
        output_times: int,
    ) -> None:
        logger = trainer.logger

        local_rank = pl_module.local_rank
        for dataset_name in dataset_names:
            data, output_tensor = self.process(pl_module, dataset_name, outputs, batch, output_times)

            # Apply spatial mask
            latlons, data, output_tensor = self.focus_mask.apply(
                pl_module.model.model._graph_data,
                self.latlons[dataset_name],
                data,
                output_tensor,
            )

        for dataset_name in dataset_names:
            data, output_tensor = self.process(pl_module, dataset_name, outputs, batch, output_times)

            # Build dictionary of indices and parameters to be plotted
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

            if pl_module.task_type == "forecaster":
                max_out_steps = min(pl_module.n_step_output, self.output_steps)
                for rollout_step in range(output_times):
                    init_step = pl_module.get_init_step(rollout_step)
                    for out_step in range(max_out_steps):
                        truth_idx = rollout_step * pl_module.n_step_output + out_step + 1
                        fig = plot_power_spectrum(
                            plot_parameters_dict_spectrum,
                            latlons,
                            data[init_step, ...].squeeze(),
                            data[truth_idx, ...].squeeze(),
                            output_tensor[rollout_step, out_step, ...],
                            min_delta=self.min_delta,
                        )

                        self._output_figure(
                            logger,
                            fig,
                            epoch=epoch,
                            tag=(
                                "pred_val_spec_"
                                f"{dataset_name}_rstep_{rollout_step:02d}_out{out_step:02d}_"
                                f"batch{batch_idx:04d}_rank{local_rank:01d}{self.focus_mask.tag}"
                            ),
                            exp_log_tag=(
                                "pred_val_spec_"
                                f"{dataset_name}_rstep_{rollout_step:02d}_out{out_step:02d}_"
                                f"rank{local_rank:01d}{self.focus_mask.tag}"
                            ),
                        )
            else:
                for rollout_step in range(output_times):
                    init_step = pl_module.get_init_step(rollout_step)
                    interp_step = rollout_step + 1
                    fig = plot_power_spectrum(
                        plot_parameters_dict_spectrum,
                        latlons,
                        data[init_step, ...].squeeze(),
                        data[rollout_step + 1, ...].squeeze(),
                        output_tensor[rollout_step, ...],
                        min_delta=self.min_delta,
                    )

                    self._output_figure(
                        logger,
                        fig,
                        epoch=epoch,
                        tag=f"pred_val_spec_{dataset_name}_istep_{interp_step:02d}_batch{batch_idx:04d}_rank{local_rank:01d}{self.focus_mask.tag}",
                        exp_log_tag=f"pred_val_spec_{dataset_name}_istep_{interp_step:02d}_rank{local_rank:01d}{self.focus_mask.tag}",
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
        output_steps: int,
        precip_and_related_fields: list[str] | None = None,
        log_scale: bool = False,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
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
        output_steps : int
            Max number of output steps to plot per rollout in forecast mode
        precip_and_related_fields : list[str] | None, optional
            Precip variable names, by default None
        every_n_batches : int | None, optional
            Override for batch frequency, by default None

        """
        super().__init__(config, dataset_names=dataset_names, every_n_batches=every_n_batches, focus_area=focus_area)
        self.sample_idx = sample_idx
        self.parameters = parameters
        self.output_steps = output_steps
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
        output_times: int,
    ) -> None:
        logger = trainer.logger

        local_rank = pl_module.local_rank

        for dataset_name in dataset_names:

            data, output_tensor = self.process(pl_module, dataset_name, outputs, batch, output_times)

            # Build dictionary of indices and parameters to be plotted
            diagnostics = (
                []
                if self.config.data.datasets[dataset_name].diagnostic is None
                else self.config.data.datasets[dataset_name].diagnostic
            )
            # Apply spatial mask
            _, data, output_tensor = self.focus_mask.apply(
                pl_module.model.model._graph_data,
                self.latlons[dataset_name],
                data,
                output_tensor,
            )

            plot_parameters_dict_histogram = {
                pl_module.data_indices[dataset_name].model.output.name_to_index[name]: (
                    name,
                    name not in diagnostics,
                )
                for name in self.parameters
            }

            if pl_module.task_type == "forecaster":
                max_out_steps = min(pl_module.n_step_output, self.output_steps)
                for rollout_step in range(output_times):
                    init_step = pl_module.get_init_step(rollout_step)
                    for out_step in range(max_out_steps):
                        truth_idx = rollout_step * pl_module.n_step_output + out_step + 1
                        fig = plot_histogram(
                            plot_parameters_dict_histogram,
                            data[init_step, ...].squeeze(),
                            data[truth_idx, ...].squeeze(),
                            output_tensor[rollout_step, out_step, ...],
                            self.precip_and_related_fields,
                            self.log_scale,
                        )

                        self._output_figure(
                            logger,
                            fig,
                            epoch=epoch,
                            tag=(
                                "pred_val_histo_"
                                f"{dataset_name}_rstep_{rollout_step:02d}_out{out_step:02d}_"
                                f"batch{batch_idx:04d}_rank{local_rank:01d}{self.focus_mask.tag}"
                            ),
                            exp_log_tag=(
                                "pred_val_histo_"
                                f"{dataset_name}_rstep_{rollout_step:02d}_out{out_step:02d}_"
                                f"rank{local_rank:01d}{self.focus_mask.tag}"
                            ),
                        )
            else:
                for rollout_step in range(output_times):
                    init_step = pl_module.get_init_step(rollout_step)
                    interp_step = rollout_step + 1
                    fig = plot_histogram(
                        plot_parameters_dict_histogram,
                        data[init_step, ...].squeeze(),
                        data[rollout_step + 1, ...].squeeze(),
                        output_tensor[rollout_step, ...],
                        self.precip_and_related_fields,
                        self.log_scale,
                    )

                    self._output_figure(
                        logger,
                        fig,
                        epoch=epoch,
                        tag=f"pred_val_histo_{dataset_name}_istep_{interp_step:02d}_batch{batch_idx:04d}_rank{local_rank:01d}{self.focus_mask.tag}",
                        exp_log_tag=f"pred_val_histo_{dataset_name}_istep_{interp_step:02d}_rank{local_rank:01d}{self.focus_mask.tag}",
                    )
