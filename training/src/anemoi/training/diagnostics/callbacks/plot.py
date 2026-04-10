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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from anemoi.training.diagnostics.evaluation.plotting.plotter import GraphFeaturePlotter
from anemoi.training.diagnostics.evaluation.plotting.plotter import HistogramPlotter
from anemoi.training.diagnostics.evaluation.plotting.plotter import LossPlotter
from anemoi.training.diagnostics.evaluation.plotting.plotter import SamplePlotter
from anemoi.training.diagnostics.evaluation.plotting.plotter import SpectrumPlotter
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.utils import reduce_to_last_dim
from anemoi.training.utils.custom_colormaps import CustomColormap

LOGGER = logging.getLogger(__name__)


@dataclass
class PlottingSettings:
    """Settings for plotting callbacks.

    TODO: Move this to Pydantic schema and inject settings in each callback.

    Parameters
    ----------
    datashader : bool
        Whether to use datashader for plotting
    projection_kind : str
        Map projection kind (e.g., "equirectangular")
    asynchronous : bool
        Whether to plot asynchronously in background thread
    save_basedir : str | None
        Base directory for saving plot files
    colormaps : dict | None
        Color mappings for different variables and error types
    precip_and_related_fields : list[str] | None
        List of precipitation and related field names
    focus_areas : dict | None
        Spatial focus areas for plotting (lat/lon bounding boxes)
    dataset_names : list[str] | None
        Dataset names to plot from
    """

    datashader: bool = True
    projection_kind: str = "equirectangular"
    asynchronous: bool = True
    save_basedir: str | None = None
    colormaps: dict | None = None
    precip_and_related_fields: list[str] | None = None
    focus_areas: dict | None = None
    dataset_names: list[str] | None = None


class BasePlotExecutor(ABC):
    """Abstract base class for plot executors.

    Defines the interface for scheduling plot function calls and shutting down.
    """

    def _run(self, fn: Any, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        """Call *fn* and force-exit the process on any unhandled exception.

        Logging is done before the exit so the error is visible in the training
        log. ``os._exit`` is used (rather than ``raise``) to guarantee the
        process terminates even when sanity-validation steps are in progress.
        """
        try:
            fn(trainer, *args, **kwargs)
        except BaseException:
            import os

            LOGGER.exception(traceback.format_exc())
            self.shutdown()
            os._exit(1)

    @abstractmethod
    def schedule(self, fn: Any, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        """Schedule *fn(trainer, *args, **kwargs)* for execution."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release any resources held by the executor."""


class SyncPlotExecutor(BasePlotExecutor):
    """Executes plot functions synchronously on the calling thread."""

    def schedule(self, fn: Any, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        self._run(fn, trainer, *args, **kwargs)

    def shutdown(self) -> None:
        pass


class AsyncPlotExecutor(BasePlotExecutor):
    """Manages asynchronous plot execution in a background thread with an event loop.

    Runs a single-threaded executor backed by a dedicated asyncio event loop,
    allowing plot functions to be submitted from the main thread without blocking it.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self.loop_thread = threading.Thread(target=self._start_event_loop, daemon=True)
        self.loop_thread.start()
        self._loop_ready.wait()  # block until the loop is running before returning

    def _start_event_loop(self) -> None:
        """Start the event loop in the background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(self._loop_ready.set)  # signal after the loop is running
        self.loop.run_forever()

    async def _submit(self, fn: Any, trainer: pl.Trainer, args: tuple, kwargs: dict) -> None:
        """Coroutine that runs *fn* via _run in the thread-pool executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, lambda: self._run(fn, trainer, *args, **kwargs))

    def schedule(self, fn: Any, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        """Schedule *fn(trainer, *args, **kwargs)* to run asynchronously."""
        asyncio.run_coroutine_threadsafe(self._submit(fn, trainer, args, kwargs), self.loop)

    def shutdown(self) -> None:
        """Shut down the executor and stop the event loop."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join()
        self.loop_thread._stop()
        self.loop_thread._delete()


class BasePlotCallback(Callback, ABC):
    """Factory for creating a callback that plots data to Experiment Logging."""

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the BasePlotCallback abstract base class.

        Parameters
        ----------
        dataset_names : list[str] | None, optional
            Dataset names, by default None (uses ["data"])
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults)

        """
        super().__init__()
        if plotting_settings is None:
            plotting_settings = PlottingSettings()
        self.plotting_settings = plotting_settings

        self.save_basedir = plotting_settings.save_basedir
        self.dataset_names = dataset_names if dataset_names is not None else ["data"]

        self.post_processors = None
        self.latlons = None

        self._error: BaseException = None
        self.datashader_plotting = plotting_settings.datashader
        self.projection_kind = plotting_settings.projection_kind
        self.asynchronous = plotting_settings.asynchronous

        if self.asynchronous:
            LOGGER.info("Setting up asynchronous plotting ...")
            self._executor: BasePlotExecutor = AsyncPlotExecutor()
        else:
            self._executor = SyncPlotExecutor()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check for NCCL timeout risk with asynchronous plotting."""
        del pl_module
        if self.asynchronous:
            read_group_size = trainer.strategy.read_group_size
            if read_group_size > 1:
                LOGGER.warning("Asynchronous plotting can result in NCCL timeouts with reader_group_size > 1.")

    @property
    def artifact_subfolder(self) -> str:
        """Return the artifact subfolder name for experiment logging.

        Used by MLflow to organize artifacts into per-callback folders.
        Derived automatically from the concrete callback class name.
        """
        return type(self).__name__

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

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Teardown the callback."""
        del trainer, pl_module, stage  # unused
        LOGGER.info("Teardown of the Plot Callback ...")

        LOGGER.info("waiting and shutting down the executor ...")
        self._executor.shutdown()

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

    @rank_zero_only
    def plot(
        self,
        trainer: pl.Trainer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Schedule the plot function via the executor (sync or async)."""
        self._executor.schedule(self._plot, trainer, *args, **kwargs)


class BasePerBatchPlotCallback(BasePlotCallback):
    """Base Callback for plotting at the end of each batch."""

    def __init__(
        self,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        plotting_settings: PlottingSettings | None = None,
    ):
        """Initialise the BasePerBatchPlotCallback.

        Parameters
        ----------
        every_n_batches : int, optional
            Batch Frequency to plot at, by default None (uses 750)
        dataset_names : list[str] | None, optional
            Dataset names, by default None
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults)

        """
        super().__init__(dataset_names=dataset_names, plotting_settings=plotting_settings)
        self.every_n_batches = every_n_batches or 750

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
                dataset_name: pl_module.allgather_batch(dataset_tensor, dataset_name)
                for dataset_name, dataset_tensor in batch.items()
            }
            # output: (loss, [pred_dict1, pred_dict2, ...]); all tasks return a list of per-step dicts.
            preds = output[1]
            if not isinstance(preds, list):

                raise TypeError(preds)
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
            # When running in Async mode, it might happen that in the last epoch these tensors
            # have been moved to the cpu (and then the denormalising would fail as the 'input_tensor' would be on CUDA
            # but internal ones would be on the cpu), The lines below allow to address this problem
            self.post_processors = copy.deepcopy(pl_module.model.post_processors)
            for dataset_name in self.post_processors:
                for post_processor in self.post_processors[dataset_name].processors.values():
                    if hasattr(post_processor, "nan_locations"):
                        post_processor.nan_locations = pl_module.allgather_batch(
                            post_processor.nan_locations,
                            dataset_name,
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
                **kwargs,
            )


class BasePerEpochPlotCallback(BasePlotCallback):
    """Base Callback for plotting at the end of each epoch."""

    def __init__(
        self,
        every_n_epochs: int | None = None,
        dataset_names: list[str] | None = None,
        plotting_settings: PlottingSettings | None = None,
    ):
        """Initialise the BasePerEpochPlotCallback.

        Parameters
        ----------
        every_n_epochs : int, optional
            Epoch frequency to plot at, by default None (uses 1)
        dataset_names : list[str] | None, optional
            Dataset names, by default None
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults)
        """
        super().__init__(dataset_names=dataset_names, plotting_settings=plotting_settings)
        self.every_n_epochs = every_n_epochs or 1

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)

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
                **kwargs,
            )


class GraphTrainableFeaturesPlot(BasePerEpochPlotCallback):
    """Visualize the node & edge trainable features defined."""

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        every_n_epochs: int | None = None,
        q_extreme_limit: float = 0.05,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the GraphTrainableFeaturesPlot callback.

        Parameters
        ----------
        dataset_names : list[str] | None, optional
            Dataset names, by default None
        every_n_epochs : int | None, optional
            Override for frequency to plot at, by default None
        q_extreme_limit : float, optional
            Quantile edges to represent, by default 0.05
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults)
        """
        super().__init__(
            dataset_names=dataset_names,
            every_n_epochs=every_n_epochs,
            plotting_settings=plotting_settings,
        )
        self.q_extreme_limit = q_extreme_limit
        self.plotter = GraphFeaturePlotter(
            datashader=self.datashader_plotting,
            q_extreme_limit=q_extreme_limit,
        )

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
        node_trainable_tensors = self.plotter.get_node_trainable_tensors(model.node_attributes)

        for dataset_name in dataset_names:
            if dataset_name in node_trainable_tensors and node_trainable_tensors[dataset_name] is not None:
                fig = self.plotter.plot_nodes(model.node_attributes, node_trainable_tensors)

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
            elif len(edge_trainable_modules := self.plotter.get_edge_trainable_modules(model, dataset_name)):
                fig = self.plotter.plot_edges(model.node_attributes, edge_trainable_modules)

                self._output_figure(
                    trainer.logger,
                    fig,
                    epoch=trainer.current_epoch,
                    tag=f"edge_trainable_params_{dataset_name}",
                    exp_log_tag=f"edge_trainable_params_{dataset_name}",
                )
            else:
                LOGGER.warning("There are no trainable edge attributes to plot.")


class PlotLoss(BasePerBatchPlotCallback):
    """Plots the unsqueezed loss over rollouts."""

    def __init__(
        self,
        parameter_groups: dict[dict[str, list[str]]],
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the PlotLoss callback.

        Parameters
        ----------
        parameter_groups : dict
            Dictionary with parameter groups with parameter names as keys
        every_n_batches : int, optional
            Override for batch frequency, by default None
        dataset_names : list[str] | None, optional
            Dataset names, by default None
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults)
        """
        super().__init__(
            every_n_batches=every_n_batches,
            dataset_names=dataset_names,
            plotting_settings=plotting_settings,
        )
        self.dataset_names = dataset_names if dataset_names is not None else ["data"]
        self.plotter = LossPlotter(parameter_groups=parameter_groups)

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

            if not isinstance(self.loss[dataset_name], BaseLoss):
                LOGGER.warning(
                    "Loss function must be a subclass of BaseLoss, or provide `squash`.",
                    RuntimeWarning,
                )

            adapter = pl_module.plot_adapter
            for rollout_step in range(adapter.loss_plot_times):
                y_hat = outputs[1][rollout_step][dataset_name]
                start = adapter.get_loss_plot_batch_start(rollout_step)
                y_time = batch[dataset_name].narrow(1, start, pl_module.n_step_output)
                var_idx = data_indices.data.output.full.to(device=batch[dataset_name].device)
                y_true = y_time.index_select(-1, var_idx)
                loss = reduce_to_last_dim(self.loss[dataset_name](y_hat, y_true, squash=False).detach().cpu().numpy())

                fig = self.plotter.plot(parameter_names, loss, metadata_variables=metadata_variables)

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
                for leaf_loss in self.loss[dataset].iter_leaf_losses():
                    if (
                        hasattr(leaf_loss, "scaler")
                        and hasattr(leaf_loss.scaler, "nan_mask_weights")
                        and leaf_loss.scaler.nan_mask_weights.shape[pl_module.grid_dim] != 1
                    ):
                        leaf_loss.scaler.nan_mask_weights = pl_module.allgather_batch(
                            leaf_loss.scaler.nan_mask_weights,
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
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the BasePlotAdditionalMetrics callback.

        Parameters
        ----------
        every_n_batches : int | None, optional
            Override for batch frequency, by default None
        dataset_names : list[str] | None, optional
            Dataset names, by default None
        focus_area : list[dict] | None, optional
            Focus area configuration, by default None
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults)
        """
        super().__init__(
            every_n_batches=every_n_batches,
            dataset_names=dataset_names,
            plotting_settings=plotting_settings,
        )
        self.focus_area = focus_area

    # !TODO - dicsuss with Vera where this should leave
    def process(
        self,
        pl_module: pl.LightningModule,
        dataset_name: str,
        outputs: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
        batch: dict[str, torch.Tensor],
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
        total_targets = pl_module.plot_adapter.get_total_plot_targets()

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

        output_tensor = pl_module.plot_adapter.prepare_plot_output_tensor(output_tensor)
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
        sample_idx: int,
        parameters: list[str],
        accumulation_levels_plot: list[float],
        output_steps: int,
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict[str, CustomColormap] | None = None,
        per_sample: int = 6,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the PlotSample callback.

        Parameters
        ----------
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
        colormaps : dict[str, CustomColormap] | None, optional
            Dictionary of colormaps, by default None
        per_sample : int, optional
            Number of plots per sample, by default 6
        every_n_batches : int, optional
            Batch frequency to plot at, by default None
        dataset_names : list[str] | None, optional
            Dataset names, by default None
        focus_area : list[dict] | None, optional
            Focus area configuration, by default None
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults)
        """
        super().__init__(
            dataset_names=dataset_names,
            every_n_batches=every_n_batches,
            focus_area=focus_area,
            plotting_settings=plotting_settings,
        )
        self.sample_idx = sample_idx
        self.parameters = parameters
        self.output_steps = output_steps

        LOGGER.info(
            "Using defined accumulation colormap for fields: %s",
            precip_and_related_fields,
        )
        self.plotter = SamplePlotter(
            per_sample=per_sample,
            accumulation_levels_plot=accumulation_levels_plot,
            precip_and_related_fields=precip_and_related_fields,
            colormaps=colormaps,
            datashader=self.datashader_plotting,
            projection_kind=self.projection_kind,
            focus_area=focus_area,
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
    ) -> None:
        logger = trainer.logger

        for dataset_name in dataset_names:
            # Build dictionary of indices and parameters to be plotted
            input_data = pl_module.data_indices[dataset_name].data.input.todict()
            index_to_name = {v: k for k, v in input_data["name_to_index"].items()}
            diagnostics = {index_to_name[int(i)] for i in input_data["diagnostic"]}
            plot_parameters_dict = {
                pl_module.data_indices[dataset_name].model.output.name_to_index[name]: (
                    name,
                    name in diagnostics,
                )
                for name in self.parameters
            }

            data, output_tensor = self.process(pl_module, dataset_name, outputs, batch)

            local_rank = pl_module.local_rank

            # Apply spatial mask
            latlons, data, output_tensor = self.plotter.focus_mask.apply(
                pl_module.model.model._graph_data,
                self.latlons[dataset_name],
                data,
                output_tensor,
            )

            for item in pl_module.plot_adapter.iter_plot_samples(
                data,
                output_tensor,
                pl_module.plot_adapter.output_times,
                max_out_steps=self.output_steps,
            ):
                if len(item) == 3:
                    x, y_pred, tag_suffix = item
                    y_true = None
                else:
                    x, y_true, y_pred, tag_suffix = item
                fig = self.plotter.plot(plot_parameters_dict, latlons, x, y_true, y_pred)

                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=(
                        f"pred_val_sample_{dataset_name}_{tag_suffix}_"
                        f"batch{batch_idx:04d}_rank{local_rank:01d}{self.plotter.focus_mask.tag}"
                    ),
                    exp_log_tag=(
                        f"val_pred_sample_{dataset_name}_{tag_suffix}_rank{local_rank:01d}{self.plotter.focus_mask.tag}"
                    ),
                )


class PlotSpectrum(BasePlotAdditionalMetrics):
    """Plots TP related metric comparing target and prediction.

    The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.

    - Power Spectrum
    """

    def __init__(
        self,
        sample_idx: int,
        parameters: list[str],
        output_steps: int,
        min_delta: float | None = None,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the PlotSpectrum callback.

        Parameters
        ----------
        sample_idx : int
            Sample to plot
        parameters : list[str]
            Parameters to plot
        output_steps : int
            Max number of output steps to plot per rollout in forecast mode
        every_n_batches : int | None, optional
            Override for batch frequency, by default None
        dataset_names : list[str] | None, optional
            Dataset names, by default None
        focus_area : list[dict] | None, optional
            Focus area configuration, by default None
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults)
        """
        super().__init__(
            dataset_names=dataset_names,
            every_n_batches=every_n_batches,
            focus_area=focus_area,
            plotting_settings=plotting_settings,
        )
        self.sample_idx = sample_idx
        self.parameters = parameters
        self.output_steps = output_steps
        self.plotter = SpectrumPlotter(min_delta=min_delta, focus_area=focus_area)

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
        logger = trainer.logger

        local_rank = pl_module.local_rank
        for dataset_name in dataset_names:
            data, output_tensor = self.process(pl_module, dataset_name, outputs, batch)

            # Build dictionary of indices and parameters to be plotted
            input_data = pl_module.data_indices[dataset_name].data.input.todict()
            index_to_name = {v: k for k, v in input_data["name_to_index"].items()}
            diagnostics = {index_to_name[int(i)] for i in input_data["diagnostic"]}
            plot_parameters_dict_spectrum = {
                pl_module.data_indices[dataset_name].model.output.name_to_index[name]: (
                    name,
                    name in diagnostics,
                )
                for name in self.parameters
            }

            # Apply spatial mask
            latlons, data, output_tensor = self.plotter.focus_mask.apply(
                pl_module.model.model._graph_data,
                self.latlons[dataset_name],
                data,
                output_tensor,
            )

            for item in pl_module.plot_adapter.iter_plot_samples(
                data,
                output_tensor,
                pl_module.plot_adapter.output_times,
                max_out_steps=self.output_steps,
            ):
                if len(item) == 3:
                    x, y_pred, tag_suffix = item
                    y_true = None
                else:
                    x, y_true, y_pred, tag_suffix = item
                fig = self.plotter.plot(plot_parameters_dict_spectrum, latlons, x, y_true, y_pred)

                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=(
                        f"pred_val_spec_{dataset_name}_{tag_suffix}_"
                        f"batch{batch_idx:04d}_rank{local_rank:01d}{self.plotter.focus_mask.tag}"
                    ),
                    exp_log_tag=(
                        f"pred_val_spec_{dataset_name}_{tag_suffix}_rank{local_rank:01d}{self.plotter.focus_mask.tag}"
                    ),
                )


class PlotHistogram(BasePlotAdditionalMetrics):
    """Plots histograms comparing target and prediction.

    The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.
    """

    def __init__(
        self,
        sample_idx: int,
        parameters: list[str],
        output_steps: int,
        precip_and_related_fields: list[str] | None = None,
        log_scale: bool = False,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the PlotHistogram callback.

        Parameters
        ----------
        sample_idx : int
            Sample to plot
        parameters : list[str]
            Parameters to plot
        output_steps : int
            Max number of output steps to plot per rollout in forecast mode
        precip_and_related_fields : list[str] | None, optional
            Precip variable names, by default None
        log_scale : bool, optional
            Whether to use logarithmic scale, by default False
        every_n_batches : int | None, optional
            Override for batch frequency, by default None
        dataset_names : list[str] | None, optional
            Dataset names, by default None
        focus_area : list[dict] | None, optional
            Focus area configuration, by default None
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults)

        """
        super().__init__(
            dataset_names=dataset_names,
            every_n_batches=every_n_batches,
            focus_area=focus_area,
            plotting_settings=plotting_settings,
        )
        self.sample_idx = sample_idx
        self.parameters = parameters
        self.output_steps = output_steps

        LOGGER.info(
            "Using precip histogram plotting method for fields: %s.",
            precip_and_related_fields,
        )
        self.plotter = HistogramPlotter(
            precip_and_related_fields=precip_and_related_fields,
            log_scale=log_scale,
            focus_area=focus_area,
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
    ) -> None:
        logger = trainer.logger

        local_rank = pl_module.local_rank

        for dataset_name in dataset_names:

            data, output_tensor = self.process(pl_module, dataset_name, outputs, batch)

            # Build dictionary of indices and parameters to be plotted
            input_data = pl_module.data_indices[dataset_name].data.input.todict()
            index_to_name = {v: k for k, v in input_data["name_to_index"].items()}
            diagnostics = {index_to_name[int(i)] for i in input_data["diagnostic"]}
            # Apply spatial mask
            _, data, output_tensor = self.plotter.focus_mask.apply(
                pl_module.model.model._graph_data,
                self.latlons[dataset_name],
                data,
                output_tensor,
            )

            plot_parameters_dict_histogram = {
                pl_module.data_indices[dataset_name].model.output.name_to_index[name]: (
                    name,
                    name in diagnostics,
                )
                for name in self.parameters
            }

            for item in pl_module.plot_adapter.iter_plot_samples(
                data,
                output_tensor,
                pl_module.plot_adapter.output_times,
                max_out_steps=self.output_steps,
            ):
                if len(item) == 3:
                    x, y_pred, tag_suffix = item
                    y_true = None
                else:
                    x, y_true, y_pred, tag_suffix = item
                fig = self.plotter.plot(plot_parameters_dict_histogram, x, y_true, y_pred)

                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=(
                        f"pred_val_histo_{dataset_name}_{tag_suffix}_"
                        f"batch{batch_idx:04d}_rank{local_rank:01d}{self.plotter.focus_mask.tag}"
                    ),
                    exp_log_tag=(
                        f"pred_val_histo_{dataset_name}_{tag_suffix}_rank{local_rank:01d}{self.plotter.focus_mask.tag}"
                    ),
                )
