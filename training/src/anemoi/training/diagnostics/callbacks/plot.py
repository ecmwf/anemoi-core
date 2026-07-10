# (C) Copyright 2024-2026 Anemoi contributors.
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
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel as PydanticBaseModel
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from anemoi.training.diagnostics.evaluation.geospatial.focus_area import build_spatial_mask
from anemoi.training.diagnostics.evaluation.plotting.graph import graph_plot_fn as _default_graph_plot_fn
from anemoi.training.diagnostics.evaluation.plotting.loss import loss_plot_fn as _default_loss_plot_fn
from anemoi.training.diagnostics.evaluation.plotting.model_introspection import extract_graph_inputs
from anemoi.training.diagnostics.evaluation.plotting.model_introspection import extract_loss_inputs
from anemoi.training.diagnostics.evaluation.plotting.model_introspection import extract_spatial_inputs
from anemoi.training.diagnostics.evaluation.plotting.settings import init_plot_settings
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.utils import reduce_to_last_dim
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.index_space import IndexSpace

LOGGER = logging.getLogger(__name__)


class _Unset:
    """Typed sentinel for kwargs that need to distinguish "not specified" from ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "UNSET"


# Sentinel distinguishing "members not specified, use the plot adapter's
# default" from an explicit `members=None` ("select all members").
_UNSET_MEMBERS: Any = _Unset()


class PlottingSettings(PydanticBaseModel):
    """Settings for plotting callbacks, shared across all plot callbacks in a run."""

    datashader: bool = True
    projection_kind: str = "equirectangular"
    asynchronous: bool = True
    save_basedir: str | Path | None = None
    colormaps: dict | None = None
    precip_and_related_fields: list[str] | None = None
    focus_areas: dict | None = None
    dataset_names: list[str] | None = None

    @classmethod
    def from_plot_config(cls, plot_cfg: DictConfig, save_basedir: str | Path | None) -> "PlottingSettings":
        """Construct from a validated diagnostics.plot config node."""
        projection_kind = plot_cfg.projection_kind
        if plot_cfg.datashader and projection_kind != "equirectangular":
            LOGGER.warning(
                "datashader=True requires equirectangular projection; ignoring projection_kind=%s",
                projection_kind,
            )
            projection_kind = "equirectangular"
        return cls(
            datashader=plot_cfg.datashader,
            projection_kind=projection_kind,
            asynchronous=plot_cfg.asynchronous,
            save_basedir=save_basedir,
            colormaps=OmegaConf.select(plot_cfg, "colormaps", default=None),
            precip_and_related_fields=OmegaConf.select(plot_cfg, "precip_and_related_fields", default=None),
            focus_areas=OmegaConf.select(plot_cfg, "focus_areas", default=None),
            dataset_names=OmegaConf.select(plot_cfg, "datasets_to_plot", default=None),
        )


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
            self.shutdown(wait=False)
            os._exit(1)

    @abstractmethod
    def schedule(self, fn: Any, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        """Schedule *fn(trainer, *args, **kwargs)* for execution."""

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Release any resources held by the executor."""


class SyncPlotExecutor(BasePlotExecutor):
    """Executes plot functions synchronously on the calling thread."""

    def schedule(self, fn: Any, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        self._run(fn, trainer, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        pass


class AsyncPlotExecutor(BasePlotExecutor):
    """Manages asynchronous plot execution in a background thread with an event loop.

    Runs a single-threaded executor backed by a dedicated asyncio event loop,
    allowing plot functions to be submitted from the main thread without blocking it.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(target=self._start_event_loop, daemon=True)
        self._loop_thread.start()
        self._loop_ready.wait()  # block until the loop is running before returning

    def _start_event_loop(self) -> None:
        """Start the event loop in the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._loop_ready.set)  # signal after the loop is running
        self._loop.run_forever()

    async def _submit(self, fn: Any, trainer: pl.Trainer, args: tuple, kwargs: dict) -> None:
        """Coroutine that runs *fn* via _run in the thread-pool executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, lambda: self._run(fn, trainer, *args, **kwargs))

    def schedule(self, fn: Any, trainer: pl.Trainer, *args: Any, **kwargs: Any) -> None:
        """Schedule *fn(trainer, *args, **kwargs)* to run asynchronously."""
        asyncio.run_coroutine_threadsafe(self._submit(fn, trainer, args, kwargs), self._loop)

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the executor and stop the event loop.

        Parameters
        ----------
        wait : bool
            If True (default), block until all pending plot tasks finish before
            stopping the loop — prevents "Task was destroyed but it is pending!"
            warnings on normal teardown.  Set to False when called from an error
            handler running on the background thread itself (to avoid deadlock).
        """
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if wait:
            self._loop_thread.join()


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

        init_plot_settings()
        # `plotting_settings` is the single source of truth for datashader,
        # projection_kind and colormaps — access it directly rather than
        # duplicating attributes here.
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

            if logger and logger.logger_name == "wandb":
                LOGGER.warning("Saving gif animations not tested for wandb.")

            if logger and logger.logger_name == "mlflow":
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

    def _plot_kwargs_from_output(
        self,
        pl_module: pl.LightningModule,
        output: TrainingStepOutput,
    ) -> dict[str, Any]:
        """Extra plot keyword arguments from step output."""
        del pl_module, output
        return {}

    def _prepare_batch(
        self,
        pl_module: pl.LightningModule,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Hook for subclasses to transform the batch before plotting.

        Default: no-op. Override to inject callback-specific batch preparation
        (e.g. :class:`PlotLoss` uses ``pl_module.plot_adapter.prepare_loss_batch``).
        """
        del pl_module
        return batch

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: TrainingStepOutput,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        **kwargs,
    ) -> None:
        if batch_idx % self.every_n_batches == 0:

            batch = self._prepare_batch(pl_module, batch)
            # gather tensors if necessary
            batch = {
                dataset_name: pl_module.allgather_batch(dataset_tensor, dataset_name)
                for dataset_name, dataset_tensor in batch.items()
            }
            preds = output.predictions
            if not isinstance(preds, list):

                raise TypeError(preds)
            gathered_predictions = [
                {
                    dataset_name: pl_module.allgather_batch(dataset_pred, dataset_name)
                    for dataset_name, dataset_pred in pred.items()
                }
                for pred in preds
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

            plot_kwargs = self._plot_kwargs_from_output(pl_module, output)
            output = TrainingStepOutput(
                loss=output.loss,
                metrics=output.metrics,
                predictions=gathered_predictions,
            )
            self.plot(
                trainer,
                pl_module,
                self.dataset_names,
                output,
                batch,
                batch_idx,
                epoch=trainer.current_epoch,
                processed_cache={},
                **plot_kwargs,
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
    """Visualize the node & edge trainable features defined.

    The visualization function is supplied via ``plot_fn`` and follows the
    same pluggable pattern as :class:`SpatialMapPlot` and :class:`PlotLoss`.
    ``plot_fn`` must yield ``(figure, tag)`` pairs.

    The callback resolves the underlying model and forwards **only**
    already-extracted graph artifacts to ``plot_fn`` (never the raw model
    object)::

        fn(*, dataset_name, node_attributes, node_trainable_tensors,
           edge_trainable_modules, q_extreme_limit, settings, **kwargs)
            -> Iterable[tuple[Figure, str]]

    Hierarchical models are handled here (``edge_trainable_modules={}``);
    the plot function does not need to check the model type.

    The default is
    :func:`anemoi.training.diagnostics.evaluation.plotting.graph.graph_plot_fn`.
    """

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        every_n_epochs: int | None = None,
        q_extreme_limit: float = 0.05,
        plot_fn: Any = None,
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
        plot_fn : Callable, optional
            Plug-in plot function yielding ``(figure, tag)`` pairs. Typically
            a Hydra ``functools.partial`` (``_partial_: true``). Defaults to
            :func:`graph_plot_fn`.
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults)
        """
        super().__init__(
            dataset_names=dataset_names,
            every_n_epochs=every_n_epochs,
            plotting_settings=plotting_settings,
        )
        self.q_extreme_limit = q_extreme_limit
        self.plot_fn = plot_fn if plot_fn is not None else _default_graph_plot_fn

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        epoch: int,
    ) -> None:
        _ = epoch

        for dataset_name in dataset_names:
            graph_inputs = extract_graph_inputs(pl_module, dataset_name)
            for fig, tag in self.plot_fn(
                **graph_inputs,
                q_extreme_limit=self.q_extreme_limit,
                settings=self.plotting_settings,
            ):
                self._output_figure(
                    trainer.logger,
                    fig,
                    epoch=trainer.current_epoch,
                    tag=tag,
                    exp_log_tag=tag,
                )


class PlotLoss(BasePerBatchPlotCallback):
    """Plots the unsqueezed loss over rollouts.

    The visualization function is supplied via ``plot_fn`` following the same
    pluggable pattern as :class:`SpatialMapPlot`. It receives the raw
    per-variable loss array plus the parameter naming/grouping context, and
    is free to decide how (or whether) to sort, group, colour and render::

        fn(loss, *, parameter_names, parameter_groups, metadata_variables,
           step_index, metric_name, task_kwargs, settings, **kwargs)
            -> matplotlib.figure.Figure

    All keyword arguments except ``loss`` and ``parameter_names`` are
    optional context: plug-in functions are expected to accept ``**kwargs``
    and only bind what they need (e.g. a per-variable bar chart uses
    ``parameter_groups``; a per-step title uses ``step_index`` /
    ``metric_name``).

    The default is
    :func:`anemoi.training.diagnostics.evaluation.plotting.loss.loss_plot_fn`,
    which reproduces the historic grouped bar-chart via
    :func:`argsort_variablename_variablelevel` +
    :func:`sort_and_color_by_parameter_group` + :func:`plot_loss`.
    """

    def __init__(
        self,
        parameter_groups: dict[dict[str, list[str]]],
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        plot_fn: Any = None,
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
        plot_fn : Callable, optional
            Plug-in plot function. Typically a Hydra ``functools.partial``
            (``_partial_: true``). Defaults to :func:`loss_plot_fn`.
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults)
        """
        super().__init__(
            every_n_batches=every_n_batches,
            dataset_names=dataset_names,
            plotting_settings=plotting_settings,
        )
        self.parameter_groups = parameter_groups
        self.dataset_names = dataset_names if dataset_names is not None else ["data"]
        if self.parameter_groups is None:
            self.parameter_groups = {}
        self.plot_fn = plot_fn if plot_fn is not None else _default_loss_plot_fn

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        outputs: TrainingStepOutput,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
        processed_cache: dict | None = None,
    ) -> None:
        logger = trainer.logger
        _ = batch_idx, processed_cache

        if self.latlons is None:
            self.latlons = {}

        for dataset_name in dataset_names:
            loss_inputs = extract_loss_inputs(pl_module, dataset_name, self.parameter_groups)

            if not isinstance(self.loss[dataset_name], BaseLoss):
                LOGGER.warning(
                    "Loss function must be a subclass of BaseLoss, or provide `squash`.",
                    RuntimeWarning,
                )

            for i, task_kwargs in enumerate(pl_module.task.steps("validation")):
                y_hat = outputs.predictions[i][dataset_name]
                y_true = pl_module.task.get_targets(
                    batch={dataset_name: batch[dataset_name]},
                    data_indices=pl_module.data_indices,
                    **task_kwargs,
                )[dataset_name]
                loss = reduce_to_last_dim(
                    self.loss[dataset_name](
                        y_hat,
                        y_true,
                        pred_layout=IndexSpace.MODEL_OUTPUT,
                        target_layout=IndexSpace.DATA_FULL,
                        squash=False,
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                )

                metric_name = pl_module.task.get_metric_name(**task_kwargs)
                fig = self.plot_fn(
                    loss,
                    **loss_inputs,
                    step_index=i,
                    metric_name=metric_name,
                    task_kwargs=task_kwargs,
                    settings=self.plotting_settings,
                )

                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=f"loss_{dataset_name}{metric_name}_rank{pl_module.local_rank:01d}",
                    exp_log_tag=f"loss_sample_{dataset_name}{metric_name}_rank{pl_module.local_rank:01d}",
                )

    def _prepare_batch(
        self,
        pl_module: pl.LightningModule,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Snapshot loss + gather nan-mask weights, then delegate batch prep to the plot adapter."""
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

        return pl_module.plot_adapter.prepare_loss_batch(batch)


class BasePlotAdditionalMetrics(BasePerBatchPlotCallback):
    """Base processing class for additional metrics."""

    def __init__(
        self,
        sample_idx: int = 0,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the BasePlotAdditionalMetrics callback.

        Parameters
        ----------
        sample_idx : int, optional
            Index of the sample within the batch to plot. Consumed by
            :meth:`process` and :meth:`process_output_tensor`. Default 0.
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
        self.sample_idx = sample_idx

        # Build focus mask
        self.focus_mask = build_spatial_mask(
            node_attribute_name=focus_area.get("mask_attr_name", None) if focus_area is not None else None,
            latlon_bbox=focus_area.get("latlon_bbox", None) if focus_area is not None else None,
            name=focus_area.get("name", None) if focus_area is not None else None,
        )

    def _gather_auxiliary(
        self,
        pl_module: pl.LightningModule,
        output: TrainingStepOutput,
    ) -> dict[str, torch.Tensor] | None:
        """Return an allgathered ``auxiliary_output`` from *output*, or ``None`` if absent."""
        auxiliary_output = output.plot_kwargs.get("auxiliary_output")
        if auxiliary_output is None:
            return None
        return {
            dataset_name: pl_module.allgather_batch(dataset_tensor, dataset_name)
            for dataset_name, dataset_tensor in auxiliary_output.items()
        }

    def process(
        self,
        pl_module: pl.LightningModule,
        dataset_name: str,
        outputs: TrainingStepOutput,
        batch: dict[str, torch.Tensor],
        members: Any = _UNSET_MEMBERS,
        processed_cache: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process the data and output tensors for plotting one dataset specified by dataset_name.

        Results are cached in ``processed_cache`` when provided, keyed by ``(dataset_name, members)``.
        Subsequent calls with the same key return the cached result without recomputation, avoiding
        redundant post-processing when multiple callbacks process the same batch.

        Parameters
        ----------
        pl_module : pl.LightningModule
            The LightningModule instance.
        dataset_name : str
            The name of the dataset to process.
        outputs : TrainingStepOutput
            The outputs from the model. The predictions must be a list of dicts
            (one per outer step).
        batch : dict[str, torch.Tensor]
            The batch of data.
        members : int | list[int] | None, optional
            Ensemble members to select. Only used when the plot adapter is ensemble-aware.
            If not given, defaults to ``pl_module.plot_adapter.default_plot_members``
            (member 0 for non-ensemble adapters, all members for ensemble adapters).
            Pass ``None`` explicitly to select all members regardless of adapter default.
        processed_cache : dict | None, optional
            Optional dict for caching computed results across callbacks within the same batch.
            Should be created fresh per batch (e.g. in ``on_validation_batch_end``) so that
            it is not shared across batches. Safe for async execution since each batch
            invocation captures its own dict. Default is None (no caching).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The post-processed input data and output tensor for plotting.
        """
        if isinstance(members, _Unset):
            members = pl_module.plot_adapter.default_plot_members

        if self.latlons is None:
            self.latlons = {}

        if dataset_name not in self.latlons:
            self.latlons[dataset_name] = pl_module.model.model._graph_data[dataset_name].x.detach()
            self.latlons[dataset_name] = np.rad2deg(self.latlons[dataset_name].cpu().numpy())

        assert isinstance(
            outputs.predictions,
            list,
        ), "outputs.predictions must be a list of per-step dicts."

        members_key = tuple(members) if isinstance(members, list) else members
        cache_key = (dataset_name, members_key)
        if processed_cache is not None and cache_key in processed_cache:
            return processed_cache[cache_key]

        # prepare input and output tensors for plotting one dataset specified by dataset_name
        feature_indices = pl_module.data_indices[dataset_name].data.output.full

        input_tensor = batch[dataset_name].detach().cpu()[..., feature_indices]

        data = self.post_processors[dataset_name](input_tensor)[self.sample_idx]
        output_tensor = self.process_output_tensor(pl_module, dataset_name, outputs.predictions, members=members)

        data[1:, ...] = pl_module.output_mask[dataset_name].apply(
            data[1:, ...],
            dim=pl_module.grid_dim,
            fill_value=np.nan,
        )
        data = data.numpy()

        result = (data, output_tensor)
        if processed_cache is not None:
            processed_cache[cache_key] = result
        return result

    def process_output_tensor(
        self,
        pl_module: pl.LightningModule,
        dataset_name: str,
        outputs: list[dict[str, torch.Tensor]],
        members: int | list[int] | None = 0,
    ) -> np.ndarray:
        """Post-process and mask per-step output tensors for plotting."""
        output_tensor = torch.cat(
            tuple(
                pl_module.plot_adapter.select_members(
                    self.post_processors[dataset_name](x[dataset_name][:, ...].detach().cpu(), in_place=False)[
                        self.sample_idx : self.sample_idx + 1
                    ],
                    members,
                )
                for x in outputs
            ),
        )

        output_tensor = pl_module.plot_adapter.prepare_plot_output_tensor(output_tensor)
        return (
            pl_module.output_mask[dataset_name].apply(output_tensor, dim=pl_module.grid_dim, fill_value=np.nan).numpy()
        )



class SpatialMapPlot(BasePlotAdditionalMetrics):
    """Generic, config-driven spatial-map plot callback.

    Handles the shared plumbing (per-dataset loop, ``process()``, focus mask,
    ``iter_plot_samples``, figure output, tag naming) for any plot function
    conforming to the ``SpatialMapPlot`` ``plot_fn`` contract (see
    ``docs/modules/diagnostics.rst``). New spatial plots can be added by
    writing that function and pointing to it from YAML — no callback subclass
    or schema entry required.

    Example
    -------
    .. code-block:: yaml

        - _target_: anemoi.training.diagnostics.callbacks.plot.SpatialMapPlot
          tag_infix: my_map
          sample_idx: 0
          parameters: [z_500, 2t]
          every_n_batches: 750
          plot_fn:
            _target_: my_package.my_plot_fn
            _partial_: true
            my_option: 42
    """

    def __init__(
        self,
        plot_fn: Any,
        tag_infix: str,
        sample_idx: int,
        parameters: list[str],
        *,
        with_auxiliary: bool = False,
        members: Any = _UNSET_MEMBERS,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: dict | None = None,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the SpatialMapPlot callback.

        Parameters
        ----------
        plot_fn : Callable
            Plot function (typically a ``functools.partial`` from Hydra with
            ``_partial_: true``) matching the ``SpatialMapPlot`` ``plot_fn``
            contract documented in ``docs/modules/diagnostics.rst``.
        tag_infix : str
            Short tag inserted into the logged artifact name to distinguish
            this callback's outputs (e.g. ``"sample"``, ``"spec"``, ``"histo"``).
        sample_idx : int
            Index of the sample within the batch to plot.
        parameters : list[str]
            Model output parameters to include in the plot.
        with_auxiliary : bool, optional
            If True, forward the optional auxiliary tensor (e.g. corrupted
            targets) to ``plot_fn``. Default False.
        members : int | list[int] | None, optional
            Ensemble members to select. Defaults to the plot adapter's default.
        every_n_batches, dataset_names, focus_area, plotting_settings
            See :class:`BasePlotAdditionalMetrics`.
        """
        super().__init__(
            sample_idx=sample_idx,
            every_n_batches=every_n_batches,
            dataset_names=dataset_names,
            focus_area=focus_area,
            plotting_settings=plotting_settings,
        )
        self.plot_fn = plot_fn
        self.tag_infix = tag_infix
        self.parameters = parameters
        self.with_auxiliary = with_auxiliary
        self._members = members

    def _get_process_members(self) -> int | list[int] | None:
        return self._members

    def _plot_kwargs_from_output(
        self,
        pl_module: pl.LightningModule,
        output: TrainingStepOutput,
    ) -> dict[str, Any]:
        if not self.with_auxiliary:
            return {}
        auxiliary_output = self._gather_auxiliary(pl_module, output)
        if auxiliary_output is None:
            return {}
        return {"auxiliary_output": auxiliary_output}

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        outputs: TrainingStepOutput,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
        auxiliary_output: dict[str, torch.Tensor] | None = None,
        processed_cache: dict | None = None,
    ) -> None:
        logger = trainer.logger
        local_rank = pl_module.local_rank

        for dataset_name in dataset_names:
            spatial_inputs = extract_spatial_inputs(pl_module, dataset_name, self.parameters)

            data, output_tensor = self.process(
                pl_module,
                dataset_name,
                outputs,
                batch,
                members=self._get_process_members(),
                processed_cache=processed_cache,
            )

            auxiliary_tensor = None
            if self.with_auxiliary and auxiliary_output is not None:
                auxiliary_tensor = self.process_output_tensor(
                    pl_module,
                    dataset_name,
                    [auxiliary_output],
                    members=self._get_process_members(),
                )

            extra_fields = (auxiliary_tensor,) if auxiliary_tensor is not None else ()
            latlons, data, output_tensor, *masked_extra = self.focus_mask.apply(
                pl_module.model.model._graph_data,
                self.latlons[dataset_name],
                data,
                output_tensor,
                *extra_fields,
            )
            auxiliary_tensor = masked_extra[0] if masked_extra else None

            auxiliary_by_suffix: dict[str, Any] = {}
            if auxiliary_tensor is not None:
                auxiliary_by_suffix = {
                    suffix: aux
                    for _, _, aux, suffix in pl_module.plot_adapter.iter_plot_samples(data, auxiliary_tensor)
                }

            for x, y_true, y_pred, tag_suffix in pl_module.plot_adapter.iter_plot_samples(data, output_tensor):
                fig = self.plot_fn(
                    **spatial_inputs,
                    x=x,
                    y_true=y_true,
                    y_pred=y_pred,
                    latlons=latlons,
                    auxiliary=auxiliary_by_suffix.get(tag_suffix),
                    settings=self.plotting_settings,
                )
                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=(
                        f"pred_val_{self.tag_infix}_{dataset_name}_{tag_suffix}_"
                        f"batch{batch_idx:04d}_rank{local_rank:01d}{self.focus_mask.tag}"
                    ),
                    exp_log_tag=(
                        f"pred_val_{self.tag_infix}_{dataset_name}_{tag_suffix}_"
                        f"rank{local_rank:01d}{self.focus_mask.tag}"
                    ),
                )

