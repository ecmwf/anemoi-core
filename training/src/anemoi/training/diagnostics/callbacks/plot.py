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
import dataclasses
import logging
import threading
import traceback
from abc import ABC
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel as PydanticBaseModel
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from anemoi.models.data import Batch
from anemoi.models.data import SourceView
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
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.index_space import IndexSpace

LOGGER = logging.getLogger(__name__)


def _allgather_view(
    pl_module: pl.LightningModule,
    prediction: SourceView,
    dataset_name: str,
) -> SourceView:
    """All-gather a per-dataset prediction :class:`SourceView`.

    Grid-shard metadata now lives on the :class:`SourceView`, so the view gathers
    itself given the model communication group (idempotent when already
    full-grid). A bare tensor carries no shard metadata and therefore cannot be
    gathered here - scream loudly so the producer is fixed to hand over a view.
    """
    if not isinstance(prediction, SourceView):
        msg = (
            f"Prediction for dataset {dataset_name!r} is a raw {type(prediction).__name__}, "
            "not a SourceView. Grid-shard metadata now lives on the Batch/SourceView, "
            "so a bare tensor cannot be all-gathered. Produce a SourceView prediction."
        )
        raise TypeError(msg)
    return prediction.allgather(pl_module.model_comm_group)


def _is_sparse_dataset(batch: "Batch | dict", dataset_name: str) -> bool:
    """Return whether ``dataset_name`` uses a sparse / tabular (``time_in_grid``) layout.

    The gridded sample / spectrum / histogram plots assume a single lat/lon grid
    shared by the input, target and prediction panels. That does not hold for
    scattered observation datasets — each timestep has a different set of points
    and coordinates — so those datasets are skipped by the gridded plot callbacks.

    Parameters
    ----------
    batch : Batch | dict
        The (gathered) validation batch.
    dataset_name : str
        Name of the dataset to inspect.

    Returns
    -------
    bool
        ``True`` for sparse/observation datasets, ``False`` otherwise.
    """
    if isinstance(batch, Batch):
        layout = batch.layouts.get(dataset_name)
        return bool(layout is not None and layout.time_in_grid)
    return False


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
        return cls(
            datashader=plot_cfg.datashader,
            projection_kind=plot_cfg.projection_kind,
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

        init_plot_settings()

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

            # Gather grid-sharded tensors while preserving the Batch envelope
            # (layouts / statistics / coordinates) so downstream view-based access
            # (Batch.__getitem__ / task.get_targets) keeps working. Shard metadata
            # lives on the Batch / SourceViews, so the batch gathers itself given
            # the model communication group (idempotent when the batch was already
            # all-gathered in ``on_after_batch_transfer``).
            if isinstance(batch, Batch):
                batch = batch.allgather(pl_module.model_comm_group)
            else:
                # A bare dict of tensors carries no grid-shard metadata, which now
                # lives exclusively on the Batch / SourceView. Scream loudly rather
                # than silently mis-gathering.
                msg = (
                    "PlotCallback received a raw dict of tensors instead of a Batch. "
                    "Grid-shard metadata now lives on the Batch/SourceView, so a bare "
                    "tensor cannot be all-gathered - pass a Batch instead."
                )
                raise TypeError(msg)
            preds = output.predictions
            if not isinstance(preds, list):

                raise TypeError(preds)
            gathered_predictions = [
                {
                    dataset_name: _allgather_view(pl_module, dataset_pred, dataset_name)
                    for dataset_name, dataset_pred in pred.items()
                }
                for pred in preds
            ]
            # When running in Async mode, it might happen that in the last epoch these tensors
            # have been moved to the cpu (and then the denormalising would fail as the 'input_tensor' would be on CUDA
            # but internal ones would be on the cpu), The lines below allow to address this problem.
            # The refactored preprocessors are stateless (no per-instance ``nan_locations`` buffer to gather),
            # so only the device move is required here.
            self.post_processors = copy.deepcopy(pl_module.model.post_processors)
            for dataset_name in self.post_processors:
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

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        outputs: TrainingStepOutput,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger
        _ = batch_idx
        data_indices = pl_module.data_indices

        for dataset_name in dataset_names:
            parameter_names = list[str](data_indices[dataset_name].model.output.name_to_index.keys())
            parameter_positions = list[int](data_indices[dataset_name].model.output.name_to_index.values())
            # reorder parameter_names by position
            parameter_names = [parameter_names[i] for i in np.argsort(parameter_positions)]
            metadata = pl_module.model.metadata
            metadata_variables = metadata["dataset"].get("variables_metadata") if metadata is not None else None

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

            for i, task_kwargs in enumerate(pl_module.task.steps("validation")):
                y_hat = outputs.predictions[i][dataset_name]
                # Pass the full target batch; index the per-dataset SourceView afterwards.
                batch_obj = batch if isinstance(batch, Batch) else Batch(data=batch)
                y_true_batch, _ = pl_module.task.get_targets(batch_obj, data_indices, **task_kwargs)
                y_true_batch = pl_module.preprocess_targets(y_true_batch)
                y_true = y_true_batch[dataset_name]
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

                sort_by_parameter_group, colors, xticks, legend_patches = self.sort_and_color_by_parameter_group(
                    parameter_names,
                )
                loss = loss[argsort_indices]
                fig = plot_loss(loss[sort_by_parameter_group], colors, xticks, legend_patches)

                metric_name = pl_module.task.get_metric_name(**task_kwargs)
                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=f"loss_{dataset_name}{metric_name}_rank{pl_module.local_rank:01d}",
                    exp_log_tag=f"loss_sample_{dataset_name}{metric_name}_rank{pl_module.local_rank:01d}",
                )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: TrainingStepOutput,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:

        if batch_idx % self.every_n_batches == 0:

            self.loss = copy.deepcopy(pl_module.loss)

            super().on_validation_batch_end(
                trainer,
                pl_module,
                output,
                pl_module.plot_adapter.prepare_loss_batch(batch),
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
        outputs: TrainingStepOutput,
        batch: Batch,
        members: int | list[int] | None = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process the data and output tensors for plotting one dataset specified by dataset_name.

        Parameters
        ----------
        pl_module : pl.LightningModule
            The LightningModule instance.
        dataset_name : str
            The name of the dataset to process.
        outputs : TrainingStepOutput
            The outputs from the model. The predictions must be a list of dicts
            (one per outer step).
        batch : Batch
            The batch of data.
        members : int | list[int] | None, optional
            Ensemble members to select. Only used when the plot adapter is ensemble-aware.
            None returns all members. Default is 0 (first member).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The lat/lon coordinates, the (post-processed) data tensor and the
            output tensor for plotting.
        """
        assert isinstance(
            outputs.predictions,
            list,
        ), "outputs.predictions must be a list of per-step dicts."

        view = batch[dataset_name]

        # lat/lon coordinates come from the SourceView (radians -> degrees).
        if isinstance(view.coordinates, torch.Tensor):
            latlons = np.rad2deg(view.coordinates.detach().cpu().numpy())
        else:
            assert isinstance(view.coordinates, list) and len(view.coordinates) > self.sample_idx, (
                "Expected view.coordinates to be a list of tensors when plotting per-sample"
                f" with length greater than {self.sample_idx}."
            )
            latlons = np.rad2deg(view.coordinates[self.sample_idx].detach().cpu().numpy())

        # Restrict to the output variables, normalize without imputation, then
        # convert back to physical space for plotting.
        feature_indices = pl_module.data_indices[dataset_name].data.output.full
        selected = batch.select(variables={dataset_name: feature_indices})
        input_view = pl_module.preprocess_targets(selected)[dataset_name].apply_func(lambda t, **_: t.detach().cpu())
        data = self.post_processors[dataset_name](input_view, in_place=False).data[self.sample_idx]

        output_tensor = self.process_output_tensor(pl_module, dataset_name, outputs.predictions, members=members)

        data[1:, ...] = pl_module.output_mask[dataset_name].apply(
            data[1:, ...],
            dim=pl_module.grid_dim,
            fill_value=np.nan,
        )
        data = data.numpy()

        return latlons, data, output_tensor

    def process_output_tensor(
        self,
        pl_module: pl.LightningModule,
        dataset_name: str,
        outputs: list[dict[str, SourceView]],
        members: int | list[int] | None = 0,
    ) -> np.ndarray:
        """Post-process and mask per-step output SourceViews for plotting."""
        post_processor = self.post_processors[dataset_name]
        output_indices_full = pl_module.data_indices[dataset_name].data.output.full

        def _post_process(prediction: SourceView) -> torch.Tensor:
            assert isinstance(
                prediction,
                SourceView,
            ), f"Expected a prediction of type SourceView, got {type(prediction)}."
            aligned = self._align_output_metadata(prediction, output_indices_full)
            processed = post_processor(aligned.apply_func(lambda t, **_: t.detach().cpu()), in_place=False).data
            # Gridded views wrap a single ``(batch, ...)`` tensor; tabular/obs views wrap a
            # list of per-sample tensors. Select the requested sample, keeping a leading
            # size-1 axis so per-step outputs can be concatenated along dim 0.
            if isinstance(processed, list):
                return processed[self.sample_idx].unsqueeze(0)
            return processed[self.sample_idx : self.sample_idx + 1]

        output_tensor = torch.cat(
            tuple(
                pl_module.plot_adapter.select_members(
                    _post_process(x[dataset_name]),
                    members,
                )
                for x in outputs
            ),
        )

        output_tensor = pl_module.plot_adapter.prepare_plot_output_tensor(output_tensor)
        return (
            pl_module.output_mask[dataset_name].apply(output_tensor, dim=pl_module.grid_dim, fill_value=np.nan).numpy()
        )

    @staticmethod
    def _align_output_metadata(view: SourceView, output_indices_full: Any) -> SourceView:
        """Re-slice a prediction view's metadata to its (model-output) variables."""
        data0 = view.data[0] if isinstance(view.data, list) else view.data
        var_width = data0.shape[view.layout.variables]
        if len(view.variables) == var_width:
            return view
        if isinstance(output_indices_full, slice):
            output_indices = list(range(len(view.variables)))[output_indices_full]
        else:
            output_indices = [int(i) for i in output_indices_full]
        sub_variables = [view.variables[i] for i in output_indices]
        sub_statistics = {key: value[output_indices] for key, value in view.statistics.items()}
        return dataclasses.replace(view, variables=sub_variables, statistics=sub_statistics)

    def _sparse_sample(
        self,
        pl_module: pl.LightningModule,
        dataset_name: str,
        outputs: TrainingStepOutput,
        batch: Batch,
        auxiliary_output: dict[str, SourceView] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Build the plotting fields and coordinates for a sparse/observation dataset.

        We extract the fields directly from the per-dataset SourceView using select_time.

        Parameters
        ----------
        pl_module : pl.LightningModule
            The training module holding task, data indices and post-processors.
        dataset_name : str
            Name of the sparse/observation dataset to plot.
        outputs : TrainingStepOutput
            Validation step output holding the predictions.
        batch : Batch
            Full batch the sample is extracted from.
        auxiliary_output : dict[str, SourceView] | None, optional
            Optional per-dataset auxiliary views (e.g. the conditioned transport
            target) to plot alongside the prediction, by default None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
            ``(input_latlons, output_latlons, x, y_true, y_pred, auxiliary)``.
        """
        feature_indices = pl_module.data_indices[dataset_name].data.output.full
        task = pl_module.task

        # Analysis-time input step (last input index) and the first validation output step.
        input_indices = list(task.get_batch_input_indices())
        step_kwargs = next(iter(task.steps("validation")))
        output_indices = task.get_batch_output_indices(**step_kwargs)

        def _field_and_coords(sub_view: SourceView) -> tuple[np.ndarray, np.ndarray]:
            field = sub_view.data[self.sample_idx]  # (grid, vars) for sparse obs
            coords = sub_view.coordinates[self.sample_idx]
            return field.detach().cpu().numpy(), np.rad2deg(coords.detach().cpu().numpy())

        def _output_field(output_view: SourceView) -> np.ndarray:
            output_view = self._align_output_metadata(output_view, feature_indices)
            output_view = self.post_processors[dataset_name](
                output_view.apply_func(lambda t, **_: t.detach().cpu()),
                in_place=False,
            )
            return output_view.data[self.sample_idx].numpy()

        # Input panel: observations at the analysis (last input) timestep.
        input_batch = batch.select(time={dataset_name: input_indices[-1]}, variables={dataset_name: feature_indices})
        input_view = input_batch[dataset_name]
        x, input_latlons = _field_and_coords(input_view)

        # Target panel: observations at the output timesteps.
        target_batch = batch.select(time={dataset_name: output_indices}, variables={dataset_name: feature_indices})
        target_view = target_batch[dataset_name]
        y_true, output_latlons = _field_and_coords(target_view)

        # Prediction (already at the output/target observation locations).
        y_pred = _output_field(outputs.predictions[0][dataset_name])

        auxiliary = None
        if auxiliary_output is not None and dataset_name in auxiliary_output:
            auxiliary = _output_field(auxiliary_output[dataset_name])

        return input_latlons, output_latlons, x, y_true, y_pred, auxiliary


class PlotSample(BasePlotAdditionalMetrics):
    """Plots a post-processed sample: input, target and prediction."""

    output_tag_prefix = "pred_val_sample"
    exp_log_tag_prefix = "val_pred_sample"

    def __init__(
        self,
        sample_idx: int,
        parameters: list[str],
        accumulation_levels_plot: list[float],
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict[str, Colormap] | None = None,
        per_sample: int = 6,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
        prediction_label: str = "pred",
        auxiliary_label: str = "corrupted targets",
        dpi: int = 200,
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
        precip_and_related_fields : list[str] | None, optional
            Precip variable names, by default None
        colormaps : dict[str, Colormap] | None, optional
            Dictionary of colormaps, by default None
        per_sample : int, optional
            Number of plots per sample, by default 6
        every_n_batches : int, optional
            Batch frequency to plot at, by default None
        dataset_names : list[str] | None, optional
            Dataset names, by default None
        focus_area : list[dict] | None, optional
            Focus area configuration, by default None
        dpi : int, optional
            Figure render resolution in dots per inch, by default 200
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

        # Per-callback value takes priority; fall back to plotting_settings if not given.
        self.precip_and_related_fields = precip_and_related_fields or (
            self.plotting_settings.precip_and_related_fields if self.plotting_settings else None
        )
        self.accumulation_levels_plot = accumulation_levels_plot
        self.per_sample = per_sample
        self.colormaps = colormaps
        self.prediction_label = prediction_label
        self.auxiliary_label = auxiliary_label
        self.dpi = dpi

        LOGGER.info(
            "Using defined accumulation colormap for fields: %s",
            self.precip_and_related_fields,
        )

    def _figure_tags(self, dataset_name: str, tag_suffix: str, batch_idx: int, local_rank: int) -> tuple[str, str]:
        focus_tag = self.focus_mask.tag
        tag = (
            f"{self.output_tag_prefix}_{dataset_name}_{tag_suffix}_"
            f"batch{batch_idx:04d}_rank{local_rank:01d}{focus_tag}"
        )
        exp_log_tag = f"{self.exp_log_tag_prefix}_{dataset_name}_{tag_suffix}_rank{local_rank:01d}{focus_tag}"
        return tag, exp_log_tag

    def _plot_kwargs_from_output(
        self,
        pl_module: pl.LightningModule,
        output: TrainingStepOutput,
    ) -> dict[str, Any]:
        """Return the optional corrupted-target field for sample plots."""
        auxiliary_output = output.plot_kwargs.get("auxiliary_output")
        if auxiliary_output is None:
            return {}

        # ``auxiliary_output`` arrives as a dict of raw, detached tensors (see
        # TransportForecaster._step -> plot_kwargs["auxiliary_output"]). Grid-shard
        # metadata now lives exclusively on the Batch / SourceView, so a bare tensor
        # cannot be all-gathered here. Scream loudly until the producer hands over a
        # SourceView / Batch instead of detached tensors.
        if any(not isinstance(value, (Batch, SourceView)) for value in auxiliary_output.values()):
            msg = (
                "auxiliary_output is a dict of raw tensors without shard metadata; "
                "cannot all-gather. Grid-shard info now lives on the Batch/SourceView - "
                "have the step output carry a SourceView/Batch for auxiliary_output "
                "instead of detached tensors."
            )
            raise TypeError(msg)

        auxiliary_output = {
            dataset_name: value.allgather(pl_module.model_comm_group)
            for dataset_name, value in auxiliary_output.items()
        }
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
        auxiliary_output: dict[str, SourceView] | None = None,
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

            local_rank = pl_module.local_rank

            if _is_sparse_dataset(batch, dataset_name):
                input_latlons, output_latlons, x, y_true, y_pred, auxiliary = self._sparse_sample(
                    pl_module,
                    dataset_name,
                    outputs,
                    batch,
                    auxiliary_output=auxiliary_output,
                )
                fig = self._make_figure(
                    plot_parameters_dict,
                    input_latlons,
                    x,
                    y_true,
                    y_pred,
                    auxiliary=auxiliary,
                    sparse=True,
                    output_latlons=output_latlons,
                )
                # match the gridded filename pattern
                step_kwargs = next(iter(pl_module.task.steps("validation")))
                tag_suffix = f"rstep{step_kwargs.get('rollout_step', 0):02d}_out00"
                tag, exp_log_tag = self._figure_tags(dataset_name, tag_suffix, batch_idx, local_rank)
                self._output_figure(logger, fig, epoch=epoch, tag=tag, exp_log_tag=exp_log_tag)
                continue

            data_latlons, data, output_tensor = self.process(
                pl_module,
                dataset_name,
                outputs,
                batch,
                members=self._get_process_members(),
            )
            auxiliary_tensor = (
                None
                if auxiliary_output is None
                else self.process_output_tensor(
                    pl_module,
                    dataset_name,
                    [auxiliary_output],
                    members=self._get_process_members(),
                )
            )

            if auxiliary_tensor is not None:
                latlons, data, output_tensor, auxiliary_tensor = self.focus_mask.apply(
                    pl_module.model.model._graph_data,
                    data_latlons,
                    data,
                    output_tensor,
                    auxiliary_tensor,
                )
            else:
                # Apply spatial mask
                latlons, data, output_tensor = self.focus_mask.apply(
                    pl_module.model.model._graph_data,
                    data_latlons,
                    data,
                    output_tensor,
                )
            auxiliary_by_suffix = {}
            if auxiliary_tensor is not None:
                auxiliary_by_suffix = {
                    auxiliary_suffix: auxiliary
                    for _, _, auxiliary, auxiliary_suffix in pl_module.plot_adapter.iter_plot_samples(
                        data,
                        auxiliary_tensor,
                    )
                }

            for x, y_true, y_pred, tag_suffix in pl_module.plot_adapter.iter_plot_samples(data, output_tensor):
                auxiliary = auxiliary_by_suffix.get(tag_suffix)
                fig = self._make_figure(
                    plot_parameters_dict,
                    latlons,
                    x,
                    y_true,
                    y_pred,
                    auxiliary=auxiliary,
                )

                tag, exp_log_tag = self._figure_tags(dataset_name, tag_suffix, batch_idx, local_rank)
                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=tag,
                    exp_log_tag=exp_log_tag,
                )

    def _get_process_members(self) -> int | list[int] | None:
        """Return the ``members`` argument passed to ``process()``.

        Default selects member 0 (deterministic view). Subclasses override
        to pass a different selection (e.g. all members for ensemble plots).
        """
        return 0

    def _make_figure(
        self,
        plot_parameters_dict: dict,
        latlons: np.ndarray,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        auxiliary: np.ndarray | None = None,
        sparse: bool = False,
        output_latlons: np.ndarray | None = None,
    ) -> Figure:
        """Create the matplotlib Figure for one (x, y_true, y_pred) triplet.

        For sparse/observation datasets, sparse=True selects the reduced
        observation panel layout and output_latlons provides the target obs coordinates.
        """
        return plot_predicted_multilevel_flat_sample(
            plot_parameters_dict,
            self.per_sample,
            latlons,
            self.accumulation_levels_plot,
            x,
            y_true,
            y_pred,
            datashader=self.datashader_plotting,
            precip_and_related_fields=self.precip_and_related_fields,
            colormaps=self.colormaps,
            projection_kind=self.projection_kind,
            prediction_label=self.prediction_label,
            auxiliary=auxiliary,
            auxiliary_label=self.auxiliary_label,
            sparse=sparse,
            output_latlons=output_latlons,
            dpi=self.dpi,
        )


class PlotEnsSample(PlotSample):
    """Plot ensemble mean, spread, and the difference of members to the mean for each variable."""

    def __init__(
        self,
        sample_idx: int,
        parameters: list[str],
        accumulation_levels_plot: list[float],
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict[str, Colormap] | None = None,
        per_sample: int = 6,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        members: list[int] | int | None = None,
        focus_area: list[dict] | None = None,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the PlotEnsSample callback.

        Parameters
        ----------
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
        every_n_batches : int | None, optional
            Batch frequency to plot at, by default None
        dataset_names : list[str] | None, optional
            Dataset names, by default None
        members : list[int] | int | None, optional
            Ensemble members to plot. None plots all members, by default None.
        focus_area : list[dict] | None, optional
            Focus area configuration, by default None
        plotting_settings : PlottingSettings | None, optional
            Plotting configuration settings, by default None (uses defaults)
        """
        super().__init__(
            sample_idx=sample_idx,
            parameters=parameters,
            accumulation_levels_plot=accumulation_levels_plot,
            precip_and_related_fields=precip_and_related_fields,
            colormaps=colormaps,
            per_sample=per_sample,
            every_n_batches=every_n_batches,
            dataset_names=dataset_names,
            focus_area=focus_area,
            plotting_settings=plotting_settings,
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
        auxiliary: np.ndarray | None = None,  # noqa: ARG002
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


class PlotSpectrum(BasePlotAdditionalMetrics):
    """Plots TP related metric comparing target and prediction.

    The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.

    - Power Spectrum
    """

    def __init__(
        self,
        sample_idx: int,
        parameters: list[str],
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
        self.min_delta = min_delta

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        outputs: TrainingStepOutput,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger

        local_rank = pl_module.local_rank
        for dataset_name in dataset_names:
            if _is_sparse_dataset(batch, dataset_name):
                LOGGER.warning(
                    "%s: skipping dataset %r: spectra plots are not yet supported for sparse observations.",
                    type(self).__name__,
                    dataset_name,
                )
                continue
            data_latlons, data, output_tensor = self.process(pl_module, dataset_name, outputs, batch)

            # Apply spatial mask
            latlons, data, output_tensor = self.focus_mask.apply(
                pl_module.model.model._graph_data,
                data_latlons,
                data,
                output_tensor,
            )

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

            for x, y_true, y_pred, tag_suffix in pl_module.plot_adapter.iter_plot_samples(data, output_tensor):
                fig = plot_power_spectrum(
                    plot_parameters_dict_spectrum,
                    latlons,
                    x,
                    y_true,
                    y_pred,
                    min_delta=self.min_delta,
                )

                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=(
                        f"pred_val_spec_{dataset_name}_{tag_suffix}_"
                        f"batch{batch_idx:04d}_rank{local_rank:01d}{self.focus_mask.tag}"
                    ),
                    exp_log_tag=(
                        f"pred_val_spec_{dataset_name}_{tag_suffix}_rank{local_rank:01d}{self.focus_mask.tag}"
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
        self.precip_and_related_fields = precip_and_related_fields
        self.log_scale = log_scale

        LOGGER.info(
            "Using precip histogram plotting method for fields: %s.",
            self.precip_and_related_fields,
        )

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        outputs: TrainingStepOutput,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger

        local_rank = pl_module.local_rank

        for dataset_name in dataset_names:
            if _is_sparse_dataset(batch, dataset_name):
                LOGGER.warning(
                    "%s: skipping dataset %r: gridded histogram plots are not supported for sparse observations.",
                    type(self).__name__,
                    dataset_name,
                )
                continue

            data_latlons, data, output_tensor = self.process(pl_module, dataset_name, outputs, batch)

            # Build dictionary of indices and parameters to be plotted
            input_data = pl_module.data_indices[dataset_name].data.input.todict()
            index_to_name = {v: k for k, v in input_data["name_to_index"].items()}
            diagnostics = {index_to_name[int(i)] for i in input_data["diagnostic"]}
            # Apply spatial mask
            _, data, output_tensor = self.focus_mask.apply(
                pl_module.model.model._graph_data,
                data_latlons,
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

            for x, y_true, y_pred, tag_suffix in pl_module.plot_adapter.iter_plot_samples(data, output_tensor):

                fig = plot_histogram(
                    plot_parameters_dict_histogram,
                    x,
                    y_true,
                    y_pred,
                    self.precip_and_related_fields,
                    self.log_scale,
                )

                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=(
                        f"pred_val_histo_{dataset_name}_{tag_suffix}_"
                        f"batch{batch_idx:04d}_rank{local_rank:01d}{self.focus_mask.tag}"
                    ),
                    exp_log_tag=(
                        f"pred_val_histo_{dataset_name}_{tag_suffix}_rank{local_rank:01d}{self.focus_mask.tag}"
                    ),
                )
