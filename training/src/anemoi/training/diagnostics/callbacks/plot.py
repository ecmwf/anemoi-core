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
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel as PydanticBaseModel
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from anemoi.models.data import Batch
from anemoi.models.data import SourceView
from anemoi.training.diagnostics.evaluation.geospatial.focus_area import build_spatial_mask
from anemoi.training.diagnostics.evaluation.plotting.graph import graph_plot_fn as _default_graph_plot_fn
from anemoi.training.diagnostics.evaluation.plotting.loss import loss_plot_fn as _default_loss_plot_fn
from anemoi.training.diagnostics.evaluation.plotting.model_introspection import extract_graph_inputs
from anemoi.training.diagnostics.evaluation.plotting.model_introspection import extract_loss_inputs
from anemoi.training.diagnostics.evaluation.plotting.model_introspection import extract_spatial_inputs
from anemoi.training.diagnostics.evaluation.plotting.protocols import BatchOutputPlotFn
from anemoi.training.diagnostics.evaluation.plotting.protocols import GraphPlotFn
from anemoi.training.diagnostics.evaluation.plotting.protocols import LossPlotFn
from anemoi.training.diagnostics.evaluation.plotting.protocols import validate_plot_fn
from anemoi.training.diagnostics.evaluation.plotting.settings import init_plot_settings
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.utils import reduce_to_last_dim
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.index_space import IndexSpace

LOGGER = logging.getLogger(__name__)


class _Unset:
    """Typed sentinel for kwargs that must distinguish "not specified" from ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "UNSET"


# Sentinel distinguishing "members not specified, use the default" from an explicit
# ``members=None`` ("select all members").
_UNSET_MEMBERS: Any = _Unset()


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
        """Construct from a validated ``diagnostics.plot`` config node.

        Rendering settings live under the ``settings`` sub-node (new pluggable
        callback structure); ``focus_areas`` / ``datasets_to_plot`` remain at the
        ``diagnostics.plot`` top level.
        """
        from hydra.utils import instantiate

        settings_cfg = OmegaConf.select(plot_cfg, "settings", default=None)
        settings_cfg = settings_cfg if settings_cfg is not None else plot_cfg
        raw_colormaps = OmegaConf.select(settings_cfg, "colormaps", default=None)
        colormaps = instantiate(raw_colormaps) if raw_colormaps is not None else None
        return cls(
            datashader=OmegaConf.select(settings_cfg, "datashader", default=True),
            projection_kind=OmegaConf.select(settings_cfg, "projection_kind", default="equirectangular"),
            asynchronous=OmegaConf.select(settings_cfg, "asynchronous", default=True),
            save_basedir=save_basedir,
            colormaps=colormaps,
            precip_and_related_fields=OmegaConf.select(settings_cfg, "precip_and_related_fields", default=None),
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build the plotting fields and coordinates for a tabular observation dataset.

        We extract the fields directly from the per-dataset SourceView using select_time.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Shape: (input_latlons, output_latlons, x, y_true, y_pred).
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

        # Input panel: observations at the analysis (last input) timestep.
        input_batch = batch.select(time={dataset_name: input_indices[-1]}, variables={dataset_name: feature_indices})
        input_view = input_batch[dataset_name]
        x, input_latlons = _field_and_coords(input_view)

        # Target panel: observations at the output timesteps.
        target_batch = batch.select(time={dataset_name: output_indices}, variables={dataset_name: feature_indices})
        target_view = target_batch[dataset_name]
        y_true, output_latlons = _field_and_coords(target_view)

        # Prediction (already at the output/target observation locations).
        prediction = self._align_output_metadata(outputs.predictions[0][dataset_name], feature_indices)
        prediction = self.post_processors[dataset_name](
            prediction.apply_func(lambda t, **_: t.detach().cpu()),
            in_place=False,
        )
        y_pred = prediction.data[self.sample_idx].numpy()

        return input_latlons, output_latlons, x, y_true, y_pred


class LossCurvePlot(BasePerBatchPlotCallback):
    """Plot the per-variable validation loss (new pluggable callback structure).

    Reuses the richer-batch loss computation (Batch/SourceView targets via
    ``task.get_targets``) and delegates the figure rendering to a pluggable
    ``plot_fn`` (default :func:`loss_plot_fn`).
    """

    def __init__(
        self,
        parameter_groups: dict[str, list[str]],
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        plot_fn: Any = None,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the LossCurvePlot callback.

        Parameters
        ----------
        parameter_groups : dict
            Dictionary with parameter groups with parameter names as keys.
        every_n_batches : int, optional
            Override for batch frequency, by default None.
        dataset_names : list[str] | None, optional
            Dataset names, by default None.
        plot_fn : callable, optional
            Plug-in figure function; defaults to :func:`loss_plot_fn`.
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults).
        """
        super().__init__(
            every_n_batches=every_n_batches,
            dataset_names=dataset_names,
            plotting_settings=plotting_settings,
        )
        self.parameter_groups = parameter_groups if parameter_groups is not None else {}
        self.dataset_names = dataset_names if dataset_names is not None else ["data"]
        self.plot_fn = plot_fn if plot_fn is not None else _default_loss_plot_fn
        validate_plot_fn(self.plot_fn, LossPlotFn, "LossCurvePlot")

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        outputs: TrainingStepOutput,
        batch: Batch,
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger
        _ = batch_idx
        data_indices = pl_module.data_indices

        for dataset_name in dataset_names:
            if not isinstance(self.loss[dataset_name], BaseLoss):
                LOGGER.warning(
                    "Loss function must be a subclass of BaseLoss, or provide `squash`.",
                    RuntimeWarning,
                )
            loss_inputs = extract_loss_inputs(pl_module, dataset_name, self.parameter_groups)

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

                fig = self.plot_fn(loss, **loss_inputs, settings=self.plotting_settings)

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
        batch: Batch,
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


class BatchOutputPlot(BasePlotAdditionalMetrics):
    """Generic per-batch spatial-output plot driven by a pluggable ``plot_fn``.

    One callback class serves the sample / spectrum / histogram map plots; the
    concrete figure is produced by the injected ``plot_fn``. Reuses the
    richer-batch data extraction (``process`` / ``process_output_tensor`` for
    gridded data, ``_sparse_sample`` for scattered observation datasets). When a
    ``plot_fn`` does not support sparse data (e.g. spectrum / histogram) it
    returns ``None`` for such datasets and the callback skips them.
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
        """Initialise the BatchOutputPlot callback.

        Parameters
        ----------
        plot_fn : callable
            Plug-in figure function conforming to :class:`BatchOutputPlotFn`.
        tag_infix : str
            Short label (e.g. ``sample`` / ``spec`` / ``histo``) placed in the
            artifact tag to disambiguate multiple BatchOutputPlot callbacks.
        sample_idx : int
            Sample to plot.
        parameters : list[str]
            Parameters to plot.
        with_auxiliary : bool, optional
            Forward the optional auxiliary tensor (e.g. corrupted targets) to
            ``plot_fn``, by default False.
        members : int | list[int] | None, optional
            Ensemble members to select; defaults to the adapter default.
        every_n_batches : int, optional
            Batch frequency to plot at, by default None.
        dataset_names : list[str] | None, optional
            Dataset names, by default None.
        focus_area : dict | None, optional
            Focus area configuration, by default None.
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults).
        """
        super().__init__(
            every_n_batches=every_n_batches,
            dataset_names=dataset_names,
            focus_area=focus_area,
            plotting_settings=plotting_settings,
        )
        validate_plot_fn(plot_fn, BatchOutputPlotFn, "BatchOutputPlot")
        self.plot_fn = plot_fn
        self.tag_infix = tag_infix
        self.sample_idx = sample_idx
        self.parameters = parameters
        self.with_auxiliary = with_auxiliary
        self._members = members

    @property
    def artifact_subfolder(self) -> str:
        """Derive the artifact subfolder from the plot function name."""
        fn = self.plot_fn
        while hasattr(fn, "func"):
            fn = fn.func
        return getattr(fn, "__name__", type(self).__name__)

    def _get_process_members(self) -> int | list[int] | None:
        """Return the ``members`` argument passed to ``process()``."""
        return 0 if isinstance(self._members, _Unset) else self._members

    def _figure_tags(self, dataset_name: str, tag_suffix: str, batch_idx: int, local_rank: int) -> tuple[str, str]:
        focus_tag = self.focus_mask.tag
        tag = (
            f"pred_val_{self.tag_infix}_{dataset_name}_{tag_suffix}_"
            f"batch{batch_idx:04d}_rank{local_rank:01d}{focus_tag}"
        )
        exp_log_tag = f"pred_val_{self.tag_infix}_{dataset_name}_{tag_suffix}_rank{local_rank:01d}{focus_tag}"
        return tag, exp_log_tag

    def _plot_kwargs_from_output(
        self,
        pl_module: pl.LightningModule,
        output: TrainingStepOutput,
    ) -> dict[str, Any]:
        """Return the optional corrupted-target field for sample plots."""
        if not self.with_auxiliary:
            return {}
        auxiliary_output = output.plot_kwargs.get("auxiliary_output")
        if auxiliary_output is None:
            return {}
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
        batch: Batch,
        batch_idx: int,
        epoch: int,
        auxiliary_output: dict[str, Any] | None = None,
        processed_cache: dict | None = None,
    ) -> None:
        _ = processed_cache
        logger = trainer.logger

        for dataset_name in dataset_names:
            spatial_inputs = extract_spatial_inputs(pl_module, dataset_name, self.parameters)
            local_rank = pl_module.local_rank

            if _is_sparse_dataset(batch, dataset_name):
                input_latlons, output_latlons, x, y_true, y_pred = self._sparse_sample(
                    pl_module,
                    dataset_name,
                    outputs,
                    batch,
                )
                fig = self.plot_fn(
                    **spatial_inputs,
                    x=x,
                    y_true=y_true,
                    y_pred=y_pred,
                    latlons=input_latlons,
                    auxiliary=None,
                    sparse=True,
                    output_latlons=output_latlons,
                    settings=self.plotting_settings,
                )
                if fig is None:
                    # plot_fn does not support sparse observations (e.g. spectrum / histogram).
                    LOGGER.warning(
                        "%s: skipping sparse dataset %r (plot_fn does not support scattered observations).",
                        type(self).__name__,
                        dataset_name,
                    )
                    continue
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
                fig = self.plot_fn(
                    **spatial_inputs,
                    x=x,
                    y_true=y_true,
                    y_pred=y_pred,
                    latlons=latlons,
                    auxiliary=auxiliary_by_suffix.get(tag_suffix),
                    sparse=False,
                    output_latlons=None,
                    settings=self.plotting_settings,
                )
                if fig is None:
                    continue
                tag, exp_log_tag = self._figure_tags(dataset_name, tag_suffix, batch_idx, local_rank)
                self._output_figure(logger, fig, epoch=epoch, tag=tag, exp_log_tag=exp_log_tag)


class GraphFeaturePlot(BasePerEpochPlotCallback):
    """Visualise node & edge trainable features (new pluggable callback structure)."""

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        every_n_epochs: int | None = None,
        q_extreme_limit: float = 0.05,
        plot_fn: Any = None,
        plotting_settings: PlottingSettings | None = None,
    ) -> None:
        """Initialise the GraphFeaturePlot callback.

        Parameters
        ----------
        dataset_names : list[str] | None, optional
            Dataset names, by default None.
        every_n_epochs : int | None, optional
            Override for frequency to plot at, by default None.
        q_extreme_limit : float, optional
            Quantile edges to represent, by default 0.05.
        plot_fn : callable, optional
            Plug-in generator function; defaults to :func:`graph_plot_fn`.
        plotting_settings : PlottingSettings, optional
            Plotting configuration settings, by default None (uses defaults).
        """
        super().__init__(
            dataset_names=dataset_names,
            every_n_epochs=every_n_epochs,
            plotting_settings=plotting_settings,
        )
        self.q_extreme_limit = q_extreme_limit
        self.plot_fn = plot_fn if plot_fn is not None else _default_graph_plot_fn
        validate_plot_fn(self.plot_fn, GraphPlotFn, "GraphFeaturePlot")

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
