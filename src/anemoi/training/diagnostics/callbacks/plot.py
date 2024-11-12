# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: ANN001

from __future__ import annotations

import copy
import logging
import sys
import time
import traceback
from abc import ABC
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from anemoi.training.diagnostics.plots import init_plot_settings
from anemoi.training.diagnostics.plots import plot_graph_edge_features
from anemoi.training.diagnostics.plots import plot_graph_node_features
from anemoi.training.diagnostics.plots import plot_histogram
from anemoi.training.diagnostics.plots import plot_loss
from anemoi.training.diagnostics.plots import plot_power_spectrum
from anemoi.training.diagnostics.plots import plot_predicted_multilevel_flat_sample
from anemoi.training.losses.weightedloss import BaseWeightedLoss

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)


class ParallelExecutor(ThreadPoolExecutor):
    """Wraps parallel execution and provides accurate information about errors.

    Extends ThreadPoolExecutor to preserve the original traceback and line number.

    Reference: https://stackoverflow.com/questions/19309514/getting-original-line-
    number-for-exception-in-concurrent-futures/24457608#24457608
    """

    def submit(self, fn: Any, *args, **kwargs) -> Callable:
        """Submits the wrapped function instead of `fn`."""
        return super().submit(self._function_wrapper, fn, *args, **kwargs)

    def _function_wrapper(self, fn: Any, *args: list, **kwargs: dict) -> Callable:
        """Wraps `fn` in order to preserve the traceback of any kind of."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise sys.exc_info()[0](traceback.format_exc()) from exc


class BasePlotCallback(Callback, ABC):
    """Factory for creating a callback that plots data to Experiment Logging."""

    def __init__(self, config: OmegaConf) -> None:
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
        self.pre_processors = None
        self.latlons = None
        init_plot_settings()

        self.plot = self._plot
        self._executor = None

        if self.config.diagnostics.plot.asynchronous:
            self._executor = ParallelExecutor(max_workers=1)
            self._error: BaseException | None = None
            self.plot = self._async_plot

    @rank_zero_only
    def _output_figure(
        self,
        logger: pl.loggers.base.LightningLoggerBase,
        fig: plt.Figure,
        epoch: int,
        tag: str = "gnn",
        exp_log_tag: str = "val_pred_sample",
    ) -> None:
        """Figure output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = Path(
                self.save_basedir,
                "plots",
                f"{tag}_epoch{epoch:03d}.png",
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            if self.config.diagnostics.log.wandb.enabled:
                import wandb

                logger.experiment.log({exp_log_tag: wandb.Image(fig)})

            if self.config.diagnostics.log.mlflow.enabled:
                run_id = logger.run_id
                logger.experiment.log_artifact(run_id, str(save_path))

        plt.close(fig)  # cleanup

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Method is called to close the threads."""
        del trainer, pl_module, stage  # unused
        if self._executor is not None:
            self._executor.shutdown(wait=True)

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
        *args,
        **kwargs,
    ) -> None:
        """Plotting function to be implemented by subclasses."""

    @rank_zero_only
    def _async_plot(
        self,
        trainer: pl.Trainer,
        *args: list,
        **kwargs: dict,
    ) -> None:
        """To execute the plot function but ensuring we catch any errors."""
        future = self._executor.submit(
            self._plot,
            trainer,
            *args,
            **kwargs,
        )
        # otherwise the error won't be thrown till the validation epoch is finished
        try:
            future.result()
        except Exception:
            LOGGER.exception("Critical error occurred in asynchronous plots.")
            sys.exit(1)


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

    @abstractmethod
    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
        **kwargs,
    ) -> None:
        """Plotting function to be implemented by subclasses."""

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        output,
        batch: torch.Tensor,
        batch_idx: int,
        **kwargs,
    ) -> None:
        if batch_idx % self.every_n_batches == 0:
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
    """Evaluates the model performance over a (longer) rollout window."""

    def __init__(
        self,
        config: OmegaConf,
        rollout: list[int],
        sample_idx: int,
        parameters: list[str],
        accumulation_levels_plot: list[float] | None = None,
        cmap_accumulation: list[str] | None = None,
        per_sample: int = 6,
        every_n_epochs: int = 1,
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
        accumulation_levels_plot : list[float] | None
            Accumulation levels to plot, by default None
        cmap_accumulation : list[str] | None
            Colors of the accumulation levels, by default None
        per_sample : int, optional
            Number of plots per sample, by default 6
        every_n_epochs : int, optional
            Epoch frequency to plot at, by default 1
        """
        super().__init__(config)

        self.every_n_epochs = every_n_epochs

        LOGGER.debug(
            "Setting up callback for plots with long rollout: rollout = %d, frequency = every %d epoch ...",
            rollout,
            every_n_epochs,
        )
        self.rollout = rollout
        self.sample_idx = sample_idx
        self.accumulation_levels_plot = accumulation_levels_plot
        self.cmap_accumulation = cmap_accumulation
        self.per_sample = per_sample
        self.parameters = parameters

    @rank_zero_only
    def _plot(
        self,
        trainer,
        pl_module: pl.LightningModule,
        output: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx,
        epoch,
    ) -> None:
        _ = output

        start_time = time.time()

        logger = trainer.logger

        # Build dictionary of inidicies and parameters to be plotted
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: (
                name,
                name not in self.config.data.get("diagnostic", []),
            )
            for name in self.parameters
        }

        if self.post_processors is None:
            # Copy to be used across all the training cycle
            self.post_processors = copy.deepcopy(pl_module.model.post_processors).cpu()
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())
        local_rank = pl_module.local_rank

        assert batch.shape[1] >= max(self.rollout) + pl_module.multi_step, (
            "Batch length not sufficient for requested validation rollout length! "
            f"Set `dataloader.validation_rollout` to at least {max(self.rollout)}"
        )

        # prepare input tensor for plotting
        input_batch = pl_module.model.pre_processors(batch, in_place=False)
        input_tensor_0 = input_batch[
            self.sample_idx,
            pl_module.multi_step - 1,
            ...,
            pl_module.data_indices.internal_data.output.full,
        ].cpu()
        data_0 = self.post_processors(input_tensor_0).numpy()

        # start rollout
        with torch.no_grad():
            for rollout_step, (_, _, y_pred) in enumerate(
                pl_module.rollout_step(
                    batch,
                    rollout=max(self.rollout),
                    validation_mode=False,
                    training_mode=False,
                ),
            ):

                if (rollout_step + 1) in self.rollout:
                    # prepare true output tensor for plotting
                    input_tensor_rollout_step = input_batch[
                        self.sample_idx,
                        pl_module.multi_step + rollout_step,  # (pl_module.multi_step - 1) + (rollout_step + 1)
                        ...,
                        pl_module.data_indices.internal_data.output.full,
                    ].cpu()
                    data_rollout_step = self.post_processors(input_tensor_rollout_step).numpy()

                    # prepare predicted output tensor for plotting
                    output_tensor = self.post_processors(
                        y_pred[self.sample_idx : self.sample_idx + 1, ...].cpu(),
                    ).numpy()

                    fig = plot_predicted_multilevel_flat_sample(
                        plot_parameters_dict,
                        self.per_sample,
                        self.latlons,
                        self.accumulation_levels_plot,
                        self.cmap_accumulation,
                        data_0.squeeze(),
                        data_rollout_step.squeeze(),
                        output_tensor[0, 0, :, :],  # rolloutstep, first member
                    )

                    self._output_figure(
                        logger,
                        fig,
                        epoch=epoch,
                        tag=f"gnn_pred_val_sample_rstep{rollout_step + 1:03d}_batch{batch_idx:04d}_rank0",
                        exp_log_tag=f"val_pred_sample_rstep{rollout_step + 1:03d}_rank{local_rank:01d}",
                    )
        LOGGER.info(
            "Time taken to plot samples after longer rollout: %s seconds",
            int(time.time() - start_time),
        )

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        output,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if (batch_idx) == 0 and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
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

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch: int,
    ) -> None:
        _ = epoch
        model = pl_module.model.module.model if hasattr(pl_module.model, "module") else pl_module.model.model

        fig = plot_graph_node_features(model)

        self._output_figure(
            trainer.logger,
            fig,
            epoch=trainer.current_epoch,
            tag="node_trainable_params",
            exp_log_tag="node_trainable_params",
        )

        fig = plot_graph_edge_features(model)

        self._output_figure(
            trainer.logger,
            fig,
            epoch=trainer.current_epoch,
            tag="edge_trainable_params",
            exp_log_tag="edge_trainable_params",
        )


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
            return parts[0]

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
        xticks = dict(zip(unique_group_list, x_tick_positions))

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
        pl_module: pl.Lightning_module,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger
        _ = batch_idx

        parameter_names = list(pl_module.data_indices.internal_model.output.name_to_index.keys())
        parameter_positions = list(pl_module.data_indices.internal_model.output.name_to_index.values())
        # reorder parameter_names by position
        self.parameter_names = [parameter_names[i] for i in np.argsort(parameter_positions)]
        if not isinstance(pl_module.loss, BaseWeightedLoss):
            logging.warning(
                "Loss function must be a subclass of BaseWeightedLoss, or provide `squash`.",
                RuntimeWarning,
            )

        batch = pl_module.model.pre_processors(batch, in_place=False)
        for rollout_step in range(pl_module.rollout):
            y_hat = outputs[1][rollout_step]
            y_true = batch[
                :,
                pl_module.multi_step + rollout_step,
                ...,
                pl_module.data_indices.internal_data.output.full,
            ]
            loss = pl_module.loss(y_hat, y_true, squash=False).cpu().numpy()

            sort_by_parameter_group, colors, xticks, legend_patches = self.sort_and_color_by_parameter_group
            fig = plot_loss(loss[sort_by_parameter_group], colors, xticks, legend_patches)

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"loss_rstep_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
                exp_log_tag=f"loss_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
            )


class PlotSample(BasePerBatchPlotCallback):
    """Plots a post-processed sample: input, target and prediction."""

    def __init__(
        self,
        config: OmegaConf,
        sample_idx: int,
        parameters: list[str],
        accumulation_levels_plot: list[float],
        cmap_accumulation: list[str],
        precip_and_related_fields: list[str] | None = None,
        per_sample: int = 6,
        every_n_batches: int | None = None,
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
        cmap_accumulation : list[str]
            Colors of the accumulation levels
        precip_and_related_fields : list[str] | None, optional
            Precip variable names, by default None
        per_sample : int, optional
            Number of plots per sample, by default 6
        every_n_batches : int, optional
            Batch frequency to plot at, by default None
        """
        super().__init__(config, every_n_batches=every_n_batches)
        self.sample_idx = sample_idx
        self.parameters = parameters

        self.precip_and_related_fields = precip_and_related_fields
        self.accumulation_levels_plot = accumulation_levels_plot
        self.cmap_accumulation = cmap_accumulation
        self.per_sample = per_sample

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

        # When running in Async mode, it might happen that in the last epoch these tensors
        # have been moved to the cpu (and then the denormalising would fail as the 'input_tensor' would be on CUDA
        # but internal ones would be on the cpu), The lines below allow to address this problem
        if self.post_processors is None:
            # Copy to be used across all the training cycle
            self.post_processors = copy.deepcopy(pl_module.model.post_processors).cpu()
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())
        local_rank = pl_module.local_rank

        batch = pl_module.model.pre_processors(batch, in_place=False)
        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
            ...,
            pl_module.data_indices.internal_data.output.full,
        ].cpu()
        data = self.post_processors(input_tensor)

        output_tensor = self.post_processors(
            torch.cat(tuple(x[self.sample_idx : self.sample_idx + 1, ...].cpu() for x in outputs[1])),
            in_place=False,
        )
        output_tensor = pl_module.output_mask.apply(output_tensor, dim=2, fill_value=np.nan).numpy()
        data[1:, ...] = pl_module.output_mask.apply(data[1:, ...], dim=2, fill_value=np.nan)
        data = data.numpy()

        for rollout_step in range(pl_module.rollout):
            fig = plot_predicted_multilevel_flat_sample(
                plot_parameters_dict,
                self.per_sample,
                self.latlons,
                self.accumulation_levels_plot,
                self.cmap_accumulation,
                data[0, ...].squeeze(),
                data[rollout_step + 1, ...].squeeze(),
                output_tensor[rollout_step, ...],
                precip_and_related_fields=self.precip_and_related_fields,
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"gnn_pred_val_sample_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank0",
                exp_log_tag=f"val_pred_sample_rstep{rollout_step:02d}_rank{local_rank:01d}",
            )


class BasePlotAdditionalMetrics(BasePerBatchPlotCallback):
    """Base processing class for additional metrics."""

    def process(
        self,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        # When running in Async mode, it might happen that in the last epoch these tensors
        # have been moved to the cpu (and then the denormalising would fail as the 'input_tensor' would be on CUDA
        # but internal ones would be on the cpu), The lines below allow to address this problem
        if self.pre_processors is None:
            # Copy to be used across all the training cycle
            self.pre_processors = copy.deepcopy(pl_module.model.pre_processors).cpu()
        if self.post_processors is None:
            # Copy to be used across all the training cycle
            self.post_processors = copy.deepcopy(pl_module.model.post_processors).cpu()
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())

        batch = pl_module.model.pre_processors(batch, in_place=False)
        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
            ...,
            pl_module.data_indices.internal_data.output.full,
        ].cpu()

        data = self.post_processors(input_tensor)
        output_tensor = self.post_processors(
            torch.cat(tuple(x[self.sample_idx : self.sample_idx + 1, ...].cpu() for x in outputs[1])),
            in_place=False,
        )
        output_tensor = pl_module.output_mask.apply(output_tensor, dim=2, fill_value=np.nan).numpy()
        data[1:, ...] = pl_module.output_mask.apply(data[1:, ...], dim=2, fill_value=np.nan)
        data = data.numpy()
        return data, output_tensor


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

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger

        local_rank = pl_module.local_rank
        data, output_tensor = self.process(pl_module, outputs, batch)

        for rollout_step in range(pl_module.rollout):
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
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"gnn_pred_val_spec_rstep_{rollout_step:02d}_batch{batch_idx:04d}_rank0",
                exp_log_tag=f"val_pred_spec_rstep_{rollout_step:02d}_rank{local_rank:01d}",
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
        LOGGER.info(
            "Using precip histogram plotting method for fields: %s.",
            self.precip_and_related_fields,
        )

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        logger = trainer.logger

        local_rank = pl_module.local_rank
        data, output_tensor = self.process(pl_module, outputs, batch)

        for rollout_step in range(pl_module.rollout):

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
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"gnn_pred_val_histo_rstep_{rollout_step:02d}_batch{batch_idx:04d}_rank0",
                exp_log_tag=f"val_pred_histo_rstep_{rollout_step:02d}_rank{local_rank:01d}",
            )
