# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# * [WHY ARE CALLBACKS UNDER __init__.py?]
# * This functionality will be restructured in the near future
# * so for now callbacks are under __init__.py

from __future__ import annotations

import copy
import logging
import sys
import time
import traceback
import uuid
from abc import ABC
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchinfo
from anemoi.utils.checkpoints import save_metadata
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from anemoi.training.diagnostics.plots import init_plot_settings
from anemoi.training.diagnostics.plots import plot_graph_features
from anemoi.training.diagnostics.plots import plot_histogram
from anemoi.training.diagnostics.plots import plot_loss
from anemoi.training.diagnostics.plots import plot_power_spectrum
from anemoi.training.diagnostics.plots import plot_predicted_multilevel_flat_sample

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from omegaconf import DictConfig
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
        self.plot_frequency = config.diagnostics.plot.frequency
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
        *args: list,
        **kwargs: dict,
    ) -> None: ...

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


class RolloutEval(Callback):
    """Evaluates the model performance over a (longer) rollout window."""

    def __init__(self, config: OmegaConf) -> None:
        """Initialize RolloutEval callback.

        Parameters
        ----------
        config : dict
            Dictionary with configuration settings

        """
        super().__init__()

        LOGGER.debug(
            "Setting up RolloutEval callback with rollout = %d, frequency = %d ...",
            config.diagnostics.eval.rollout,
            config.diagnostics.eval.frequency,
        )
        self.rollout = config.diagnostics.eval.rollout
        self.frequency = config.diagnostics.eval.frequency

    def _eval(
        self,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
    ) -> None:
        loss = torch.zeros(1, dtype=batch.dtype, device=pl_module.device, requires_grad=False)
        metrics = {}

        # start rollout
        batch = pl_module.model.pre_processors(batch, in_place=False)
        x = batch[
            :,
            0 : pl_module.multi_step,
            ...,
            pl_module.data_indices.internal_data.input.full,
        ]  # (bs, multi_step, latlon, nvar)
        assert (
            batch.shape[1] >= self.rollout + pl_module.multi_step
        ), "Batch length not sufficient for requested rollout length!"

        with torch.no_grad():
            for rollout_step in range(self.rollout):
                y_pred = pl_module(x)  # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
                y = batch[
                    :,
                    pl_module.multi_step + rollout_step,
                    ...,
                    pl_module.data_indices.internal_data.output.full,
                ]  # target, shape = (bs, latlon, nvar)
                # y includes the auxiliary variables, so we must leave those out when computing the loss
                loss += pl_module.loss(y_pred, y)

                x = pl_module.advance_input(x, y_pred, batch, rollout_step)

                metrics_next, _ = pl_module.calculate_val_metrics(y_pred, y, rollout_step)
                metrics.update(metrics_next)

            # scale loss
            loss *= 1.0 / self.rollout
            self._log(pl_module, loss, metrics, batch.shape[0])

    def _log(self, pl_module: pl.LightningModule, loss: torch.Tensor, metrics: dict, bs: int) -> None:
        pl_module.log(
            f"val_r{self.rollout}_wmse",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=False,
            logger=pl_module.logger_enabled,
            batch_size=bs,
            sync_dist=False,
            rank_zero_only=True,
        )
        for mname, mvalue in metrics.items():
            pl_module.log(
                f"val_r{self.rollout}_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=pl_module.logger_enabled,
                batch_size=bs,
                sync_dist=False,
                rank_zero_only=True,
            )

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        del outputs  # outputs are not used
        if batch_idx % self.frequency == 0:
            precision_mapping = {
                "16-mixed": torch.float16,
                "bf16-mixed": torch.bfloat16,
            }
            prec = trainer.precision
            dtype = precision_mapping.get(prec)
            context = torch.autocast(device_type=batch.device.type, dtype=dtype) if dtype is not None else nullcontext()

            with context:
                self._eval(pl_module, batch)


class LongRolloutPlots(BasePlotCallback):
    """Evaluates the model performance over a (longer) rollout window."""

    def __init__(self, config) -> None:
        """Initialize RolloutEval callback.

        Parameters
        ----------
        config : dict
            Dictionary with configuration settings
        """
        super().__init__(config)

        LOGGER.debug(
            "Setting up callback for plots with long rollout: rollout = %d, frequency = every %d epoch ...",
            config.diagnostics.plot.longrollout.rollout,
            config.diagnostics.plot.longrollout.frequency,
        )
        self.rollout = config.diagnostics.plot.longrollout.rollout
        self.eval_frequency = config.diagnostics.plot.longrollout.frequency
        self.sample_idx = self.config.diagnostics.plot.sample_idx

    @rank_zero_only
    def _plot(
        self,
        trainer,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
        batch_idx,
        epoch,
    ) -> None:

        start_time = time.time()

        logger = trainer.logger

        # Build dictionary of inidicies and parameters to be plotted
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: (
                name,
                name not in self.config.data.get("diagnostic", []),
            )
            for name in self.config.diagnostics.plot.parameters
        }

        if self.post_processors is None:
            # Copy to be used across all the training cycle
            self.post_processors = copy.deepcopy(pl_module.model.post_processors).cpu()
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())
        local_rank = pl_module.local_rank

        batch = pl_module.model.pre_processors(batch, in_place=False)
        # prepare input tensor for rollout from preprocessed batch
        x = batch[
            :,
            0 : pl_module.multi_step,
            ...,
            pl_module.data_indices.internal_data.input.full,
        ]  # (bs, multi_step, latlon, nvar)
        assert (
            batch.shape[1] >= max(self.rollout) + pl_module.multi_step
        ), "Batch length not sufficient for requested rollout length!"

        # prepare input tensor for plotting
        input_tensor_0 = batch[
            self.sample_idx,
            pl_module.multi_step - 1,
            ...,
            pl_module.data_indices.internal_data.output.full,
        ].cpu()
        data_0 = self.post_processors(input_tensor_0).numpy()

        # start rollout
        with torch.no_grad():
            for rollout_step in range(max(self.rollout)):
                y_pred = pl_module(x)  # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)

                x = pl_module.advance_input(x, y_pred, batch, rollout_step)

                if (rollout_step + 1) in self.rollout:
                    # prepare true output tensor for plotting
                    input_tensor_rollout_step = batch[
                        self.sample_idx,
                        pl_module.multi_step + rollout_step,  # (pl_module.multi_step - 1) + (rollout_step + 1)
                        ...,
                        pl_module.data_indices.internal_data.output.full,
                    ].cpu()
                    data_rollout_step = self.post_processors(input_tensor_rollout_step).numpy()

                    # prepare predicted output tensor for plotting
                    output_tensor = self.post_processors(
                        y_pred[self.sample_idx : self.sample_idx + 1, ...].cpu()
                    ).numpy()

                    fig = plot_predicted_multilevel_flat_sample(
                        plot_parameters_dict,
                        self.config.diagnostics.plot.per_sample,
                        self.latlons,
                        self.config.diagnostics.plot.get("accumulation_levels_plot", None),
                        self.config.diagnostics.plot.get("cmap_accumulation", None),
                        data_0.squeeze(),
                        data_rollout_step.squeeze(),
                        output_tensor[0, 0, :, :],  # rolloutstep, first member
                        # force_global_view=self.show_entire_globe,
                    )

                    self._output_figure(
                        logger,
                        fig,
                        epoch=epoch,
                        tag=f"gnn_pred_val_sample_rstep{rollout_step:03d}_batch{batch_idx:04d}_rank0",
                        exp_log_tag=f"val_pred_sample_rstep{rollout_step:03d}_rank{local_rank:01d}",
                    )
        LOGGER.info(f"Time taken to plot samples after longer rollout: {int(time.time() - start_time)} seconds")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, output, batch, batch_idx) -> None:
        if (batch_idx) % self.plot_frequency == 0 and (trainer.current_epoch + 1) % self.eval_frequency == 0:
            precision_mapping = {
                "16-mixed": torch.float16,
                "bf16-mixed": torch.bfloat16,
            }
            prec = trainer.precision
            dtype = precision_mapping.get(prec)
            context = torch.autocast(device_type=batch.device.type, dtype=dtype) if dtype is not None else nullcontext()

            with context:
                self._plot(trainer, pl_module, batch, batch_idx, epoch=trainer.current_epoch)


class GraphTrainableFeaturesPlot(BasePlotCallback):
    """Visualize the trainable features defined at the data and hidden graph nodes.

    TODO: How best to visualize the learned edge embeddings? Offline, perhaps - using code from @Simon's notebook?
    """

    def __init__(self, config: OmegaConf) -> None:
        """Initialise the GraphTrainableFeaturesPlot callback.

        Parameters
        ----------
        config : OmegaConf
            Config object

        """
        super().__init__(config)
        self._graph_name_data = config.graph.data
        self._graph_name_hidden = config.graph.hidden

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        latlons: np.ndarray,
        features: np.ndarray,
        epoch: int,
        tag: str,
        exp_log_tag: str,
    ) -> None:
        fig = plot_graph_features(latlons, features)
        self._output_figure(trainer.logger, fig, epoch=epoch, tag=tag, exp_log_tag=exp_log_tag)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        model = pl_module.model.module.model if hasattr(pl_module.model, "module") else pl_module.model.model
        graph = pl_module.graph_data.cpu().detach()
        epoch = trainer.current_epoch

        if model.trainable_data is not None:
            data_coords = np.rad2deg(graph[(self._graph_name_data, "to", self._graph_name_data)].ecoords_rad.numpy())

            self.plot(
                trainer,
                data_coords,
                model.trainable_data.trainable.cpu().detach().numpy(),
                epoch=epoch,
                tag="trainable_data",
                exp_log_tag="trainable_data",
            )

        if model.trainable_hidden is not None:
            hidden_coords = np.rad2deg(
                graph[(self._graph_name_hidden, "to", self._graph_name_hidden)].hcoords_rad.numpy(),
            )

            self.plot(
                trainer,
                hidden_coords,
                model.trainable_hidden.trainable.cpu().detach().numpy(),
                epoch=epoch,
                tag="trainable_hidden",
                exp_log_tag="trainable_hidden",
            )


class PlotLoss(BasePlotCallback):
    """Plots the unsqueezed loss over rollouts."""

    def __init__(self, config: OmegaConf) -> None:
        """Initialise the PlotLoss callback.

        Parameters
        ----------
        config : OmegaConf
            Object with configuration settings

        """
        super().__init__(config)
        self.parameter_names = None
        self.parameter_groups = self.config.diagnostics.plot.parameter_groups
        if self.parameter_groups is None:
            self.parameter_groups = {}

    @cached_property
    def sort_and_color_by_parameter_group(self) -> tuple[np.ndarray, np.ndarray, dict, list]:
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
                    unique_group_list[tn] if count > 1 or unique_group_list[tn] in self.parameter_groups else "other"
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

        return sort_by_parameter_group, bar_color_per_group[group_inverse], xticks, legend_patches

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

        parameter_names = list(pl_module.data_indices.internal_model.output.name_to_index.keys())
        parameter_positions = list(pl_module.data_indices.internal_model.output.name_to_index.values())
        # reorder parameter_names by position
        self.parameter_names = [parameter_names[i] for i in np.argsort(parameter_positions)]

        batch = pl_module.model.pre_processors(batch, in_place=False)
        for rollout_step in range(pl_module.rollout):
            y_hat = outputs[1][rollout_step]
            y_true = batch[
                :, pl_module.multi_step + rollout_step, ..., pl_module.data_indices.internal_data.output.full
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

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.plot_frequency == 0:
            self.plot(trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch)


class PlotSample(BasePlotCallback):
    """Plots a post-processed sample: input, target and prediction."""

    def __init__(self, config: OmegaConf) -> None:
        """Initialise the PlotSample callback.

        Parameters
        ----------
        config : OmegaConf
            Config object

        """
        super().__init__(config)
        self.sample_idx = self.config.diagnostics.plot.sample_idx
        self.precip_and_related_fields = self.config.diagnostics.plot.precip_and_related_fields
        LOGGER.info(f"Using defined accumulation colormap for fields: {self.precip_and_related_fields}")

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

        # Build dictionary of indices and parameters to be plotted
        diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
            for name in self.config.diagnostics.plot.parameters
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
                self.config.diagnostics.plot.per_sample,
                self.latlons,
                self.config.diagnostics.plot.accumulation_levels_plot,
                self.config.diagnostics.plot.cmap_accumulation,
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

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.Lightning_module,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.plot_frequency == 0:
            self.plot(trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch)


class PlotAdditionalMetrics(BasePlotCallback):
    """Plots TP related metric comparing target and prediction.

    The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.

    - Power Spectrum
    - Histograms
    """

    def __init__(self, config: OmegaConf) -> None:
        """Initialise the PlotAdditionalMetrics callback.

        Parameters
        ----------
        config : OmegaConf
            Config object

        """
        super().__init__(config)
        self.sample_idx = self.config.diagnostics.plot.sample_idx
        self.precip_and_related_fields = self.config.diagnostics.plot.precip_and_related_fields
        LOGGER.info(f"Using precip histogram plotting method for fields: {self.precip_and_related_fields}")

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
            if self.config.diagnostics.plot.parameters_histogram is not None:
                # Build dictionary of inidicies and parameters to be plotted

                diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
                plot_parameters_dict_histogram = {
                    pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
                    for name in self.config.diagnostics.plot.parameters_histogram
                }

                fig = plot_histogram(
                    plot_parameters_dict_histogram,
                    data[0, ...].squeeze(),
                    data[rollout_step + 1, ...].squeeze(),
                    output_tensor[rollout_step, ...],
                    precip_and_related_fields=self.precip_and_related_fields,
                )

                self._output_figure(
                    logger,
                    fig,
                    epoch=epoch,
                    tag=f"gnn_pred_val_histo_rstep_{rollout_step:02d}_batch{batch_idx:04d}_rank0",
                    exp_log_tag=f"val_pred_histo_rstep_{rollout_step:02d}_rank{local_rank:01d}",
                )

            if self.config.diagnostics.plot.parameters_spectrum is not None:
                # Build dictionary of inidicies and parameters to be plotted
                diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic

                plot_parameters_dict_spectrum = {
                    pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
                    for name in self.config.diagnostics.plot.parameters_spectrum
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

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.plot_frequency == 0:
            self.plot(trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch)


class ParentUUIDCallback(Callback):
    """A callback that retrieves the parent UUID for a model, if it is a child model."""

    def __init__(self, config: OmegaConf) -> None:
        """Initialise the ParentUUIDCallback callback.

        Parameters
        ----------
        config : OmegaConf
            Config object

        """
        super().__init__()
        self.config = config

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: torch.nn.Module,
    ) -> None:
        del trainer  # unused
        pl_module.hparams["metadata"]["parent_uuid"] = checkpoint["hyper_parameters"]["metadata"]["uuid"]


class AnemoiCheckpoint(ModelCheckpoint):
    """A checkpoint callback that saves the model after every validation epoch."""

    def __init__(self, config: OmegaConf, **kwargs: dict) -> None:
        """Initialise the AnemoiCheckpoint callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        kwargs : dict
            Additional keyword arguments for Pytorch ModelCheckpoint

        """
        super().__init__(**kwargs)
        self.config = config
        self.start = time.time()
        self._model_metadata = None
        self._tracker_metadata = None
        self._tracker_name = None

    @staticmethod
    def _torch_drop_down(trainer: pl.Trainer) -> torch.nn.Module:
        # Get the model from the DataParallel wrapper, for single and multi-gpu cases
        assert hasattr(trainer, "model"), "Trainer has no attribute 'model'! Is the Pytorch Lightning version correct?"
        return trainer.model.module.model if hasattr(trainer.model, "module") else trainer.model.model

    @rank_zero_only
    def model_metadata(self, model: torch.nn.Module) -> dict:
        if self._model_metadata is not None:
            return self._model_metadata

        self._model_metadata = {
            "model": model.__class__.__name__,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "summary": repr(
                torchinfo.summary(
                    model,
                    depth=50,
                    verbose=0,
                    row_settings=["var_names"],
                ),
            ),
        }

        return self._model_metadata

    def tracker_metadata(self, trainer: pl.Trainer) -> dict:
        if self._tracker_metadata is not None:
            return {self._tracker_name: self._tracker_metadata}

        if self.config.diagnostics.log.wandb.enabled:
            self._tracker_name = "wand"
            import wandb

            run = wandb.run
            if run is not None:
                self._tracker_metadata = {
                    "id": run.id,
                    "name": run.name,
                    "url": run.url,
                    "project": run.project,
                }
            return {self._tracker_name: self._tracker_metadata}

        if self.config.diagnostics.log.mlflow.enabled:
            self._tracker_name = "mlflow"

            from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger

            mlflow_logger = next(logger for logger in trainer.loggers if isinstance(logger, AnemoiMLflowLogger))
            run_id = mlflow_logger.run_id
            run = mlflow_logger._mlflow_client.get_run(run_id)

            if run is not None:
                self._tracker_metadata = {
                    "id": run.info.run_id,
                    "name": run.info.run_name,
                    "url": run.info.artifact_uri,
                    "project": run.info.experiment_id,
                }
            return {self._tracker_name: self._tracker_metadata}

        return {}

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Calls the strategy to remove the checkpoint file."""
        super()._remove_checkpoint(trainer, filepath)
        trainer.strategy.remove_checkpoint(self._get_inference_checkpoint_filepath(filepath))

    def _get_inference_checkpoint_filepath(self, filepath: str) -> str:
        """Defines the filepath for the inference checkpoint."""
        return Path(filepath).parent / Path("inference-" + str(Path(filepath).name))

    def _save_checkpoint(self, trainer: pl.Trainer, lightning_checkpoint_filepath: str) -> None:
        if trainer.is_global_zero:
            model = self._torch_drop_down(trainer)

            # We want a different uuid each time we save the model
            # so we can tell them apart in the catalogue (i.e. different epochs)
            checkpoint_uuid = str(uuid.uuid4())
            trainer.lightning_module._hparams["metadata"]["uuid"] = checkpoint_uuid

            trainer.lightning_module._hparams["metadata"]["model"] = self.model_metadata(model)
            trainer.lightning_module._hparams["metadata"]["tracker"] = self.tracker_metadata(trainer)

            trainer.lightning_module._hparams["metadata"]["training"] = {
                "current_epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "elapsed_time": time.time() - self.start,
            }

            Path(lightning_checkpoint_filepath).parent.mkdir(parents=True, exist_ok=True)

            save_config = model.config
            model.config = None

            tmp_metadata = model.metadata
            model.metadata = None

            metadata = dict(**tmp_metadata)

            inference_checkpoint_filepath = self._get_inference_checkpoint_filepath(lightning_checkpoint_filepath)

            torch.save(model, inference_checkpoint_filepath)

            save_metadata(inference_checkpoint_filepath, metadata)

            model.config = save_config
            model.metadata = tmp_metadata

            self._last_global_step_saved = trainer.global_step

        trainer.strategy.barrier()

        # saving checkpoint used for pytorch-lightning based training
        trainer.save_checkpoint(lightning_checkpoint_filepath, self.save_weights_only)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = lightning_checkpoint_filepath

        if trainer.is_global_zero:
            from weakref import proxy

            # save metadata for the training checkpoint in the same format as inference
            save_metadata(lightning_checkpoint_filepath, metadata)

            # notify loggers
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


def get_callbacks(config: DictConfig) -> list:  # noqa: C901
    """Setup callbacks for PyTorch Lightning trainer.

    Parameters
    ----------
    config : DictConfig
        Job configuration

    Returns
    -------
    List
        A list of PyTorch Lightning callbacks

    """
    checkpoint_settings = {
        "dirpath": config.hardware.paths.checkpoints,
        "verbose": False,
        # save weights, optimizer states, LR-schedule states, hyperparameters etc.
        # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html#contents-of-a-checkpoint
        "save_weights_only": False,
        "auto_insert_metric_name": False,
        # save after every validation epoch, if we've improved
        "save_on_train_epoch_end": False,
        "enable_version_counter": False,
    }

    ckpt_frequency_save_dict = {}
    for key, frequency_dict in config.diagnostics.checkpoint.items():
        frequency = frequency_dict["save_frequency"]
        n_saved = frequency_dict["num_models_saved"]
        if key == "every_n_minutes" and frequency_dict["save_frequency"] is not None:
            target = "train_time_interval"
            frequency = timedelta(minutes=frequency_dict["save_frequency"])
        else:
            target = key
        ckpt_frequency_save_dict[target] = (config.hardware.files.checkpoint[key], frequency, n_saved)

    trainer_callbacks = []
    if not config.diagnostics.profiler:
        for save_key, (name, save_frequency, save_n_models) in ckpt_frequency_save_dict.items():
            if save_frequency is not None:
                LOGGER.debug("Checkpoint callback at %s = %s ...", save_key, save_frequency)
                trainer_callbacks.extend(
                    # save_top_k: the save_top_k flag can either save the best or the last k checkpoints
                    # depending on the monitor flag on ModelCheckpoint.
                    # See https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html for reference
                    [
                        AnemoiCheckpoint(
                            config=config,
                            filename=name,
                            save_last=True,
                            **{save_key: save_frequency},
                            # if save_top_k == k, last k models saved; if save_top_k == -1, all models are saved
                            save_top_k=save_n_models,
                            monitor="step",
                            mode="max",
                            **checkpoint_settings,
                        ),
                    ],
                )
            else:
                LOGGER.debug("Not setting up a checkpoint callback with %s", save_key)
    else:
        # the tensorboard logger + pytorch profiler cause pickling errors when writing checkpoints
        LOGGER.warning("Profiling is enabled - will not write any training or inference model checkpoints!")

    if any([config.diagnostics.log.wandb.enabled, config.diagnostics.log.mlflow.enabled]):
        from pytorch_lightning.callbacks import LearningRateMonitor

        trainer_callbacks.append(
            LearningRateMonitor(
                logging_interval="step",
                log_momentum=False,
            ),
        )

    if config.diagnostics.eval.enabled:
        trainer_callbacks.append(RolloutEval(config))

    if config.diagnostics.plot.enabled:
        trainer_callbacks.extend(
            [
                PlotLoss(config),
                PlotSample(config),
            ],
        )
        if (config.diagnostics.plot.parameters_histogram or config.diagnostics.plot.parameters_spectrum) is not None:
            trainer_callbacks.extend([PlotAdditionalMetrics(config)])
        if config.diagnostics.plot.get("longrollout") and config.diagnostics.plot.longrollout.enabled:
            trainer_callbacks.extend([LongRolloutPlots(config)])

    if config.training.swa.enabled:
        from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

        trainer_callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=config.training.swa.lr,
                swa_epoch_start=min(
                    int(0.75 * config.training.max_epochs),
                    config.training.max_epochs - 1,
                ),
                annealing_epochs=max(int(0.25 * config.training.max_epochs), 1),
                annealing_strategy="cos",
                device=None,
            ),
        )

    trainer_callbacks.append(ParentUUIDCallback(config))

    if config.diagnostics.plot.learned_features:
        LOGGER.debug("Setting up a callback to plot the trainable graph node features ...")
        trainer_callbacks.append(GraphTrainableFeaturesPlot(config))
    return trainer_callbacks
