# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchinfo
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from anemoi.training.diagnostics.plots import init_plot_settings
from anemoi.training.diagnostics.plots import plot_graph_features
from anemoi.training.diagnostics.plots import plot_histogram
from anemoi.training.diagnostics.plots import plot_loss
from anemoi.training.diagnostics.plots import plot_power_spectrum
from anemoi.training.diagnostics.plots import plot_predicted_multilevel_flat_sample

LOGGER = logging.getLogger(__name__)


class PlotCallback(Callback):
    """Factory for creating a callback that plots data to Weights and Biases."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.save_basedir = config.hardware.paths.plots
        self.plot_frequency = config.diagnostics.plot.frequency
        self.normalizer = None
        self.latlons = None
        init_plot_settings()

    def _output_figure(self, logger, fig, epoch: int, tag: str = "gnn") -> None:
        """Figure output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = Path(
                self.save_basedir,
                "plots",
                f"{tag}_epoch{epoch:03d}.png",
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=100, bbox_inches="tight")

            if self.config.diagnostics.log.mlflow.enabled:
                run_id = logger.run_id
                logger.experiment.log_artifact(run_id, str(save_path))

        plt.close(fig)  # cleanup


class AsyncPlotCallback(PlotCallback):
    """Factory for creating a callback that plots data to Weights and Biases."""

    def __init__(self, config) -> None:
        super().__init__(config)

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._error: Optional[BaseException] = None

    def teardown(self, trainer, pl_module, stage) -> None:
        """Close the threads."""
        self._executor.shutdown(wait=True)
        self.check_error()

    def check_error(self) -> None:
        # if an error was raised anytime in any of the `executor.submit` calls
        if self._error:
            raise self._error

    def _plot(
        *args,
        **kwargs,
    ) -> None:
        NotImplementedError

    def _async_plot(
        self,
        trainer,
        *args,
        **kwargs,
    ) -> None:
        """Execute the plot function but ensuring we catch any errors."""
        try:
            if trainer.is_global_zero:
                self._plot(trainer, *args, **kwargs)
        except BaseException as ex:
            self._error = ex


class RolloutEval(Callback):
    """Evaluates the model performance over a (longer) rollout window."""

    def __init__(self, config) -> None:
        """Initialize RolloutEval callback.

        Parameters
        ----------
        config : dict
            Dictionary with configuration settings
        """
        super().__init__()

        LOGGER.setLevel(config.diagnostics.log.code.level)

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
        # NB! the batch is already normalized in-place - see pl_model.validation_step()
        metrics = {}

        # start rollout
        x = batch[
            :, 0 : pl_module.multi_step, ..., pl_module.data_indices.data.input.full
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
                    pl_module.data_indices.data.output.full,
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

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        del trainer, outputs  # not used
        if batch_idx % self.frequency == 3 and pl_module.global_rank == 0:
            self._eval(pl_module, batch)


class GraphTrainableFeaturesPlot(AsyncPlotCallback):
    """Visualize the trainable features defined at the data and hidden graph nodes.

    TODO: How best to visualize the learned edge embeddings? Offline, perhaps - using code from @Simon's notebook?
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._graph_name_data = config.graph.data
        self._graph_name_hidden = config.graph.hidden

    def _plot(
        # self, trainer, latlons:np.ndarray, features:np.ndarray, tag:str, exp_log_tag:str
        self,
        trainer,
        latlons,
        features,
        epoch,
        tag,
        exp_log_tag,
    ) -> None:
        fig = plot_graph_features(latlons, features)
        self._output_figure(trainer.logger, fig, epoch=epoch, tag=tag, exp_log_tag=exp_log_tag)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if pl_module.global_rank == 0:
            model = pl_module.model.module.model if hasattr(pl_module.model, "module") else pl_module.model.model
            graph = pl_module.graph_data.cpu()
            epoch = trainer.current_epoch

            if model.trainable_data is not None:
                data_coords = np.rad2deg(
                    graph[(self._graph_name_data, "to", self._graph_name_data)].ecoords_rad.numpy()
                )

                self._executor.submit(
                    self._async_plot,
                    trainer,
                    data_coords,
                    model.trainable_data.trainable.cpu(),
                    epoch=epoch,
                    tag="trainable_data",
                    exp_log_tag="trainable_data",
                )

            if model.trainable_hidden is not None:
                hidden_coords = np.rad2deg(
                    graph[(self._graph_name_hidden, "to", self._graph_name_hidden)].hcoords_rad.numpy()
                )

                self._executor.submit(
                    self._async_plot,
                    trainer,
                    hidden_coords,
                    model.trainable_hidden.trainable.cpu(),
                    epoch=epoch,
                    tag="trainable_hidden",
                    exp_log_tag="trainable_hidden",
                )

        self.check_error()


class PlotLoss(AsyncPlotCallback):
    """Plots the unsqueezed loss over rollouts."""

    def __init__(self, config) -> None:
        super().__init__(config)

    def _plot(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        epoch,
    ) -> None:
        logger = trainer.logger
        del trainer
        for rollout_step in range(pl_module.rollout):
            y_hat = outputs[1][rollout_step]
            y_true = batch[:, pl_module.multi_step + rollout_step, ..., pl_module.data_indices.data.output.full]
            loss = pl_module.loss(y_hat, y_true, squash=False).cpu().numpy()
            fig = plot_loss(loss)
            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"loss_rstep_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
                exp_log_tag=f"loss_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._async_plot(trainer, pl_module, outputs, batch, epoch=trainer.current_epoch)

        self.check_error()


class PlotSample(AsyncPlotCallback):
    """Plots a denormalized sample: input, target and prediction."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.sample_idx = self.config.diagnostics.plot.sample_idx

    def _plot(
        # batch_idx: int, rollout_step: int, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor,
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        epoch,
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
        if self.normalizer is None:
            # Copy to be used across all the training cycle
            self.normalizer = copy.deepcopy(pl_module.model.normalizer).cpu()
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())
        local_rank = pl_module.local_rank

        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
            ...,
            pl_module.data_indices.data.output.full,
        ].cpu()
        data = self.normalizer.denormalize(input_tensor).numpy()

        output_tensor = self.normalizer.denormalize(
            torch.cat(tuple(x[self.sample_idx : self.sample_idx + 1, ...].cpu() for x in outputs[1])),
            in_place=False,
        ).numpy()

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
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"gnn_pred_val_sample_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank0",
                exp_log_tag=f"val_pred_sample_rstep{rollout_step:02d}_rank{local_rank:01d}",
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._executor.submit(
                self._async_plot, trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch
            )

        self.check_error()


class PlotAdditionalMetrics(AsyncPlotCallback):
    """Plots TP related metric comparing target and prediction.

    The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.

    - Power Spectrum
    - Histograms
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.sample_idx = self.config.diagnostics.plot.sample_idx

    def _plot(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        epoch,
    ) -> None:
        logger = trainer.logger

        # When running in Async mode, it might happen that in the last epoch these tensors
        # have been moved to the cpu (and then the denormalising would fail as the 'input_tensor' would be on CUDA
        # but internal ones would be on the cpu), The lines below allow to address this problem
        if self.normalizer is None:
            # Copy to be used across all the training cycle
            self.normalizer = copy.deepcopy(pl_module.model.normalizer).cpu()
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.data_latlons.clone().cpu().numpy())
        local_rank = pl_module.local_rank

        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
            ...,
            pl_module.data_indices.data.output.full,
        ].cpu()
        data = self.normalizer.denormalize(input_tensor).numpy()
        output_tensor = self.normalizer.denormalize(
            torch.cat(tuple(x[self.sample_idx : self.sample_idx + 1, ...].cpu() for x in outputs[1])),
            in_place=False,
        ).numpy()

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

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._executor.submit(
                self._async_plot, trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch
            )

        self.check_error()


class ParentUUIDCallback(Callback):
    """A callback that retrieves the parent UUID for a model, if it is a child model."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        pl_module.hparams["metadata"]["parent_uuid"] = checkpoint["hyper_parameters"]["metadata"]["uuid"]


class AnemoiCheckpoint(ModelCheckpoint):
    """A checkpoint callback that saves the model after every validation epoch."""

    def __init__(self, config, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.start = time.time()
        self._model_metadata = None
        self._tracker_metadata = None
        self._tracker_name = None

    def _torch_drop_down(self, trainer: pl.Trainer) -> torch.nn.Module:
        # Get the model from the DataParallel wrapper, for single and multi-gpu cases
        assert hasattr(trainer, "model"), "Trainer has no attribute 'model'! Is the Pytorch Lightning version correct?"
        return trainer.model.module.model if hasattr(trainer.model, "module") else trainer.model.model

    def model_metadata(self, model):
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

    def tracker_metadata(self, trainer):
        if self._tracker_metadata is not None:
            return {self._tracker_name: self._tracker_metadata}

        elif self.config.diagnostics.log.mlflow.enabled:
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
            else:
                self._tracker_metadata = {}

        return {self._tracker_name: self._tracker_metadata}

    def _save_checkpoint(self, trainer: pl.Trainer, lightning_checkpoint_filepath: str) -> None:
        if trainer.is_global_zero:
            model = self._torch_drop_down(trainer)

            Path(lightning_checkpoint_filepath).parent.mkdir(parents=True, exist_ok=True)

            # If we are saving the model, we need to remove the config and metadata
            # otherwise they will be twice in the checkpoint, once with the model and once `save_metadata.`
            save_config = model.config
            model.config = None

            save_metadata = model.metadata
            model.metadata = None

            metadata = save_metadata.copy()
            metadata["version"] = "1.0.0"
            # We want a different uuid each time we save the model
            # so we can tell them apart in the catalogue (i.e. different epochs)
            metadata["uuid"] = str(uuid.uuid4())
            metadata["tracker"] = self.tracker_metadata(trainer)
            metadata["training"] = {
                "current_epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "time_since_last_restart": time.time() - self.start,
            }

            inference_checkpoint_filepath = Path(lightning_checkpoint_filepath).parent / Path(
                "inference-" + str(Path(lightning_checkpoint_filepath).name),
            )

            # Save the model
            torch.save(model, inference_checkpoint_filepath)

            # Save the metadata
            save_metadata(inference_checkpoint_filepath, metadata)

            # Save the model info separately, because it is large and not useful for inference, only for display
            save_metadata(inference_checkpoint_filepath, self.model_metadata(model), "model.json")

            # Restore the model's config and metadata
            model.config = save_config
            model.metadata = save_metadata

            self._last_global_step_saved = trainer.global_step

        trainer.strategy.barrier()

        # saving checkpoint used for pytorch-lightning based training
        trainer.save_checkpoint(lightning_checkpoint_filepath, self.save_weights_only)
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = lightning_checkpoint_filepath

        # notify loggers
        if trainer.is_global_zero:
            from weakref import proxy

            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


def get_callbacks(config: DictConfig) -> list:
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
    LOGGER.setLevel(config.diagnostics.log.code.level)

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
        LOGGER.warning("Profiling is enabled - AIFS will not write any training or inference model checkpoints!")

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
                # TODO: do we want the averaging to happen on the CPU, to save memory?
                device=None,
            ),
        )

    trainer_callbacks.append(ParentUUIDCallback(config))

    if config.diagnostics.plot.learned_features:
        LOGGER.debug("Setting up a callback to plot the trainable graph node features ...")
        trainer_callbacks.append(GraphTrainableFeaturesPlot(config))

    return trainer_callbacks
