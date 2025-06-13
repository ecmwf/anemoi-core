import copy
import logging
import time
from contextlib import nullcontext
from typing import Any

import einops
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from anemoi.models.distributed.graph import gather_tensor
from anemoi.training.diagnostics.callbacks.evaluation import RolloutEval
from anemoi.training.diagnostics.callbacks.plot import BasePerBatchPlotCallback
from anemoi.training.diagnostics.callbacks.plot import LongRolloutPlots
from anemoi.training.diagnostics.callbacks.plot import PlotHistogram
from anemoi.training.diagnostics.callbacks.plot import PlotLoss
from anemoi.training.diagnostics.callbacks.plot import PlotSample
from anemoi.training.diagnostics.callbacks.plot import PlotSpectrum
from anemoi.training.diagnostics.plots import plot_histogram
from anemoi.training.diagnostics.plots import plot_power_spectrum
from anemoi.training.diagnostics.plots import plot_predicted_ensemble

# from aifs.diagnostics.metrics.ranks import RankHistogram
# from aifs.diagnostics.metrics.spread import SpreadSkill

LOGGER = logging.getLogger(__name__)


class BasePerBatchEnsPlotCallback(BasePerBatchPlotCallback):
    """Base Callback for plotting at the end of each batch."""

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        **kwargs,
    ) -> None:
        if (
            self.config.diagnostics.plot.asynchronous
            and self.config.dataloader.read_group_size > 1
            and pl_module.local_rank == 0
        ):
            LOGGER.warning("Asynchronous plotting can result in NCCL timeouts with reader_group_size > 1.")

        if batch_idx % self.every_n_batches == 0:
            batch = pl_module.allgather_batch(batch[0])

            self.plot(
                trainer,
                pl_module,
                output,
                batch,
                batch_idx,
                epoch=trainer.current_epoch,
                **kwargs,
            )


class RolloutEvalEns(RolloutEval):
    """Evaluates the model performance over a (longer) rollout window."""

    def _eval(self, pl_module: pl.LightningModule, batch: torch.Tensor, ens_ic: torch.Tensor) -> None:
        """Rolls out the model and calculates the validation metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            Lightning module object
        batch: torch.Tensor
            Batch tensor (bs, input_steps + forecast_steps, latlon, nvar)
        ens_ic: torch.Tensor
            Ensemble initial condition
        """
        x = ens_ic
        loss = torch.zeros(1, dtype=batch[0].dtype, device=pl_module.device, requires_grad=False)
        # NB! the batch is already normalized in-place - see pl_model.validation_step()
        metrics = {}

        assert (
            batch[0].shape[1] >= self.rollout + pl_module.multi_step
        ), "Batch length not sufficient for requested rollout length!"

        with torch.no_grad():
            for loss_next, metrics_next, _, _ in pl_module.rollout_step(
                batch=batch,
                rollout=self.rollout,
                training_mode=True,
                validation_mode=True,
            ):
                loss += loss_next
                metrics.update(metrics_next)

            # scale loss
            loss *= 1.0 / self.rollout
            self._log(pl_module, loss, metrics, batch[0].shape[0])

    def _log(self, pl_module: pl.LightningModule, loss: torch.Tensor, metrics: dict, bs: int) -> None:
        pl_module.log(
            f"rval_r{self.rollout}_" + pl_module.loss.name,
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=False,
            logger=pl_module.logger_enabled,
            batch_size=bs,
            sync_dist=True,
            rank_zero_only=True,
        )
        for mname, mvalue in metrics.items():
            pl_module.log(
                f"rval_r{self.rollout}_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=pl_module.logger_enabled,
                batch_size=bs,
                sync_dist=True,
                rank_zero_only=True,
            )

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ) -> None:
        """Plotting function to be implemented by subclasses."""

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.every_n_batches == 0 and pl_module.ens_comm_group_rank == 0:
            precision_mapping = {
                "16-mixed": torch.float16,
                "bf16-mixed": torch.bfloat16,
            }
            prec = trainer.precision
            dtype = precision_mapping.get(prec)
            context = (
                torch.autocast(device_type=batch[0].device.type, dtype=dtype) if dtype is not None else nullcontext()
            )

            with context:
                self._eval(pl_module, batch, outputs[-1])


# class RankHistogramPlot(BasePerBatchEnsPlotCallback):
#     """Plots spread skill rank histogram."""

#     def __init__(self, config: DictConfig, rollout: int):
#         super().__init__(config)

#         self.rollout = rollout

#         self.nens_per_group = (
#             config.training.ensemble_size_per_device * config.hardware.num_gpus_per_ensemble // config.hardware.num_gpus_per_model
#         )

#         # Rank histogram (accumulates statistics for _all_ output variables, both prognostic and diagnostic)
#         self.ranks = RankHistogram(rollout=self.rollout, nens=self.nens_per_group, nvar=len(config.diagnostics.plot.parameters))

#     def _eval(self, pl_module: pl.LightningModule, batch: torch.Tensor, ens_ic: torch.Tensor):

#         x = ens_ic

#         with torch.no_grad():
#             for rollout_step, (_, _, y_pred_group, y_true) in enumerate(
#                 pl_module.rollout_step(
#                     x,
#                     batch,
#                     rollout=pl_module.rollout,
#                     validation_mode=True,
#                     training_mode=True,
#                 ),
#             ):
#                 # denormalize:
#                 y_denorm = pl_module.model.post_processors(y_true, in_place=False)  # ground truth
#                 y_pred_denorm = pl_module.model.post_processors(y_pred_group, in_place=False)  # ens

#                 # rank histograms - update metric state
#                 _ = self.ranks(rollout_step, y_denorm, y_pred_denorm, pl_module.device)

#     def _plot(self, trainer: pl.Trainer, pl_module: pl.LightningModule, epoch_tag: int) -> None:
#         # Build dictionary of inidicies and parameters to be plotted
#         diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
#         plot_parameters_dict = {
#             pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
#             for name in self.config.diagnostics.plot.parameters
#         }

#         for rollout_step in range(pl_module.rollout):
#             fig = plot_rank_histograms(plot_parameters_dict, self.ranks.ranks.cpu().numpy()[rollout_step])
#             self._output_figure(
#                 trainer,
#                 fig,
#                 epoch=epoch_tag,
#                 tag=f"ens_rank_hist_rstep{rollout_step:02d}",
#                 exp_log_tag=f"val_rank_hist_rstep{rollout_step:02d}_rank{pl_module.global_rank}",
#             )
#         self.ranks.reset()

#     def on_validation_batch_end(
#         self,
#         trainer: pl.Trainer,
#         pl_module: pl.LightningModule,
#         outputs: Any,
#         batch: torch.Tensor,
#         batch_idx: int,
#     ) -> None:

#         if batch_idx % self.every_n_batches == 0 and pl_module.ens_comm_group_rank == 0:
#             precision_mapping = {
#                 "16-mixed": torch.float16,
#                 "bf16-mixed": torch.bfloat16,
#             }
#             prec = trainer.precision
#             dtype = precision_mapping.get(prec)
#             context = torch.autocast(device_type=batch[0].device.type, dtype=dtype) if dtype is not None else nullcontext()

#             with context:
#                 self._eval(pl_module, batch[0], outputs[-1])
#             self.plot(trainer.logger, pl_module, epoch_tag=trainer.current_epoch)


# class SpreadSkillPlot(BasePerBatchEnsPlotCallback):
#     """Plots spread skill diagonal plot."""

#     def __init__(self, config: DictConfig, nbins: int, rollout: int, plot_bins=True):
#         config.diagnostics.plot.asynchronous = False
#         super().__init__(config)

#         self.rollout = rollout
#         self.spread_skill = SpreadSkill(
#             rollout=self.rollout,
#             nvar=len(config.diagnostics.plot.parameters),
#             nbins=nbins,
#         )
#         self.eval_plot_parameters = config.diagnostics.plot.parameters
#         self.nbins = nbins
#         self.plot_bins = plot_bins

#     def _eval(self, pl_module: pl.LightningModule, batch: torch.Tensor, ens_ic: torch.Tensor):

#         x = ens_ic

#         with torch.no_grad():
#             rmse = torch.zeros((self.rollout, len(self.eval_plot_parameters)), dtype=batch.dtype, device=pl_module.device)
#             spread = torch.zeros_like(rmse)

#             binned_rmse = torch.zeros(
#                 (self.rollout, len(self.eval_plot_parameters), self.nbins - 1),
#                 dtype=batch.dtype,
#                 device=pl_module.device,
#             )
#             binned_spread = torch.zeros_like(binned_rmse)

#             node_weights = (
#                 pl_module.graph_data[pl_module.config.graph.data][pl_module.config.model.node_loss_weight]
#                 .squeeze()
#                 .to(device=pl_module.device)
#             )

#             diagnostics = self.config.data.diagnostic or []
#             plot_parameters_dict = {
#                 pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
#                 for name in self.config.diagnostics.plot.parameters
#             }

#             for rollout_step, (_, _, y_pred_group, y_true) in enumerate(
#                 pl_module.rollout_step(
#                     x,
#                     batch,
#                     rollout=self.rollout,
#                     validation_mode=True,
#                     training_mode=True,
#                 ),
#             ):
#                 # denormalize:
#                 y_denorm = pl_module.model.post_processors(y_true, in_place=False)  # ground truth
#                 y_pred_denorm = pl_module.model.post_processors(y_pred_group, in_place=False)  # ens

#                 # spread-skill diagnostic
#                 for midx, (pidx, _) in enumerate(plot_parameters_dict.items()):
#                     (
#                         rmse[rollout_step, midx],
#                         spread[rollout_step, midx],
#                         binned_rmse[rollout_step, midx],
#                         binned_spread[rollout_step, midx],
#                     ) = self.spread_skill.calculate_spread_skill(y_pred_denorm, y_denorm, pidx, node_weights)

#             # update spread-skill metric state
#             _ = self.spread_skill(rmse, spread, binned_rmse, binned_spread, pl_module.device)

#     def _plot(
#         self,
#         trainer: pl.Trainer,
#         pl_module: pl.LightningModule,
#         epoch_tag: int,
#     ) -> None:
#         # Build dictionary of inidicies and parameters to be plotted
#         diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
#         plot_parameters_dict = {
#             pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
#             for name in self.config.diagnostics.plot.parameters
#         }

#         rmse, spread, bins_rmse, bins_spread = (r.cpu().numpy() for r in self.spread_skill.compute())
#         fig = plot_spread_skill(plot_parameters_dict, (rmse, spread), self.spread_skill.time_step)
#         self._output_figure(
#             trainer,
#             fig,
#             epoch=epoch_tag,
#             tag="ens_spread_skill",
#             exp_log_tag=f"val_spread_skill_{pl_module.global_rank}",
#         )
#         if self.plot_bins:
#             fig = plot_spread_skill_bins(plot_parameters_dict, (bins_rmse, bins_spread), self.spread_skill.time_step)
#             self._output_figure(
#                 trainer,
#                 fig,
#                 epoch=epoch_tag,
#                 tag="ens_spread_skill_bins",
#                 exp_log_tag=f"val_spread_skill_bins_{pl_module.global_rank}",
#             )

#     def on_validation_batch_end(
#         self,
#         trainer: pl.Trainer,
#         pl_module: pl.LightningModule,
#         outputs: Any,
#         batch: torch.Tensor,
#         batch_idx: int,
#     ) -> None:
#         if batch_idx % self.every_n_batches == 0 and pl_module.ens_comm_group_rank == 0:
#             precision_mapping = {
#                 "16-mixed": torch.float16,
#                 "bf16-mixed": torch.bfloat16,
#             }
#             prec = trainer.precision
#             dtype = precision_mapping.get(prec)
#             context = torch.autocast(device_type=batch[0].device.type, dtype=dtype) if dtype is not None else nullcontext()
#             with context:
#                 self._eval(pl_module, batch[0], outputs[-1])
#             self.plot(trainer.logger, pl_module, epoch_tag=trainer.current_epoch)


class PlotEnsLoss(PlotLoss):
    """Plots the unsqueezed loss over rollouts."""

    # def _plot(
    #     self,
    #     trainer: pl.Trainer,
    #     pl_module: pl.LightningModule,
    #     outputs: list[torch.Tensor],
    #     batch: torch.Tensor,
    #     batch_idx: int,
    #     epoch: int,
    # ) -> None:
    #     logger = trainer.logger
    #     _ = batch_idx

    #     parameter_names = list(pl_module.data_indices.internal_model.output.name_to_index.keys())
    #     parameter_positions = list(pl_module.data_indices.internal_model.output.name_to_index.values())
    #     # reorder parameter_names by position
    #     self.parameter_names = [parameter_names[i] for i in np.argsort(parameter_positions)]
    #     if not isinstance(pl_module.loss, BaseLoss):
    #         logging.warning(
    #             "Loss function must be a subclass of BaseWeightedLoss, or provide `squash`.",
    #             RuntimeWarning,
    #         )

    #     for rollout_step in range(pl_module.rollout):
    #         y_hat = outputs[1][rollout_step]
    #         y_true = batch[
    #             :,
    #             pl_module.multi_step + rollout_step,
    #             ...,
    #             pl_module.data_indices.internal_data.output.full,
    #         ]
    #         loss = pl_module.loss(y_hat, y_true, squash=False).cpu().numpy()
    #         sort_by_parameter_group, colors, xticks, legend_patches = self.sort_and_color_by_parameter_group
    #         fig = plot_loss(loss[sort_by_parameter_group], colors, xticks, legend_patches)

    #         self._output_figure(
    #             logger,
    #             fig,
    #             epoch=epoch,
    #             tag=f"loss_rstep_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
    #             exp_log_tag=f"loss_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
    #         )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.every_n_batches == 0 and pl_module.ens_comm_group_rank == 0:
            self._plot(
                trainer,
                pl_module,
                outputs,
                batch[0][:, :, 0, :, :],
                batch_idx=None,
                epoch=trainer.current_epoch,
            )


class PlotEnsembleInitialConditions(BasePerBatchEnsPlotCallback):
    """Plots ensemble initial conditions."""

    def __init__(self, config: DictConfig, accumulation_levels_plot: list[float], cmap_accumulation: list[str]):
        super().__init__(config)
        self.sample_idx = self.config.diagnostics.plot.sample_idx
        self.accumulation_levels_plot = accumulation_levels_plot
        self.cmap_accumulation = cmap_accumulation

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        ens_ic: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        # Build dictionary of inidicies and parameters to be plotted
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

        input_tensor = ens_ic.cpu()

        denorm_ens_ic = (
            self.post_processors.processors.normalizer.inverse_transform(
                input_tensor,
                in_place=False,
                data_index=pl_module.data_indices.data.input.full,
            )
        ).numpy()

        for step in range(pl_module.multi_step):
            # plot the perturbations at current step
            fig = plot_predicted_ensemble(
                plot_parameters_dict,
                1,
                np.rad2deg(pl_module.latlons_data.numpy()),
                self.accumulation_levels_plot,
                self.cmap_accumulation,
                denorm_ens_ic[self.sample_idx, step, ...].squeeze(),
                denorm_ens_ic[self.sample_idx, step, ...].squeeze(),
                datashader=self.datashader_plotting,
                initial_condition=True,
            )
            fig.tight_layout()
            self._output_figure(
                trainer,
                fig,
                epoch=epoch,
                tag=f"ens_ic_val_mstep{step:02d}_batch{batch_idx:05d}_rank{pl_module.global_rank:03d}",
                exp_log_tag=f"ens_ic_val_mstep{step:02d}_rank{pl_module.global_rank:03d}",
            )

    def _gather_group_initial_conditions(self, pl_module: pl.LightningModule, my_ens_ic: torch.Tensor) -> torch.Tensor:
        """Gathers all the initial conditions in a device group to a single tensor."""
        group_ens_ic = gather_tensor(
            my_ens_ic,
            dim=1,
            shapes=[my_ens_ic.shape] * pl_module.ens_comm_group_size,
            mgroup=pl_module.ens_comm_group,
        )
        return group_ens_ic

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        del batch  # not used
        # gather all ICs (may contain duplicates)
        group_ens_ic = self._gather_group_initial_conditions(pl_module, outputs[-1])
        # remove duplicates
        group_ens_ic = einops.rearrange(
            group_ens_ic,
            "bs s e latlon v -> bs s v latlon e",
        )  # ensemble dim must come last
        group_ens_ic = group_ens_ic @ pl_module._gather_matrix
        group_ens_ic = einops.rearrange(
            group_ens_ic,
            "bs s v latlon e -> bs s e latlon v",
        )  # reshape back to what it was
        # plotting happens only on ens_comm_group_rank == 0 (one device per group)
        # ens_comm_group_rank == 0 has gathered all initial conditions generated by its group
        if batch_idx % self.every_n_batches == 0 and pl_module.ens_comm_group_rank == 0:
            self.plot(trainer.logger, pl_module, group_ens_ic, batch_idx, epoch=trainer.current_epoch)


class PlotEnsSample(PlotSample):

    def _plot(
        # batch_idx: int, rollout_step: int, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor,
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

        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
            ...,
            pl_module.data_indices.data.output.full,
        ].cpu()
        data = self.post_processors(input_tensor)

        output_tensor = self.post_processors(
            torch.cat(tuple(x[self.sample_idx : self.sample_idx + 1, ...].cpu() for x in outputs[1])),
            in_place=False,
        )
        output_tensor = pl_module.output_mask.apply(output_tensor, dim=1, fill_value=np.nan).numpy()
        data[1:, ...] = pl_module.output_mask.apply(data[1:, ...], dim=2, fill_value=np.nan)
        data = data.numpy()

        for rollout_step in range(pl_module.rollout):
            fig = plot_predicted_ensemble(
                plot_parameters_dict,
                4,
                self.latlons,
                self.accumulation_levels_plot,
                data[rollout_step + 1, ...].squeeze(),
                output_tensor[rollout_step, ...],
                datashader=self.datashader_plotting,
                initial_condition=False,
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
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        # plotting happens only on ens_comm_group_rank == 0 (one device per group)
        # ens_comm_group_rank == 0 has gathered all ensemble members generated by its group
        if batch_idx % self.every_n_batches == 0 and pl_module.ens_comm_group_rank == 0:
            self.plot(trainer, pl_module, outputs, batch[0], batch_idx, epoch=trainer.current_epoch)


class PlotEnsHistogram(PlotHistogram):
    """Plots histograms comparing target and prediction.

    The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.
    """

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
        # data, output_tensor = self.process(pl_module, outputs, batch[0])

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
        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
            ...,
            pl_module.data_indices.data.output.full,
        ].cpu()
        data = input_tensor.numpy()
        output_tensor = (
            torch.cat(tuple(x[self.sample_idx : self.sample_idx + 1, ...].cpu() for x in outputs[1]))
        ).numpy()
        for rollout_step in range(pl_module.rollout):

            # Build dictionary of indices and parameters to be plotted
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
                output_tensor[rollout_step, 0, ...],
                self.precip_and_related_fields,
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"gnn_pred_val_histo_rstep_{rollout_step:02d}_batch{batch_idx:04d}_rank0",
                exp_log_tag=f"val_pred_histo_rstep_{rollout_step:02d}_rank{local_rank:01d}",
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        output: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        **kwargs,
    ) -> None:
        if (
            self.config.diagnostics.plot.asynchronous
            and self.config.dataloader.read_group_size > 1
            and pl_module.local_rank == 0
        ):
            LOGGER.warning("Asynchronous plotting can result in NCCL timeouts with reader_group_size > 1.")

        if batch_idx % self.every_n_batches == 0:
            batch = pl_module.allgather_batch(batch[0])

            self.plot(
                trainer,
                pl_module,
                output,
                batch,
                batch_idx,
                epoch=trainer.current_epoch,
                **kwargs,
            )


class LongEnsRolloutPlots(LongRolloutPlots):
    """Evaluates the model performance over a (longer) rollout window.

    This function allows evaluating the performance of the model over an extended number
    of rollout steps to observe long-term behavior.
    Add the callback to the configuration file as follows:
    ```
      - _target_:  anemoi.training.diagnostics.callbacks.plot.LongRolloutPlots
        rollout:
            - ${dataloader.validation_rollout}
        video_rollout: ${dataloader.validation_rollout}
        every_n_epochs: 1
        sample_idx: ${diagnostics.plot.sample_idx}
        parameters: ${diagnostics.plot.parameters}
    ```
    The selected rollout steps for plots and video need to be lower or equal to dataloader.validation_rollout.
    Increasing dataloader.validation_rollout has no effect on the rollout steps during training.
    It ensures, that enough time steps are available for the plots and video in the validation batches.

    The runtime of creating one animation of one variable for 56 rollout steps is about 1 minute.
    Recommended use for video generation: Fork the run using fork_run_id for 1 additional epochs and enabled videos.
    """

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        start_time = time.time()
        logger = trainer.logger

        x = outputs[-1]
        # Initialize required variables for plotting
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: (
                name,
                name not in self.config.data.get("diagnostic", []),
            )
            for name in self.parameters
        }
        if self.post_processors is None:
            self.post_processors = copy.deepcopy(pl_module.model.post_processors).cpu()
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())

        assert batch.shape[1] >= self.max_rollout + pl_module.multi_step, (
            "Batch length not sufficient for requested validation rollout length! "
            f"Set `dataloader.validation_rollout` to at least {max(self.rollout)}"
        )

        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step - 1,
            ...,
            pl_module.data_indices.data.output.full,
        ].cpu()
        data_0 = self.post_processors(input_tensor).numpy()

        if self.video_rollout:
            data_over_time = []
            # collect min and max values for each variable for the colorbar
            vmin, vmax = (np.inf * np.ones(len(plot_parameters_dict)), -np.inf * np.ones(len(plot_parameters_dict)))

        # Plot for each rollout step# Plot for each rollout step
        with torch.no_grad():
            for rollout_step, (_, _, y_pred, _) in enumerate(
                pl_module.rollout_step(
                    x,
                    batch,
                    rollout=self.max_rollout,
                    validation_mode=True,
                    training_mode=True,
                ),
            ):
                # plot only if the current rollout step is in the list of rollout steps
                if (rollout_step + 1) in self.rollout:
                    self._plot_rollout_step(
                        pl_module,
                        plot_parameters_dict,
                        batch,
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

    def _plot_rollout_step(
        self,
        pl_module: pl.LightningModule,
        plot_parameters_dict: dict,
        batch: torch.Tensor,
        rollout_step: int,
        y_pred: torch.Tensor,
        batch_idx: int,
        epoch: int,
        logger: pl.loggers,
    ) -> None:
        """Plot predicted output, input, target and error for given rollout."""
        # prepare true output tensor for plotting
        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step + rollout_step,
            ...,
            pl_module.data_indices.data.output.full,
        ].cpu()
        data_rollout_step = self.post_processors(input_tensor, in_place=False).numpy()
        # predicted output tensor
        output_tensor = self.post_processors(y_pred[self.sample_idx : self.sample_idx + 1, ...].cpu()).numpy()

        fig = plot_predicted_ensemble(
            plot_parameters_dict,
            4,
            self.latlons,
            self.accumulation_levels_plot,
            self.cmap_accumulation,
            data_rollout_step.squeeze(),
            output_tensor[0],
            datashader=self.datashader_plotting,
            initial_condition=False,
        )

        self._output_figure(
            logger,
            fig,
            epoch=epoch,
            tag=f"gnn_pred_val_sample_rstep{rollout_step + 1:03d}_batch{batch_idx:04d}_rank0",
            exp_log_tag=f"val_pred_sample_rstep{rollout_step + 1:03d}_rank{pl_module.local_rank:01d}",
        )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if (batch_idx) == 0 and (trainer.current_epoch + 1) % self.every_n_epochs == 0:

            batch = pl_module.allgather_batch(batch)
            precision_mapping = {
                "16-mixed": torch.float16,
                "bf16-mixed": torch.bfloat16,
            }
            prec = trainer.precision
            dtype = precision_mapping.get(prec)
            context = (
                torch.autocast(device_type=batch[0].device.type, dtype=dtype) if dtype is not None else nullcontext()
            )

            if self.config.diagnostics.plot.asynchronous:
                LOGGER.warning("Asynchronous plotting not supported for long rollout plots.")

            with context:
                # Issue with running asyncronously, so call the plot function directly
                self._plot(trainer, pl_module, outputs, batch[0], batch_idx, trainer.current_epoch)


class PlotEnsSpectrum(PlotSpectrum):
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
        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
            ...,
            pl_module.data_indices.data.output.full,
        ].cpu()
        data = self.post_processors(input_tensor).numpy()
        output_tensor = self.post_processors(
            torch.cat(tuple(x[self.sample_idx : self.sample_idx + 1, ...].cpu() for x in outputs[1])),
            in_place=False,
        ).numpy()
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
                output_tensor[rollout_step, 0, ...],  # power spectra for ensemble member 0
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
        output: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        **kwargs,
    ) -> None:
        if (
            self.config.diagnostics.plot.asynchronous
            and self.config.dataloader.read_group_size > 1
            and pl_module.local_rank == 0
        ):
            LOGGER.warning("Asynchronous plotting can result in NCCL timeouts with reader_group_size > 1.")

        if batch_idx % self.every_n_batches == 0:
            batch = pl_module.allgather_batch(batch[0])

            self.plot(
                trainer,
                pl_module,
                output,
                batch,
                batch_idx,
                epoch=trainer.current_epoch,
                **kwargs,
            )
