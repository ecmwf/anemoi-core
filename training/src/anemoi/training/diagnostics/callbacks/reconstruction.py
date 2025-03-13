import copy

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Optional
from zipfile import ZipFile

import einops
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torchinfo
from omegaconf import DictConfig, OmegaConf
from omegaconf import ListConfig
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from anemoi.training.diagnostics.callbacks import BaseEvalCallback, BasePlotCallback, GraphTrainableFeaturesPlot, ParentUUIDCallback, WeightGradOutputLoggerCallback
from torch.utils.checkpoint import checkpoint
from anemoi.training.diagnostics.callbacks import BaseLossBarPlot
from anemoi.training.diagnostics.callbacks import AnemoiCheckpoint
from anemoi.training.diagnostics.callbacks import MemCleanUpCallback

from anemoi.training.diagnostics.callbacks.plot import plot_reconstructed_multilevel_sample

from anemoi.training.diagnostics.plots.plots import plot_loss_map
from anemoi.training.diagnostics.plots.plots import plot_loss
from anemoi.training.diagnostics.plots.plots import plot_power_spectrum
from anemoi.training.diagnostics.callbacks import MemCleanUpCallback

from anemoi.training.diagnostics.callbacks import BasePlotCallback, BaseLossMapPlot
from anemoi.training.diagnostics.callbacks import safe_cast_to_numpy
from anemoi.training.diagnostics.callbacks.common_callbacks import get_common_callbacks
import logging
from anemoi.training.diagnostics.plots.plots import plot_predicted_multilevel_flat_sample

from pytorch_lightning.callbacks import Callback
from anemoi.training.losses import BaseWeightedLoss
from anemoi.training.diagnostics import safe_cast_to_numpy
LOGGER = logging.getLogger(__name__)


from anemoi.training.diagnostics.callbacks.plot import PlotLoss
from anemoi.training.diagnostics.callbacks.plot import PlotSample
from anemoi.training.diagnostics.callbacks.plot import PlotSpectrum


class ReconstructionLossBarPlot(PlotLoss):
    """Plots the reconstruction loss accumulated over validation batches and printed once per validation epoch."""
    #NOTE: parameter sub selection not implemented here

    def __init__(self, config, val_dset_len, **kwargs):
        super().__init__(config, parameter_groups=None)

    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.Lightning_module,
        outputs: list[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,) -> None:
        logger = trainer.logger

        parameter_names = list(pl_module.data_indices.internal_model.output.name_to_index.keys())
        parameter_positions = list(pl_module.data_indices.internal_model.output.name_to_index.values())
        # reorder parameter_names by position
        self.parameter_names = [parameter_names[i] for i in np.argsort(parameter_positions)]
        if not isinstance(pl_module.loss, BaseWeightedLoss):
            LOGGER.warning(
                "Loss function must be a subclass of BaseWeightedLoss, or provide `squash`.",
                RuntimeWarning,
            )


        for rollout_step in range(pl_module.rollout):
            # y_hat = outputs[1][rollout_step]
            # y_true = batch[
            #     :,
            #     pl_module.multi_step + rollout_step,
            #     ...,
            #     pl_module.data_indices.internal_data.output.full,
            # ]
            # loss = pl_module.loss(y_hat, y_true, squash=False).cpu().numpy()

            loss = pl_module.loss(
                        preds=outputs["x_rec"][:, rollout_step],
                        target=outputs["x_target"][:, rollout_step],
                        **outputs,
                        squash=(-5, -3, -2),
                    )

            sort_by_parameter_group, colors, xticks, legend_patches = self.sort_and_color_by_parameter_group
            fig = plot_loss(loss[sort_by_parameter_group], colors, xticks, legend_patches)

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"loss_rstep_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
                exp_log_tag=f"loss_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
            )

class PlotReconstructedSample(PlotSample):
    """Plots a denormalized reconstructed sample: input and reconstruction."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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
        # diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic

        params_not_reconstructed = self.config.data.diagnostic or {}
        
        # plot_parameters_dict = {
        #     pl_module.data_indices.model.output.name_to_index[name]: (
        #         name,
        #         name not in diagnostics,
        #     )
        #     for name in self.parameters
        # }

        # NOTE: in (name, bool), the bool means whether or not we want to show the image for the tendency e.g. input-output/pred (True) or just show the output (False)
        plot_parameters_dict_reconstructed_output = {
            pl_module.data_indices.model.output.name_to_index[name]: (name, True) for name in self.plot_parameters if (name in pl_module.data_indices.model.output.name_to_index and name not in params_not_reconstructed)
        }
        plot_parameters_dict_reconstructed_input = {
            pl_module.data_indices.model.input.name_to_index[name]: (name, True) for name in self.plot_parameters if (name in pl_module.data_indices.model.input.name_to_index and name not in params_not_reconstructed)
        }
        plot_parameters_constructed_output = {
            pl_module.data_indices.model.output.name_to_index[name]: (name, True) for name in self.plot_parameters if (name in pl_module.data_indices.model.output.name_to_index and name in params_not_reconstructed)
        }   


        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())
        local_rank = pl_module.local_rank

        x_input = safe_cast_to_numpy(outputs["x_inp_postprocessed"])
        x_rec = safe_cast_to_numpy(outputs["x_rec_postprocessed"])
        x_target = safe_cast_to_numpy(outputs["x_target_postprocessed"])

        time = x_input.shape[1]
        ens_index = 0

        for t_idx in range(time):
                
            # HERE NOTE: here if sample only exists in output then use the OLD plot_multilevel_sample
            if len(plot_parameters_dict_reconstructed_output) > 0:
                x_input_ = x_input[self.sample_idx, t_idx, ens_index]
                x_rec_ = x_rec[self.sample_idx, t_idx, ens_index]
                
                fig = plot_reconstructed_multilevel_sample(
                    plot_parameters_dict_reconstructed_output,
                    plot_parameters_dict_reconstructed_input,
                    self.latlons,
                    x_input_,
                    x_rec_,
                    cmap_precip=self.cmap_accumulation,
                    precip_and_related_fields=self.precip_and_related_fields,
                    
                    )

                self._output_figure(
                    trainer.logger,
                    fig,
                    tag=f"sample_prognostic_epoch_{epoch:03d}_batch{batch_idx:04d}_t{t_idx:02d}_global_step{pl_module.global_step:06d}",
                    exp_log_tag=f"sample_prognostic_rank{local_rank:01d}_t{t_idx:02d}_global_step{pl_module.global_step:06d}",
                )

            # Making plots for outputs that were constructed and not in input
            if len(plot_parameters_constructed_output) > 0:

                x_target_ = x_target[self.sample_idx, t_idx, ens_index]
                x_rec_ = x_rec[self.sample_idx, t_idx, ens_index]

                fig = plot_predicted_multilevel_flat_sample(
                    plot_parameters_constructed_output,
                    n_plots_per_sample=self.per_sample,
                    latlons=self.latlons,
                    clevels=self.accumulation_levels_plot,
                    cmap_precip=self.cmap_accumulation,
                    x = None ,
                    y_true = x_target_,
                    y_pred = x_rec_,
                    precip_and_related_fields=self.precip_and_related_fields,
                )

                self._output_figure(
                    logger=trainer.logger,
                    fig=fig,
                    tag=f"sample_diagnostic_epoch_{epoch:03d}_batch{batch_idx:04d}_t{t_idx:02d}_global_step{pl_module.global_step:06d}",
                    exp_log_tag=f"sample_diagnostic_rank{local_rank:01d}_t{t_idx:02d}_global_step{pl_module.global_step:06d}",
            )

class PlotReconstructionPowerSpectrum(PlotSpectrum):
    """Plots power spectrum metrics comparing target and prediction."""

    def __init__(self, **kwargs):
        """Initialise the PlotPowerSpectrum callback.

        The actual increment (output - input) is plot for prognostic variables while the output is plot for diagnostic ones.

        Parameters
        ----------
        config : OmegaConf
            Config object
        """
        super().__init__(self, **kwargs)

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:

        logger = trainer.logger

        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_output.clone().cpu().numpy())
        
        # NOTE: currently we investigate spatial spectrum only
        x_inp_postprocessed = safe_cast_to_numpy(outputs["x_inp_postprocessed"])
        x_target_postprocessed = safe_cast_to_numpy(outputs["x_target_postprocessed"])
        x_rec_postprocessed = safe_cast_to_numpy(outputs["x_rec_postprocessed"])
        multi_step = x_inp_postprocessed.shape[2]
        
        for t_idx in (self.time_index or range(multi_step)):
            diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic

            plot_parameters_dict_spectrum = {
                pl_module.data_indices.model.output.name_to_index[name]: (name, True) if name in diagnostics else (name, False)
                for name in self.plot_parameters if name in pl_module.data_indices.model.output.name_to_index
            }

            fig = plot_power_spectrum(
                plot_parameters_dict_spectrum,
                self.latlons,
                x_inp_postprocessed[self.sample_idx, t_idx],
                x_target_postprocessed[self.sample_idx, t_idx],
                x_rec_postprocessed[self.sample_idx, t_idx],
            )

            
            self._output_figure(
                logger,
                fig,
                tag=f"power_spectrum_rstep{t_idx:02d}_epoch_{epoch:03d}_batch{batch_idx:04d}_rank{pl_module.local_rank:01d}_global_step{pl_module.global_step:06d}",
                exp_log_tag=f"power_spectrum_rstep_{t_idx:02d}_rank{pl_module.local_rank:01d}_global_step{pl_module.global_step:06d}",
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if self.op_on_this_batch(batch_idx):
            self.plot(trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch)

class CodebookUsageCallback(Callback):
    """Tracks and logs codebook usage statistics during validation."""
    
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.reset_accumulators()
        self.num_codes = config.model.vector_quantizer.vq_config.codebook_size
        self.op_frequency = self.get_op_frequency(kwargs.get("op_frequency", config.diagnostics.plot.codebook_usage.op_frequency), None)

    def reset_accumulators(self):
        """Reset accumulated statistics."""
        self.code_counts = None
        self.total_counts = 0
        
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if not self.op_on_this_batch(batch_idx):
            return
        
        encoding_indices = outputs.get("encoding_indices")
        if encoding_indices is None:
            return
            
        # Convert to numpy if tensor
        if torch.is_tensor(encoding_indices):
            encoding_indices = encoding_indices.detach().cpu().numpy()
            
        # Initialize counts array if first batch
        if self.code_counts is None:
            self.code_counts = np.zeros(self.num_codes)
            
        # Update counts
        unique, counts = np.unique(encoding_indices, return_counts=True)
        self.code_counts[unique] += counts
        self.total_counts += encoding_indices.size
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.code_counts is None:
            return
            
        # Calculate probabilities and entropy
        # I want to reduce the code counts across ranks as a summation across ranks but have the output go to code_counts
        code_counts_tensor = torch.tensor(self.code_counts).to(pl_module.device)
        total_counts_tensor = torch.tensor(self.total_counts).to(pl_module.device)

        dist.reduce(code_counts_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_counts_tensor, dst=0, op=dist.ReduceOp.SUM)

        if pl_module.global_rank == 0:
            
            code_counts_tensor = code_counts_tensor.cpu()
            total_counts_tensor = total_counts_tensor.cpu()

            probs = code_counts_tensor / total_counts_tensor
            valid_probs = probs[probs > 0]  # Avoid log(0)
            entropy = -torch.sum(valid_probs * torch.log(valid_probs))

            # Calculate % of dead codes (used less than 0.1%) e.g occur less than 1 per 1000 codes
            active_codes_001 = torch.sum(probs > 0.001)
            dead_codes_pct_001 = (self.num_codes - active_codes_001) / self.num_codes * 100
            
        # Calculate % of dead codes (used less than 0.01%) e.g occur less than 1 per 10000 codes
            active_codes_00001 = torch.sum(probs > 0.0001)
            dead_codes_pct_00001 = (self.num_codes - active_codes_00001) / self.num_codes * 100


            # Calculate % of dead codes (used less than 0.001%) e.g occur less than 1 per 100000 codes
            active_codes_000001 = torch.sum(probs > 0.00001)
            dead_codes_pct_000001 = (self.num_codes - active_codes_000001) / self.num_codes * 100

            active_codes = torch.sum(probs > 0)
            dead_codes_pct = (self.num_codes - active_codes) / self.num_codes * 100            

            
            # Reduce across ranks
            # Log metrics on rank 0
            pl_module.log("val/codebook/entropy", entropy, 
                        on_epoch=True, on_step=False, sync_dist=False)
            pl_module.log("val/codebook/dead_codes_percent_1pe5", dead_codes_pct_000001,
                        on_epoch=True, on_step=False, sync_dist=False)
            pl_module.log("val/codebook/dead_codes_percent_1pe4", dead_codes_pct_00001,
                        on_epoch=True, on_step=False, sync_dist=False)
            pl_module.log("val/codebook/dead_codes_percent_1pe3", dead_codes_pct_001,
                        on_epoch=True, on_step=False, sync_dist=False)
            pl_module.log("val/codebook/dead_codes_percent", dead_codes_pct,
                        on_epoch=True, on_step=False, sync_dist=False)

        torch.distributed.barrier()

        self.reset_accumulators()

    def get_op_frequency(self, op_frequency: int | float, val_dset_len: int | None) -> int:
        
        if isinstance(op_frequency, int):
            freq = op_frequency

        # If frequency is a float, calculate based on dataset length
        elif isinstance(op_frequency, float):

            raise NotImplementedError("Plot frequency must be an integer as we have not implemented PR for calculating the length of a dataset")
    
        return freq

    def op_on_this_batch(self, batch_idx):
        return ((batch_idx + 1) % self.op_frequency) == 0




