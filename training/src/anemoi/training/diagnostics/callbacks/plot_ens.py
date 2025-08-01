# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_only

from anemoi.training.diagnostics.callbacks.plot import GraphTrainableFeaturesPlot as _GraphTrainableFeaturesPlot
from anemoi.training.diagnostics.callbacks.plot import PlotHistogram as _PlotHistogram
from anemoi.training.diagnostics.callbacks.plot import PlotLoss as _PlotLoss
from anemoi.training.diagnostics.callbacks.plot import PlotSample as _PlotSample
from anemoi.training.diagnostics.callbacks.plot import PlotSpectrum as _PlotSpectrum

if TYPE_CHECKING:
    from typing import Any

    import pytorch_lightning as pl
    from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


class EnsemblePlotMixin:
    """Mixin class for ensemble-specific plotting."""

    def _handle_ensemble_batch_and_output(
        self,
        pl_module: pl.LightningModule,
        output: list[torch.Tensor],
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Handle ensemble batch and output structure.

        Returns
        -------
        tuple
            Processed batch and predictions
        """
        # For ensemble models, batch is a tuple - allgather the full batch first
        batch = pl_module.allgather_batch(batch)
        # Extract ensemble predictions
        loss, y_preds, ens_ic = output
        y_preds = [pl_module.allgather_batch(pred) for pred in y_preds]

        # Return batch[0] (normalized data) and structured output like regular forecaster
        return batch[0] if isinstance(batch, list | tuple) else batch, [loss, y_preds]

    def _extract_first_ensemble_member_from_predictions(self, outputs: list, sample_idx: int) -> list:
        """Extract first ensemble member from prediction to use callbacks from single."""
        if len(outputs) > 1 and isinstance(outputs[1], list):
            ensemble_outputs = []
            for pred in outputs[1]:
                ensemble_outputs.append(pred[sample_idx : sample_idx + 1, 0, ...].cpu())
            return [outputs[0], ensemble_outputs]
        return outputs

    def process(
        self,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process ensemble outputs for metrics plotting.

        Note: Return only the first ensemble member!!!

        Parameters
        ----------
        pl_module : pl.LightningModule
            Lightning module object
        outputs : list
            List of outputs from the model
        batch : torch.Tensor
            Batch tensor (bs, input_steps + forecast_steps, latlon, nvar)

        Returns
        -------
        tuple
            Processed batch and predictions
        """
        # When running in Async mode, it might happen that in the last epoch these tensors
        # have been moved to the cpu (and then the denormalising would fail as the 'input_tensor' would be on CUDA
        # but internal ones would be on the cpu), The lines below allow to address this problem
        if self.post_processors is None:
            # Copy to be used across all the training cycle
            self.post_processors = copy.deepcopy(pl_module.model.post_processors).cpu()
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())

        input_tensor = batch[
            self.sample_idx,
            pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
            ...,
            pl_module.data_indices.data.output.full,
        ].cpu()

        data = self.post_processors(input_tensor)

        # Use the mixin method to extract first ensemble member
        processed_outputs = self._extract_first_ensemble_member_from_predictions(outputs, self.sample_idx)

        output_tensor = self.post_processors(
            torch.cat(tuple(processed_outputs[1])),
            in_place=False,
        )
        output_tensor = pl_module.output_mask.apply(output_tensor, dim=1, fill_value=np.nan).numpy()
        data[1:, ...] = pl_module.output_mask.apply(data[1:, ...], dim=2, fill_value=np.nan)
        data = data.numpy()
        return data, output_tensor


class EnsemblePerBatchPlotMixin(EnsemblePlotMixin):
    """Mixin for per-batch ensemble plotting callbacks."""

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
            processed_batch, processed_output = self._handle_ensemble_batch_and_output(pl_module, output, batch)

            self.plot(
                trainer,
                pl_module,
                processed_output,
                processed_batch,
                batch_idx,
                epoch=trainer.current_epoch,
                **kwargs,
            )


class BaseEnsemblePlotCallback(EnsemblePerBatchPlotMixin):
    """Base class for ensemble plotting callbacks that ensures proper inheritance order."""

    def __init_subclass__(cls, **kwargs):
        """Ensure ensemble mixin comes first in MRO."""
        super().__init_subclass__(**kwargs)
        mro = cls.__mro__

        # Find positions of our key classes
        ensemble_mixin_pos = None
        base_plot_pos = None

        for i, base in enumerate(mro):
            if base.__name__ == "EnsemblePerBatchPlotMixin":
                ensemble_mixin_pos = i
            elif hasattr(base, "__name__") and "BasePerBatchPlotCallback" in base.__name__:
                base_plot_pos = i
                break

        # Warn if ordering might cause issues
        if ensemble_mixin_pos is not None and base_plot_pos is not None and ensemble_mixin_pos > base_plot_pos:
            import warnings

            warnings.warn(
                f"In {cls.__name__}, EnsemblePerBatchPlotMixin should come before "
                f"BasePerBatchPlotCallback in inheritance hierarchy to ensure proper method resolution.",
                UserWarning,
            )


class PlotEnsLoss(_PlotLoss):
    """Plots the unsqueezed loss over rollouts for ensemble models."""

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


class PlotEnsSample(EnsemblePerBatchPlotMixin, _PlotSample):
    """Plots a post-processed ensemble sample: input, target and prediction."""

    def __init__(
        self,
        config: DictConfig,
        sample_idx: int,
        parameters: list[str],
        accumulation_levels_plot: list[float],
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict[str] | None = None,
        per_sample: int = 6,
        every_n_batches: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize PlotSample first
        _PlotSample.__init__(
            self,
            config,
            sample_idx,
            parameters,
            accumulation_levels_plot,
            precip_and_related_fields,
            colormaps,
            per_sample,
            every_n_batches,
            **kwargs,
        )

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[torch.Tensor],  # Now expects [loss, y_preds] format
        batch: torch.Tensor,
        batch_idx: int,
        epoch: int,
    ) -> None:
        from anemoi.training.diagnostics.plots import plot_predicted_ensemble

        logger = trainer.logger

        # Extract y_preds from structured output [loss, y_preds]
        loss, y_preds = outputs

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

        target_data = pl_module.model.post_processors(
            batch[
                self.sample_idx,
                pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
                ...,
                pl_module.data_indices.data.output.full,
            ],
            in_place=False,
        ).cpu()
        target_data = pl_module.output_mask.apply(target_data, dim=2, fill_value=np.nan).numpy()

        # Predictions
        pred = self.post_processors(
            torch.cat(tuple(x[self.sample_idx : self.sample_idx + 1, ...].cpu() for x in y_preds)),
            in_place=False,
        )
        pred = pl_module.output_mask.apply(pred, dim=1, fill_value=np.nan).numpy()

        for rollout_step in range(pl_module.rollout):
            fig = plot_predicted_ensemble(
                plot_parameters_dict,
                4,
                self.latlons,
                self.accumulation_levels_plot,
                target_data[rollout_step + 1, ...].squeeze(),
                pred[rollout_step, ...],
                datashader=self.datashader_plotting,
                initial_condition=False,
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"pred_val_sample_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank0",
                exp_log_tag=f"pred_val_sample_rstep{rollout_step:02d}_rank{local_rank:01d}",
            )


# Overload callbacks from single forecaster by using them with the first ensemble member
# ================================
class PlotSpectrum(BaseEnsemblePlotCallback, _PlotSpectrum):
    """Plots Spectrum of first ensemble member using regular PlotSpectrum logic."""

    def __init__(
        self,
        config: DictConfig,
        sample_idx: int,
        parameters: list[str],
        min_delta: float | None = None,
        every_n_batches: int | None = None,
    ) -> None:
        """Initialise the PlotSpectrum callback."""
        _PlotSpectrum.__init__(self, config, sample_idx, parameters, min_delta, every_n_batches)


class PlotSample(BaseEnsemblePlotCallback, _PlotSample):
    """Plots a post-processed sample using regular PlotSample logic on first ensemble member."""

    def __init__(
        self,
        config: DictConfig,
        sample_idx: int,
        parameters: list[str],
        accumulation_levels_plot: list[float],
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict[str] | None = None,
        per_sample: int = 6,
        every_n_batches: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the PlotSample callback."""
        _PlotSample.__init__(
            self,
            config,
            sample_idx,
            parameters,
            accumulation_levels_plot,
            precip_and_related_fields,
            colormaps,
            per_sample,
            every_n_batches,
            **kwargs,
        )


class PlotHistogram(BaseEnsemblePlotCallback, _PlotHistogram):
    """Plots histograms comparing target and prediction for ensemble models using first member."""

    def __init__(
        self,
        config: DictConfig,
        sample_idx: int,
        parameters: list[str],
        precip_and_related_fields: list[str] | None = None,
        every_n_batches: int | None = None,
    ) -> None:
        """Initialise the PlotHistogram callback."""
        _PlotHistogram.__init__(self, config, sample_idx, parameters, precip_and_related_fields, every_n_batches)


class GraphTrainableFeaturesPlot(_GraphTrainableFeaturesPlot):
    """Visualize the node & edge trainable features for ensemble models."""

    def __init__(self, config: DictConfig, every_n_epochs: int | None = None) -> None:
        """Initialise the GraphTrainableFeaturesPlot callback."""
        _GraphTrainableFeaturesPlot.__init__(self, config, every_n_epochs)
