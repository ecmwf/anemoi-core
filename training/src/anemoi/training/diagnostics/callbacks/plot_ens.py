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
    from typing import Union

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
        loss, y_preds, _ = output
        y_preds = [pl_module.allgather_batch(pred) for pred in y_preds]

        # Return batch[0] (normalized data) and structured output like regular forecaster
        return batch[0] if isinstance(batch, list | tuple) else batch, [loss, y_preds]

    def process(
        self,
        pl_module: pl.LightningModule,
        outputs: list,
        batch: torch.Tensor,
        members: Union[int, list[int]] = 0,
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
        members : int, list[int], optional
            Ensemble members to plot. If None, all members are returned. Default to 0.

        Returns
        -------
        tuple
            Processed batch and predictions
        """
        # When running in Async mode, it might happen that in the last epoch these tensors
        # have been moved to the cpu (and then the denormalising would fail as the 'input_tensor' would be on CUDA
        # but internal ones would be on the cpu), The lines below allow to address this problem
        if self.latlons is None:
            self.latlons = np.rad2deg(pl_module.latlons_data.clone().cpu().numpy())

        input_tensor = (
            batch[
                :,
                pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
                ...,
                pl_module.data_indices.data.output.full,
            ]
            .detach()
            .cpu()
        )
        data = self.post_processors(input_tensor)[self.sample_idx]
        output_tensor = torch.cat(
            tuple(
                self.post_processors(x[:, ...].detach().cpu(), in_place=False)[
                    self.sample_idx : self.sample_idx + 1,
                    members,
                    ...,
                ]
                for x in outputs[1]
            ),
        )

        output_tensor = pl_module.output_mask.apply(output_tensor, dim=-2, fill_value=np.nan).numpy()
        data[1:, ...] = pl_module.output_mask.apply(data[1:, ...], dim=-2, fill_value=np.nan)
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

            # When running in Async mode, it might happen that in the last epoch these tensors
            # have been moved to the cpu (and then the denormalising would fail as the 'input_tensor' would be on CUDA
            # but internal ones would be on the cpu), The lines below allow to address this problem
            self.post_processors = copy.deepcopy(pl_module.model.post_processors)
            for post_processor in self.post_processors.processors.values():
                if hasattr(post_processor, "nan_locations"):
                    post_processor.nan_locations = pl_module.allgather_batch(post_processor.nan_locations)
            self.post_processors = self.post_processors.cpu()

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
        members: list | None = None,
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
        self.plot_members = members

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

        # Build dictionary of indices and parameters to be plotted
        diagnostics = [] if self.config.data.diagnostic is None else self.config.data.diagnostic
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: (name, name not in diagnostics)
            for name in self.config.diagnostics.plot.parameters
        }

        data, output_tensor = self.process(pl_module, outputs, batch, members=self.plot_members)

        local_rank = pl_module.local_rank
        for rollout_step in range(pl_module.rollout):
            fig = plot_predicted_ensemble(
                parameters=plot_parameters_dict,
                n_plots_per_sample=4,
                latlons=self.latlons,
                clevels=self.accumulation_levels_plot,
                y_true=data[rollout_step + 1, ...].squeeze(),
                y_pred=output_tensor[rollout_step, ...].squeeze(),
                datashader=self.datashader_plotting,
                precip_and_related_fields=self.precip_and_related_fields,
                colormaps=self.colormaps,
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"pred_val_sample_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank{local_rank:01d}",
                exp_log_tag=f"pred_val_sample_rstep{rollout_step:02d}_rank{local_rank:01d}",
            )


# Overload callbacks from single forecaster by using them with the first ensemble member
# ================================
class PlotLoss(_PlotLoss):
    """Plots the unsqueezed loss over rollouts for ensemble models."""

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        super().on_validation_batch_end(
            trainer,
            pl_module,
            outputs,
            batch[0][:, :, 0, :, :],
            batch_idx,
        )


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
