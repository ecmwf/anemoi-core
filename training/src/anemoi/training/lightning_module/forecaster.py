# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections import defaultdict
from collections.abc import Generator
from typing import Optional, Union, Mapping

import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface.forecast import AnemoiModelForecastInterface
from anemoi.training.lightning_module.base import AnemoiLightningModule
from anemoi.training.losses.utils import grad_scaler
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class ForecastLightningModule(AnemoiLightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint
        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        graph_data = graph_data.to(self.device)

        if config.model.get("output_mask", None) is not None:
            self.output_mask = Boolean1DMask(graph_data[config.graph.data][config.model.output_mask])
        else:
            self.output_mask = NoOutputMask()

        self.model = AnemoiModelForecastInterface(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays | self.output_mask.supporting_arrays,
            graph_data=graph_data,
            config=DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )

        self.latlons_data = graph_data[config.graph.data].x
        self.node_weights = self.get_node_weights(config, graph_data)
        self.node_weights = self.output_mask.apply(self.node_weights, dim=0, fill_value=0.0)

        variable_scaling = self.get_variable_scaling(config, data_indices)

        self.internal_metric_ranges, self.val_metric_ranges = self.get_val_metric_ranges(config, data_indices)

        # Check if the model is a stretched grid
        if graph_data["hidden"].node_type == "StretchedTriNodes":
            mask_name = config.graph.nodes.hidden.node_builder.mask_attr_name
            limited_area_mask = graph_data[config.graph.data][mask_name].squeeze().bool()
        else:
            limited_area_mask = torch.ones((1,))

        # Kwargs to pass to the loss function
        loss_kwargs = {"node_weights": self.node_weights}
        # Scalars to include in the loss function, must be of form (dim, scalar)
        # Use -1 for the variable dimension, -2 for the latlon dimension
        # Add mask multiplying NaN locations with zero. At this stage at [[1]].
        # Filled after first application of preprocessor. dimension=[-2, -1] (latlon, n_outputs).
        self.scalars = {
            "variable": (-1, variable_scaling),
            "loss_weights_mask": ((-2, -1), torch.ones((1, 1))),
            "limited_area_mask": (2, limited_area_mask),
        }
        self.updated_loss_mask = False

        self.loss = self.get_loss_function(config.training.training_loss, scalars=self.scalars, **loss_kwargs)

        assert isinstance(self.loss, BaseWeightedLoss) and not isinstance(
            self.loss,
            torch.nn.ModuleList,
        ), f"Loss function must be a `BaseWeightedLoss`, not a {type(self.loss).__name__!r}"

        self.metrics = self.get_loss_function(config.training.validation_metrics, scalars=self.scalars, **loss_kwargs)
        if not isinstance(self.metrics, torch.nn.ModuleList):
            self.metrics = torch.nn.ModuleList([self.metrics])

        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.multi_step = config.training.multistep_input
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

    @staticmethod
    def get_val_metric_ranges(config: DictConfig, data_indices: IndexCollection) -> tuple[dict, dict]:
        """Get validation metric ranges.

        Parameters
        ----------
        config : DictConfig
            Configuration
        data_indices : IndexCollection
            Data indices

        Returns
        -------
        tuple[dict, dict]
            Internal metric ranges and validation metric ranges
        """
        metric_ranges = defaultdict(list)
        metric_ranges_validation = defaultdict(list)

        for key, idx in data_indices.internal_model.output.name_to_index.items():
            split = key.split("_")
            if len(split) > 1 and split[-1].isdigit():
                # Group metrics for pressure levels (e.g., Q, T, U, V, etc.)
                metric_ranges[f"pl_{split[0]}"].append(idx)
            else:
                metric_ranges[f"sfc_{key}"].append(idx)

            # Specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges[key] = [idx]

        # Add the full list of output indices
        metric_ranges["all"] = data_indices.internal_model.output.full.tolist()

        # metric for validation, after postprocessing
        for key, idx in data_indices.model.output.name_to_index.items():
            # Split pressure levels on "_" separator
            split = key.split("_")
            if len(split) > 1 and split[1].isdigit():
                # Create grouped metrics for pressure levels (e.g. Q, T, U, V, etc.) for logger
                metric_ranges_validation[f"pl_{split[0]}"].append(idx)
            else:
                metric_ranges_validation[f"sfc_{key}"].append(idx)
            # Create specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges_validation[key] = [idx]

        # Add the full list of output indices
        metric_ranges_validation["all"] = data_indices.model.output.full.tolist()

        return metric_ranges, metric_ranges_validation

    @staticmethod
    def get_variable_scaling(
        config: DictConfig,
        data_indices: IndexCollection,
    ) -> torch.Tensor:
        """Get variable scaling.

        Parameters
        ----------
        config : DictConfig
            Configuration
        data_indices : IndexCollection
            Data indices

        Returns
        -------
        torch.Tensor
            Variable scaling tensor
        """
        variable_loss_scaling = (
            np.ones((len(data_indices.internal_data.output.full),), dtype=np.float32)
            * config.training.variable_loss_scaling.default
        )
        pressure_level = instantiate(config.training.pressure_level_scaler)

        LOGGER.info(
            "Pressure level scaling: use scaler %s with slope %.4f and minimum %.2f",
            type(pressure_level).__name__,
            pressure_level.slope,
            pressure_level.minimum,
        )

        for key, idx in data_indices.internal_model.output.name_to_index.items():
            split = key.split("_")
            if len(split) > 1 and split[-1].isdigit():
                # Apply pressure level scaling
                if split[0] in config.training.variable_loss_scaling.pl:
                    variable_loss_scaling[idx] = config.training.variable_loss_scaling.pl[
                        split[0]
                    ] * pressure_level.scaler(
                        int(split[-1]),
                    )
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            else:
                # Apply surface variable scaling
                if key in config.training.variable_loss_scaling.sfc:
                    variable_loss_scaling[idx] = config.training.variable_loss_scaling.sfc[key]
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)

        return torch.from_numpy(variable_loss_scaling)

    @staticmethod
    def get_node_weights(config: DictConfig, graph_data: HeteroData) -> torch.Tensor:
        """Get node weights.

        Parameters
        ----------
        config : DictConfig
            Configuration
        graph_data : HeteroData
            Graph data

        Returns
        -------
        torch.Tensor
            Node weights tensor
        """
        node_weighting = instantiate(config.training.node_loss_weights)
        return node_weighting.weights(graph_data)

    def training_weights_for_imputed_variables(
        self,
        batch: torch.Tensor,
    ) -> None:
        """Update the loss weights mask for imputed variables.

        Parameters
        ----------
        batch : torch.Tensor
            Batch tensor
        """
        if "loss_weights_mask" in self.loss.scalar:
            loss_weights_mask = torch.ones((1, 1), device=batch.device)
            found_loss_mask_training = False
            # iterate over all pre-processors and check if they have a loss_mask_training attribute
            for pre_processor in self.model.pre_processors.processors.values():
                if hasattr(pre_processor, "loss_mask_training"):
                    loss_weights_mask = loss_weights_mask * pre_processor.loss_mask_training
                    found_loss_mask_training = True
                # if transform_loss_mask function exists for preprocessor apply it
                if hasattr(pre_processor, "transform_loss_mask") and found_loss_mask_training:
                    loss_weights_mask = pre_processor.transform_loss_mask(loss_weights_mask)
            # update scaler with loss_weights_mask retrieved from preprocessors
            self.loss.update_scalar(scalar=loss_weights_mask.cpu(), name="loss_weights_mask")
            self.scalars["loss_weights_mask"] = ((-2, -1), loss_weights_mask.cpu())

        self.updated_loss_mask = True

    def advance_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int,
    ) -> torch.Tensor:
        """Advance input for rollout.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        y_pred : torch.Tensor
            Predicted output tensor
        batch : torch.Tensor
            Batch tensor
        rollout_step : int
            Rollout step

        Returns
        -------
        torch.Tensor
            Advanced input tensor
        """
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, self.data_indices.internal_model.input.prognostic] = y_pred[
            ...,
            self.data_indices.internal_model.output.prognostic,
        ]

        x[:, -1] = self.output_mask.rollout_boundary(x[:, -1], batch[:, -1], self.data_indices)

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, self.data_indices.internal_model.input.forcing] = batch[
            :,
            self.multi_step + rollout_step,
            :,
            :,
            self.data_indices.internal_data.input.forcing,
        ]
        return x

    def rollout_step(
        self,
        batch: torch.Tensor,
        rollout: Optional[int] = None,
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]:
        """Rollout step for the forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : torch.Tensor
            Batch to use for rollout
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        training_mode : bool, optional
            Whether in training mode and to calculate the loss, by default True
            If False, loss will be None
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)
        """
        # for validation not normalized in-place because remappers cannot be applied in-place
        batch = self.model.pre_processors(batch, in_place=not validation_mode)

        if not self.updated_loss_mask:
            # update loss scalar after first application and initialization of preprocessors
            self.training_weights_for_imputed_variables(batch)

        # start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.internal_data.input.full,
        ]  # (bs, multi_step, latlon, nvar)
        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            y_pred = self(x)

            y = batch[:, self.multi_step + rollout_step, ..., self.data_indices.internal_data.output.full]
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss = checkpoint(self.loss, y_pred, y, use_reentrant=False) if training_mode else None

            x = self.advance_input(x, y_pred, batch, rollout_step)

            metrics_next = {}
            if validation_mode:
                metrics_next = self.calculate_val_metrics(
                    y_pred,
                    y,
                    rollout_step,
                )
            yield loss, metrics_next, y_pred

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Perform a step (training or validation).

        Parameters
        ----------
        batch : torch.Tensor
            Batch tensor
        batch_idx : int
            Batch index
        validation_mode : bool, optional
            Whether in validation mode, by default False

        Returns
        -------
        tuple[torch.Tensor, Mapping[str, torch.Tensor]]
            Loss, metrics, and predictions
        """
        del batch_idx
        batch = self.allgather_batch(batch)

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        for loss_next, metrics_next, y_preds_next in self.rollout_step(
            batch,
            rollout=self.rollout,
            training_mode=True,
            validation_mode=validation_mode,
        ):
            loss += loss_next
            metrics.update(metrics_next)
            y_preds.extend(y_preds_next)

        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds

    def calculate_val_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int,
    ) -> tuple[dict, list[torch.Tensor]]:
        """Calculate metrics on the validation output.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted ensemble
        y: torch.Tensor
            Ground truth (target).
        rollout_step: int
            Rollout step

        Returns
        -------
        dict
            Validation metrics
        """
        metrics = {}
        y_postprocessed = self.model.post_processors(y, in_place=False)
        y_pred_postprocessed = self.model.post_processors(y_pred, in_place=False)

        for metric in self.metrics:
            metric_name = getattr(metric, "name", metric.__class__.__name__.lower())

            if not isinstance(metric, BaseWeightedLoss):
                # If not a weighted loss, we cannot feature scale, so call normally
                metrics[f"{metric_name}/{rollout_step + 1}"] = metric(
                    y_pred_postprocessed,
                    y_postprocessed,
                )
                continue

            for mkey, indices in self.val_metric_ranges.items():
                if "scale_validation_metrics" in self.config.training and (
                    mkey in self.config.training.scale_validation_metrics.metrics
                    or "*" in self.config.training.scale_validation_metrics.metrics
                ):
                    with metric.scalar.freeze_state():
                        for key in self.config.training.scale_validation_metrics.scalars_to_apply:
                            metric.add_scalar(*self.scalars[key], name=key)

                        # Use internal model space indices
                        internal_model_indices = self.internal_metric_ranges[mkey]

                        metrics[f"{metric_name}/{mkey}/{rollout_step + 1}"] = metric(
                            y_pred,
                            y,
                            scalar_indices=[..., internal_model_indices],
                        )
                else:
                    if -1 in metric.scalar:
                        exception_msg = (
                            "Validation metrics cannot be scaled over the variable dimension"
                            " in the post processed space. Please specify them in the config"
                            " at `scale_validation_metrics`."
                        )
                        raise ValueError(exception_msg)

                    metrics[f"{metric_name}/{mkey}/{rollout_step + 1}"] = metric(
                        y_pred_postprocessed,
                        y_postprocessed,
                        scalar_indices=[..., indices],
                    )

        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Perform a training step.

        Parameters
        ----------
        batch : torch.Tensor
            Batch tensor
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Loss tensor
        """
        train_loss, _, _ = self._step(batch, batch_idx)
        self.log(
            f"train_{getattr(self.loss, 'name', self.loss.__class__.__name__.lower())}",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )
        return train_loss

    def on_train_epoch_end(self) -> None:
        """Handle end of training epoch, potentially increasing rollout length."""
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a validation step.

        Parameters
        ----------
        batch : torch.Tensor
            Batch tensor
        batch_idx : int
            Batch index

        Returns
        -------
        tuple
            Validation loss and predictions
        """
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, batch_idx, validation_mode=True)

        self.log(
            f"val_{getattr(self.loss, 'name', self.loss.__class__.__name__.lower())}",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )

        for mname, mvalue in metrics.items():
            self.log(
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch.shape[0],
                sync_dist=True,
            )

        return val_loss, y_preds