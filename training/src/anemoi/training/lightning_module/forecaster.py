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
from torch.nn import ModuleList

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface.forecast import AnemoiModelForecastInterface
from anemoi.models.interface.latent_forecast import AnemoiModelInterfaceVAEForecasting
from anemoi.training.lightning_module.base import AnemoiLightningModule
from anemoi.training.losses.utils import grad_scaler
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.utils.config import DotDict
from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.utils.debug_hydra import instantiate_debug

LOGGER = logging.getLogger(__name__)

def get_time_step(timestep_hours: int, rollout_steps: int) -> str:
    """Format a time step string based on hours and rollout steps.
    
    Parameters
    ----------
    timestep_hours : int
        Hours per time step
    rollout_steps : int
        Number of rollout steps
        
    Returns
    -------
    str
        Formatted time step string (e.g., "24h" for 24 hours)
    """
    hours = timestep_hours * rollout_steps
    if hours % 24 == 0 and hours >= 24:
        return f"{hours//24}d"
    return f"{hours}h"

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
    


class LightningModuleLatentForecasting(AnemoiLightningModule):
    """Graph neural network latent forecaster for PyTorch Lightning.
    
    This class specializes in training a forecasting model that operates in the latent
    space of a pre-trained VAE model. It loads a VAE from a checkpoint and uses it
    to encode inputs to latent space, forecast in latent space, and decode predictions.
    """

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
        """Initialize latent space forecasting model.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data
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

        # Move graph data to device
        graph_data = graph_data.to(self.device)

        # Initialize output mask
        if config.model.get("output_mask", None) is not None:
            self.output_mask = Boolean1DMask(graph_data[config.graph.data][config.model.output_mask])
        else:
            self.output_mask = NoOutputMask()

        # Initialize the model with VAE forecasting interface
        self.model = AnemoiModelInterfaceVAEForecasting(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays | self.output_mask.supporting_arrays,
            graph_data=graph_data,
            config=DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )

        # Store latent representation type from the model
        self.latent_representation = self.model.model.latent_representation_format  # "continuous" or "discrete"
        
        # Configure VAE freezing if needed
        # NOTE: This needs to be extended to allow params to be unfrozen sometime into training
        LOGGER.info("Currently we only have functionality for VAE params to be frozen/non frozen from start of training. Will extend this to allow staggered unfreezing w/ custom scheduler")
        if getattr(config.model.vae, "freeze_parameters", False):
            self.model.model.freeze_parameters()
            
        
        # Get necessary attributes for training
        self.latlons_data = self.model.model.vae_interface.graph_data[self.model.model._graph_name_inp].x
        self.node_weights = self.model.model.vae_interface.graph_data[self.model.model._graph_name_inp][config.model.node_loss_weight].squeeze(-1)

        self.latent_node_weights = self.model.model.vae_interface.graph_data[self.model.model._graph_name_hidden][config.model.node_loss_weight].squeeze(-1)

        # Rollout configuration
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max
        self.multi_step = config.training.multistep_input

        # Loss and metrics initialization
        variable_scaling = self.get_variable_scaling(config, data_indices)
        self.internal_metric_ranges, self.val_metric_ranges = self.get_val_metric_ranges(config, data_indices)

        # Set up loss function
        loss_kwargs = {
            "node_weights": self.latent_node_weights,
            "feature_weights": self.latent_node_weights.new_ones([self.model.model.vae_interface.model.latent_space_dim]),
        }
        
        # Scalars to include in the loss function
        self.scalars = {
            "variable": (-1, variable_scaling),
            "loss_weights_mask": ((-2, -1), torch.ones((1, 1))),
        }
        
        # Initialize loss function
        self.loss = self.get_loss_function(config.training.loss, scalars=self.scalars, **loss_kwargs)
        
        # Initialize validation metrics
        self.metrics = self.get_loss_function(config.training.validation_metrics, scalars=self.scalars, **loss_kwargs)
        if not isinstance(self.metrics, torch.nn.ModuleList):
            self.metrics = torch.nn.ModuleList([self.metrics])
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Execute a training step.
        
        This follows the same pattern as ForecastLightningModule but adds logging
        for the current rollout length.
        
        Parameters
        ----------
        batch : torch.Tensor
            Input batch
        batch_idx : int
            Batch index
            
        Returns
        -------
        torch.Tensor
            Training loss
        """
        # Use the standard training step from the base class
        train_loss = super().training_step(batch, batch_idx)

        # Log the current rollout value
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
        """Adjust rollout length at the end of each training epoch."""
        super().on_train_epoch_end()
        
        # Increment rollout if configured
        if self.rollout_epoch_increment > 0 and (self.current_epoch + 1) % self.rollout_epoch_increment == 0:
            self.rollout = min(self.rollout + 1, self.rollout_max)
            LOGGER.info(f"Increased rollout to {self.rollout}")

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
        batch_target: Optional[torch.Tensor] = None,
        use_checkpoint: bool = True,
        validation_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], dict]:
        """Execute a single step (training or validation).
        
        This method:
        1. Encodes the input to latent space using the VAE encoder
        2. Performs a multi-step rollout in latent space
        3. Decodes predictions back to observation space (for validation only)
        
        Parameters
        ----------
        batch : torch.Tensor
            Input batch
        batch_idx : int
            Batch index
        validation_mode : bool, optional
            Whether running in validation mode, by default False
        batch_target : Optional[torch.Tensor], optional
            Optional target batch, by default None
        use_checkpoint : bool, optional
            Whether to use gradient checkpointing, by default True
        validation_kwargs : dict | None, optional
            Additional validation arguments, by default None
            
        Returns
        -------
        tuple[torch.Tensor, Mapping[str, torch.Tensor], dict]
            Loss, metrics, and output tensors
        """
        validation_kwargs = validation_kwargs or {}
        
        # Preprocess the input batch
        batch_preprocessed = self.model.pre_processors_state(batch, in_place=False) 
        
        with torch.no_grad():
            if batch_target is not None:
                # Target is provided for validation
                batch_inverse = self.model.post_processors_state(batch_target, in_place=False, inverse=True)
                batch_latent_target = self.model.model.encode(
                    batch_inverse[:, self.multi_step:self.multi_step + self.rollout, ..., self.data_indices.internal_data.input.full]
                )
                batch_recon_target = batch_inverse[:, self.multi_step:self.multi_step + self.rollout, ..., self.data_indices.internal_data.output.full]
            else:
                # For training, use the batch itself as the target
                batch_inverse = self.model.post_processors_state(batch, in_place=False, inverse=True)
                batch_latent_target = self.model.model.encode(
                    batch_inverse[:, self.multi_step:self.multi_step + self.rollout, ..., self.data_indices.internal_data.input.full]
                )
                batch_recon_target = batch_inverse[:, self.multi_step:self.multi_step + self.rollout, ..., self.data_indices.internal_data.output.full]

            # Encode input sequence to latent representation
            x = batch_preprocessed[:, 0:self.multi_step, ..., self.data_indices.internal_data.input.full]
            x_encoded = self.model.model.encode(x)
        
        # Get latent state from encoding (different for discrete and continuous latent spaces)
        x_latent_state = self.model.model.get_latent_representation(x_encoded, sample_from_latent_space=False)
        
        # Initialize tensor to store predictions
        y_preds_latent = torch.zeros_like(batch_latent_target)
        
        # Rollout forecasting in latent space
        for rollout_step in range(self.rollout):
            y_pred_latent = self.model.model.forward_latent_state(x_latent_state, qauntize_next_latent=False) # NOTE: this needs to be updated. w/ beta vae we don't need to quantize, but w/ vq vae it should be optional to project back to a restrained state space
            y_preds_latent[:, rollout_step:rollout_step + 1, ...] = y_pred_latent
            x_latent_state = self.advance_latent_input(x_latent_state, y_pred_latent)
        
        # Compute loss (with checkpointing if requested)
        if use_checkpoint:  
            loss = checkpoint(self.loss, y_preds_latent, batch_latent_target, squash=True, use_reentrant=False)
        else:
            loss = self.loss(y_preds_latent, batch_latent_target, squash=True)
        
        # Scale loss by rollout length (losses don't do any temporal scaling)
        loss *= 1.0 / self.rollout
        
        outputs = {}
        metrics = {}
        
        # For validation, decode predictions and compute metrics
        if validation_mode:
            # Decode predictions differently based on latent representation type
            if self.latent_representation == "discrete":
                y_quantized = self.model.model.get_latent_representation(y_preds_latent)
                y_preds = self.model.model.decode_latent_state(y_quantized, model_comm_group=self.model_comm_group)
            else: 
                y_preds = self.model.model.decode_latent_state(y_preds_latent) 

            y = batch_recon_target

            outputs_ = self.get_proc_and_unproc_data(y_pred=y_preds, y=y)
            metrics = self.calculate_val_metrics(**outputs_, **validation_kwargs)

        return loss, metrics, outputs_
        

    def get_proc_and_unproc_data(self, y_pred=None, y=None, y_pred_postprocessed=None, y_postprocessed=None):        
        assert y_pred is not None or y_pred_postprocessed is not None, "Either y_pred or y_pred_postprocessed must be provided"
        assert y is not None or y_postprocessed is not None, "Either y or y_postprocessed must be provided"

        if y_postprocessed is None:
            y_postprocessed = self.model.post_processors_state(y, in_place=False)
        if y_pred_postprocessed is None:
            y_pred_postprocessed = self.model.post_processors_state(y_pred, in_place=False)

        if y_pred is None:
            y_pred = self.model.pre_processors_state(y_pred_postprocessed, in_place=False, data_index=self.data_indices.data.output.full)
        if y is None:
            y = self.model.pre_processors_state(y_postprocessed, in_place=False, data_index=self.data_indices.data.output.full)

        return {
            "y": y,
            "y_pred": y_pred,
            "y_postprocessed": y_postprocessed,
            "y_pred_postprocessed": y_pred_postprocessed,
        }

    def calculate_val_metrics(self, y_pred, y, y_pred_postprocessed, y_postprocessed, time_index:list[int]|None=None, val_metrics:Optional[ModuleList]=None, val_metric_ranges:Optional[dict]=None, **kwargs):
        """
        Calculate validation metrics.

        Parameters
        ----------
        y_pred : torch.Tensor (bs, ens_pred, timesteps, latlon, nvar)
            Predicted output tensor.
        y : torch.Tensor (bs, timesteps, ens_target latlon, nvar)
            Target output tensor.
        y_pred_postprocessed : torch.Tensor (bs, end_pred, timesteps, latlon, nvar)
            Postprocessed predicted output tensor.
        y_postprocessed : torch.Tensor (bs, end_target, timesteps, latlon, nvar)
            Postprocessed target output tensor.

        Returns
        -------
        dict[str, torch.Tensor]
        
        """
        metric_vals = {}
        timesteps = y_pred.shape[1]

        if val_metrics is None:
            val_metrics = self.val_metrics
        if val_metric_ranges is None:
            val_metric_ranges = self.val_metric_ranges

        for metric in self.metrics:
            metric_name = getattr(metric, "name", metric.__class__.__name__.lower())

            if not isinstance(metric, BaseWeightedLoss):
                # If not a weighted loss, we cannot feature scale, so call normally

                # metrics[f"{metric_name}/{rollout_step + 1}"] = metric(
                #     y_pred_postprocessed,
                #     y_postprocessed,
                # )
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
                        
                        timesteps = x_rec.shape[1]

                        for ts in range(timesteps):

                            if len(internal_model_indices) == 1:                                
                                _args = (y_pred_postprocessed[:, ts, ..., internal_model_indices], y_postprocessed[:, ts, ..., internal_model_indices])
                                feature_scale = False
                            else:
                                _args = (y_pred[:, ts, ..., indices], y[:, ts, ..., indices])
                                feature_scale = True

                            metrics[f"{metric_name}/{mkey}/{ts + 1}"] = metric(
                                *_args,
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

                    for ts in range(timesteps):

                        metrics[f"{metric_name}/{mkey}/{ts + 1}"] = metric(
                            y_pred_postprocessed[:, ts, ..., indices],
                            y_postprocessed[:, ts, ..., indices],
                            scalar_indices=[..., indices],
                        )

        return metrics

    def advance_latent_input(
        self,
        x_latent_state: torch.Tensor,
        y_pred_latent: torch.Tensor,
    ) -> torch.Tensor:
        """

        This operates on non-processed data or processed data. The user must ensure both x and y_pred are either non-processed or processed.

        x: (bs, multi_step, ens, latlon, nvars_full)
        y_pred: (bs, timesteps, ens_pred, latlon, nvars_full)
        batch: (bs, timesteps, ens_target latlon, nvars_full)
        
        """
        x_latent_state = x_latent_state.roll(-1, dims=-4)

        # Get prognostic variables
        x_latent_state[:, -1:, ...] = y_pred_latent

        return x_latent_state
    
    def validation_output_and_metrics(self, y_pred, y, validation_kwargs={} ):
        
    
        outputs_ = self.get_proc_and_unproc_data(y_pred=y_pred, y=y)
        metrics = self.calculate_val_metrics(**outputs_, **validation_kwargs)

        return metrics, outputs_
