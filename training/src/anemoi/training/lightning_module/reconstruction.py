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
from collections.abc import Mapping
from typing import Optional, Union, Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface.reconstruct import AnemoiModelReconstructInterface
from anemoi.training.losses.utils import grad_scaler
from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask
from anemoi.utils.config import DotDict
from anemoi.training.losses.reconstruction import VAELoss, VQLoss

LOGGER = logging.getLogger(__name__)


class ReconstructionLightningModule(AnemoiLightningModule):
    """VAE-based reconstruction model for PyTorch Lightning."""

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
        """Initialize reconstruction model.

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

        self.model = AnemoiModelReconstructInterface(
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

        # Scalars to include in the loss function
        self.scalars = {
            "variable": (-1, variable_scaling),
            "loss_weights_mask": ((-2, -1), torch.ones((1, 1))),
        }

        # Initialize loss based on model type
        self.model_type = self.model.model.latent_representation_format
        if self.model_type == "continuous":  # Beta-VAE
            self.loss = VAELoss(
                reconstruction_loss=self.get_loss_function(
                    config.training.reconstruction_loss, 
                    scalars=self.scalars, 
                    node_weights=self.node_weights
                ),
                kl_weight=config.training.kl_weight,
            )
        elif self.model_type == "discrete":  # VQ-VAE
            self.loss = VQLoss(
                reconstruction_loss=self.get_loss_function(
                    config.training.reconstruction_loss, 
                    scalars=self.scalars, 
                    node_weights=self.node_weights
                ),
                commitment_weight=config.training.commitment_weight,
                codebook_weight=config.training.codebook_weight,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Set up metrics for validation
        self.metrics = self.get_loss_function(config.training.validation_metrics, scalars=self.scalars, node_weights=self.node_weights)
        if not isinstance(self.metrics, torch.nn.ModuleList):
            self.metrics = torch.nn.ModuleList([self.metrics])

        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.multi_step = config.training.multistep_input
        
    @staticmethod
    def get_variable_scaling(
        config: DictConfig,
        data_indices: IndexCollection,
    ) -> torch.Tensor:
        """Get variable scaling for reconstruction loss.

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
        # Simplified variable scaling for reconstruction
        variable_loss_scaling = torch.ones(len(data_indices.internal_data.output.full)) * config.training.get("variable_loss_scaling", 1.0)
        return variable_loss_scaling

    @staticmethod
    def get_node_weights(config: DictConfig, graph_data: HeteroData) -> torch.Tensor:
        """Get node weights for the reconstruction loss.

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
        from anemoi.training.utils.node_weights import NodeWeights
        node_weighting = NodeWeights()
        return node_weighting.weights(graph_data)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        Dict[str, Any]
            Model outputs
        """
        return self.model(x)

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], Dict[str, Any]]:
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
        tuple
            Loss, metrics, and predictions
        """
        del batch_idx
        batch = self.allgather_batch(batch)
        
        # Apply pre-processing
        batch = self.model.pre_processors(batch, in_place=not validation_mode)
        
        # Extract input and target
        x = batch[:, :self.multi_step, ..., self.data_indices.internal_data.input.full]
        y = batch[:, 0, ..., self.data_indices.internal_data.output.full]  # First timestep for reconstruction
        
        # Forward pass through model
        outputs = self(x)
        
        # Calculate loss depending on model type
        if self.model_type == "continuous":  # Beta-VAE
            loss = self.loss(
                outputs["x_rec"], 
                y, 
                outputs["x_latent"], 
                outputs["x_latent_logvar"]
            )
        else:  # VQ-VAE
            loss = self.loss(
                outputs["x_rec"], 
                y, 
                outputs["map_loss_breakdown"]
            )
        
        metrics = {}
        if validation_mode:
            metrics = self.calculate_validation_metrics(outputs, y)
            
        return loss, metrics, outputs

    def calculate_validation_metrics(
        self, 
        outputs: Dict[str, torch.Tensor], 
        y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Calculate validation metrics.

        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs
        y : torch.Tensor
            Target tensor

        Returns
        -------
        Dict[str, torch.Tensor]
            Metrics dictionary
        """
        metrics = {}
        
        # Apply post-processing for metrics
        y_postprocessed = self.model.post_processors(y, in_place=False)
        outputs_postprocessed = self.model.post_processors(outputs["x_rec"], in_place=False)
        
        # Calculate reconstruction metrics
        for metric in self.metrics:
            metric_name = getattr(metric, "name", metric.__class__.__name__.lower())
            metrics[f"{metric_name}"] = metric(outputs_postprocessed, y_postprocessed)
        
        # Add latent space metrics
        if self.model_type == "continuous":  # Beta-VAE
            metrics["kl_divergence"] = self.loss.kl_divergence(outputs["x_latent"], outputs["x_latent_logvar"])
        elif self.model_type == "discrete":  # VQ-VAE
            metrics["commitment_loss"] = outputs["map_loss_breakdown"].get("commitment_loss", torch.tensor(0.0))
            metrics["codebook_loss"] = outputs["map_loss_breakdown"].get("codebook_loss", torch.tensor(0.0))
            
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
            f"train_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        return train_loss

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
            Validation loss and outputs
        """
        with torch.no_grad():
            val_loss, metrics, outputs = self._step(batch, batch_idx, validation_mode=True)

        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )

        for mname, mvalue in metrics.items():
            self.log(
                f"val_{mname}",
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch.shape[0],
                sync_dist=True,
            )

        return val_loss, outputs

    def configure_optimizers(self) -> tuple:
        """Configure optimizers and learning rate schedulers.
        
        Returns
        -------
        tuple
            Optimizers and schedulers
        """
        if self.discriminator is None:
            return super().configure_optimizers()
        
        if self.config.training.scale_lr_by_gpus:
            lr_vae = (
                (self.config.hardware.num_nodes
                    * self.config.hardware.num_gpus_per_node
                    * self.config.training.optimizer.lr)
                    / self.config.hardware.num_gpus_per_model
                )
        else:
            lr_vae = self.config.training.optimizer.lr

        lr_discriminator = self.config.training.optimizer_discriminator.lr_ratio_to_vae * lr_vae
        
        LOGGER.debug("Effective VAE learning rate: %.3e", lr_vae)

        # VAE Encoder/Decoder optimizer
        vae_named_params = self.model.named_parameters()
        if self.config.training.optimizer.weight_decay > 0:
            vae_param_groups = self.split_params_by_weight_decay(vae_named_params, self.config.training.optimizer.weight_decay, lr_vae)
        else:
            vae_param_groups = vae_named_params

        optimizer_vae = torch.optim.AdamW(
            vae_param_groups,
            betas=(0.9, 0.95),
            lr=lr_vae,
            weight_decay=self.config.training.optimizer.weight_decay,
        )
        scheduler_vae = CosineLRScheduler(
            optimizer_vae,
            lr_min=self.config.training.scheduler.lr_min,
            t_initial=self.config.training.scheduler.t_initial or self.training_steps(),
            warmup_t= self.config.training.scheduler.warmup_t if self.config.training.scheduler.warmup_t.is_integer() else int(self.config.training.scheduler.warmup_t * self.training_steps()),
        )

        # Discriminator optimizer
        discriminator_named_params = self.discriminator.named_parameters()
        if self.config.training.optimizer_discriminator.weight_decay > 0:
            discriminator_param_groups = self.split_params_by_weight_decay(discriminator_named_params, self.config.training.optimizer_discriminator.weight_decay, lr_discriminator)
        else:
            discriminator_param_groups = discriminator_named_params

        optimizer_discriminator = torch.optim.AdamW(
            discriminator_param_groups,
            betas=(0.9, 0.95),
            lr=lr_discriminator,
            weight_decay=self.config.training.optimizer_discriminator.weight_decay,
        )
        
        scheduler_discriminator = CosineLRScheduler(
            optimizer_discriminator,
            lr_min=self.config.training.scheduler.lr_min,
            t_initial=self.config.training.scheduler.t_initial or (self.training_steps() - self.discriminator_training_start_step),
            warmup_t= self.config.training.scheduler.warmup_t if self.config.training.scheduler.warmup_t.is_integer() else int(self.config.training.scheduler.warmup_t * (self.training_steps() - self.discriminator_training_start_step) ),
        )
        
        return [optimizer_vae, optimizer_discriminator], [{"scheduler": scheduler_vae, "interval": "step"}, {"scheduler": scheduler_discriminator, "interval": "step"}]

    def split_params_by_weight_decay(self, named_params, weight_decay=0.01, lr=0.001) -> list[dict]:
        """Split parameters by weight decay.
        
        Parameters
        ----------
        named_params : iterator
            Named parameters
        weight_decay : float, optional
            Weight decay value, by default 0.01
        lr : float, optional
            Learning rate, by default 0.001
            
        Returns
        -------
        list[dict]
            Parameter groups
        """
        decay_params = []
        no_decay_params = []
        for name, param in named_params:
            if not param.requires_grad:
                continue  # skip frozen parameters
            # Do not apply weight decay to bias terms or LayerNorm parameters.
            if "bias" in name or "norm" in name or "ln" in name or 'qk_scale' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": lr}
        ]

    def training_steps(self) -> int:
        """Get the total number of training steps.
        
        Returns
        -------
        int
            Total number of training steps
        """
        return self.trainer.max_steps
    

class ReconstructionLightningModule_fromOriginalBranch(AnemoiLightningModule):
    def __init__(self, config, graph_data, statistics, statistics_tendencies, data_indices, metadata):
        super().__init__(config, graph_data, statistics, statistics_tendencies, data_indices, metadata, model_cls=ReconstructionModelInterface)

        self.latlons_hidden = self.get_latlons_hidden(graph_data, config)
        self.latlons_input = self.get_latlons_input(graph_data, config)
        self.latlons_output = self.get_latlons_output(graph_data, config)
        self.latent_node_weights = self.get_latent_node_weights(graph_data, config)
        self.node_weights = self.get_node_weights(graph_data, config)

        self.modelling_strategy = 0

        if config.training.masked_weather_modelling:
            self.modelling_strategy = 1
            
            # feature_mapping = { v:self.data_indices.model.output.name_to_index[k] for k,v in self.data_indices.model.input.name_to_index.items() if k in self.data_indices.model.output.name_to_index }

            feature_mapping = { k:{'model_output_idx':v_model_ouput_idx, 'model_input_idx': self.data_indices.model.input.name_to_index[k]} for k, v_model_ouput_idx in self.data_indices.model.output.name_to_index.items() }
            
            self.masked_weather_modeller = instantiate_debug(
                                                                config.training.masked_weather_modelling,
                                                                  feature_in_size=len(self.data_indices.data.input.full),
                                                                  feature_out_size=len(self.data_indices.data.output.full),
                                                                  feature_mapping=feature_mapping,
                                                                  graph = graph_data,
                                                                  target_grid_name=config.graph.decoder[-1],
                                                                  latent_grid_name=config.graph.encoder[-1],
                                                                  model_output_name_to_index=self.data_indices.model.output.name_to_index,
                                                                  
                                                                  )
            assert config.data.normalizers.state.default == "min-max-floor", "Masked weather modelling requires min-max-floor normalization - since we set masked values to 0 e.g. no input"

        # Instantiate the discriminator
        if hasattr(config, "model_discriminator"):
            self.discriminator = instantiate_debug(
                config.model_discriminator.model,
                config,
                data_indices=self.data_indices,
                graph_data=graph_data    
            )

            self.discriminator_loss = instantiate_debug(
                config.training.discriminator_loss,
                node_weights=self.node_weights,
                feature_weights=torch.ones(self.discriminator.num_output_channels)
            )

            self.discriminator_loss_weight = self.config.training.discriminator_loss_weight

            self.automatic_optimization = False
            self.modelling_strategy = 2
            self.accum_grad_batches = config.training.accum_grad_batches

            # NOTE : when doing automatic_optimization=False, the trainer.accum_grad_batches will be set to 1 for PytorchLightning compatibility
            # We set self.accum_grad_batches to the value specified in the config

            # Setting up parameters for delayed discriminator training
            self.discriminator_training_start_step = config.training.get("discriminator_training_start_step", 0)
            self.discriminator_training_start_step = self.discriminator_training_start_step if (self.discriminator_training_start_step % 1 == 0) else int(self.discriminator_training_start_step * self.training_steps())
            self.discriminator_backprop_to_vae_dec_start_step = config.training.get("discriminator_backprop_to_vae_dec_start_step", 0)
            self.discriminator_backprop_to_vae_dec_start_step = self.discriminator_backprop_to_vae_dec_start_step if (self.discriminator_backprop_to_vae_dec_start_step % 1 == 0) else int(self.discriminator_backprop_to_vae_dec_start_step * self.training_steps())
            assert self.discriminator_training_start_step < self.discriminator_backprop_to_vae_dec_start_step, "Discriminator training start step must be before the discriminator backprop to vae start step"
            LOGGER.info(f"Discriminator training start step: {self.discriminator_training_start_step}, Discriminator backprop to vae start step: {self.discriminator_backprop_to_vae_dec_start_step}")
            LOGGER.info(f"Note: This is the number of gradient update steps not the number of training batches seen - it aligns with max_steps")


        else:
            self.discriminator = None
            self.discriminator_loss = None
            

        self.multi_step = config.training.multistep_input
        assert self.multi_step == 1, "Logic for multistep not implemented yet: Callbacks, Training, What would be reconstructing 2 or 1 states>?"

        self.loss = self.instantiate_loss(config, graph_data)
        self.val_metrics = self.instantiate_val_metrics(config, graph_data)

        self.step_functions = {
                0: self._step_normal,
                1: self._step_masked,
                2: self._step_masked_w_discriminator,
            }


    def configure_optimizers(self):
        if self.discriminator is None:
            return super().configure_optimizers()
        
        if self.config.training.scale_lr_by_gpus:
            lr_vae = (
                (self.config.hardware.num_nodes
                    * self.config.hardware.num_gpus_per_node
                    * self.config.training.optimizer.lr)
                    / self.config.hardware.num_gpus_per_model
                )
        else:
            lr_vae = self.config.training.optimizer.lr

        lr_discriminator = self.config.training.optimizer_discriminator.lr_ratio_to_vae * lr_vae
        
        LOGGER.debug("Effective VAE learning rate: %.3e", lr_vae)

        # VAE Encoder/Decoder  optimizer
        vae_named_params = self.model.named_parameters()
        if self.config.training.optimizer.weight_decay > 0:
            vae_param_groups = self.split_params_by_weight_decay(vae_named_params, self.config.training.optimizer.weight_decay, lr_vae)
        else:
            vae_param_groups = vae_named_params

        optimizer_vae = torch.optim.AdamW(
            vae_param_groups,
            betas=(0.9, 0.95),
            lr=lr_vae,
            weight_decay=self.config.training.optimizer.weight_decay,
        )
        scheduler_vae = CosineLRScheduler(
            optimizer_vae,
            lr_min=self.config.training.scheduler.lr_min,
            t_initial=self.config.training.scheduler.t_initial or self.training_steps(),
            warmup_t= self.config.training.scheduler.warmup_t if self.config.training.scheduler.warmup_t.is_integer() else int(self.config.training.scheduler.warmup_t * self.training_steps()),
        )

        # Discriminator optimizer
        discriminator_named_params = self.discriminator.named_parameters()
        if self.config.training.optimizer_discriminator.weight_decay > 0:
            discriminator_param_groups = self.split_params_by_weight_decay(discriminator_named_params, self.config.training.optimizer_discriminator.weight_decay, lr_discriminator)
        else:
            discriminator_param_groups = discriminator_named_params

        optimizer_discriminator = torch.optim.AdamW(
            discriminator_param_groups,
            betas=(0.9, 0.95),
            lr=lr_discriminator,
            weight_decay=self.config.training.optimizer_discriminator.weight_decay,
        )
        # Need to adjust the scheduler to account for if the discriminator is trained from the start or not
        
        scheduler_discriminator = CosineLRScheduler(
            optimizer_discriminator,
            lr_min=self.config.training.scheduler.lr_min,
            t_initial=self.config.training.scheduler.t_initial or (self.training_steps() - self.discriminator_training_start_step),
            warmup_t= self.config.training.scheduler.warmup_t if self.config.training.scheduler.warmup_t.is_integer() else int(self.config.training.scheduler.warmup_t * (self.training_steps() - self.discriminator_training_start_step) ),
        )
        
        return [optimizer_vae, optimizer_discriminator], [{"scheduler": scheduler_vae, "interval": "step"}, {"scheduler": scheduler_discriminator, "interval": "step"}]

    def split_params_by_weight_decay(self, named_params, weight_decay=0.01, lr=0.001) -> list[dict]:

        decay_params = []
        no_decay_params = []
        for name, param in named_params:
            if not param.requires_grad:
                continue  # skip frozen parameters
            # Do not apply weight decay to bias terms or LayerNorm parameters.
            if "bias" in name or "norm" in name or "ln" in name or 'qk_scale' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": lr}
        ]

    def get_latlons_input(self, graph_data, config):
        return graph_data[config.graph.encoder[0]].x

    def get_latlons_output(self, graph_data, config):
        return graph_data[config.graph.decoder[-1]].x
    
    def get_latlons_hidden(self, graph_data, config):
        return [graph_data[h_name].x for h_name in config.graph.encoder]
    
    def get_latent_node_weights(self, graph_data, config):
        return graph_data[config.graph.encoder[-1]][config.model.node_loss_weight].squeeze()

    def get_node_weights(self, graph_data, config):
        output_grid = config.graph.decoder[-1]
        return graph_data[output_grid][config.model.node_loss_weight].squeeze()

    def _step( 
        self,
        batch: Tensor, # shape (bs, multistep, latlon, nvars_full)
        batch_idx: int = 0,
        validation_mode: bool = False,
        batch_target: Optional[torch.Tensor] = None,
        validation_kwargs: dict|None = None,
    ) -> tuple[Tensor, Mapping, Tensor]:
        loss, metrics, dict_tensors = self.step_functions[self.modelling_strategy](batch, batch_idx, validation_mode, batch_target, validation_kwargs)
        
        return loss, metrics, dict_tensors

    def _step_normal(
        self,
        batch: Tensor, # shape (bs, multistep, latlon, nvars_full)
        batch_idx: int = 0,
        validation_mode: bool = False,
        batch_target: Optional[torch.Tensor|dict[str, torch.Tensor]] = None,
        validation_kwargs: dict|None = None,
        ) -> tuple[Tensor, Mapping, Tensor]:
        raise NotImplementedError("Not implemented")
        batch_inp = self.model.pre_processors_state(batch, in_place=False)  # shape = (bs, nens_input, multistep, 

        x_inp = batch_inp[..., self.data_indices.data.input.full]

        if batch_target is not None:
            batch_target = self.model.post_processors_state(batch_target, in_place=False, inverse=True)
            x_target = batch_target[
                ..., self.data_indices.data.output.full,
            ]
        else:

            batch_target = self.model.post_processors_state(batch, in_place=False, inverse=True)
            x_target = batch_target[
                ..., self.data_indices.data.output.full,
            ]

        metrics = {}
        
        dict_model_outputs = self(x_inp) #  z_mu, li_z_mu, z_logvar
        x_rec = dict_model_outputs.pop("x_rec")

        loss: dict = checkpoint(self.loss, x_rec, x_target, **dict_model_outputs, squash=True, use_reentrant=False)

        if validation_mode:
            dict_tensors = self.get_proc_and_unproc_data(x_rec, x_target, x_inp)
            metrics_new = self.calculate_val_metrics(**dict_tensors, x_quantized=dict_model_outputs.get("x_quantized"), x_latent=dict_model_outputs.get("x_latent"), validation_kwargs=validation_kwargs)

            metrics.update(metrics_new)
        else:
            dict_tensors = {}

        # Reducing the time dimension out (leave it in)
        # TODO (rilwan-ade) make sure the z_logvar based evals handle the fact that there is a TIME dimension
        # TODO (rilwan-ade) make sure losses factor in this time and can report per time step in the sequence
        return loss, metrics, {**dict_tensors, **dict_model_outputs}

    def _step_masked(self, batch: Tensor, batch_idx: int, validation_mode: bool = False, batch_target: Optional[torch.Tensor] = None, validation_kwargs: dict|None = None):
        
        batch_inp = self.model.pre_processors_state(batch, in_place=False)  # shape = (bs, nens_input, multistep, 

        x_inp = batch_inp[..., self.data_indices.data.input.full]

        if batch_target is not None:
            batch_target = self.model.post_processors_state(batch_target, in_place=False, inverse=True)
            x_target = batch_target[
                ..., self.data_indices.data.output.full,
            ]
        else:

            batch_target = self.model.post_processors_state(batch, in_place=False, inverse=True)
            x_target = batch_target[
                ..., self.data_indices.data.output.full,
            ]

        # The mask represents 1 where we want to calculate loss and 0 where we don't
        # which also means 1 where the input is masked and 0 where the input is not masked
        input_mask, target_mask = self.masked_weather_modeller.generate_masks( x_target, x_inp )
        x_inp_masked = torch.where(input_mask == 1, torch.zeros_like(x_inp), x_inp  ) # NOTE: not using perceiver so need to make sure the 0 value is outside the valid range of the data and output of the model is bounded w/ Relu at a value m matching the standardization
        dict_model_outputs = self(x_inp_masked) #  z_mu, li_z_mu, z_logvar
        x_rec = dict_model_outputs.pop("x_rec")
        # NOTE: we only want to calculate loss on unmasked data so at the points where we don't want to calculate loss we set the target to the input
        x_rec_for_loss = torch.where(target_mask == 1, x_rec, x_target)


        loss: dict = checkpoint(self.loss, x_rec_for_loss, x_target, **dict_model_outputs, squash=True, use_reentrant=False, mwm_mask=target_mask)

        if validation_mode:
            # NOTE: validation metrics are calculated based on the full reconstruction
            dict_tensors = self.get_proc_and_unproc_data(x_rec, x_target, x_inp)
            
            metrics = self.calculate_val_metrics(**dict_tensors, x_quantized=dict_model_outputs.get("x_quantized"), x_latent=dict_model_outputs.get("x_latent"), validation_kwargs=validation_kwargs)

        else:
            dict_tensors = {}
            metrics = {}

        return loss, metrics, {**dict_tensors, **dict_model_outputs}


    def _step_masked_w_discriminator(self, batch: Tensor, batch_idx: int, validation_mode: bool = False, batch_target: Optional[torch.Tensor] = None, validation_kwargs: dict|None = None):
        # Optimization strategy:
        # From VAEGAN paper:
        # 1. VAE Encoder: 1) reconstruction loss 2) kl divergence  / feature matching loss
        # 2. VAE Decoder: 1) reconstruction loss / feature matching loss 2) discriminator loss
        # 3. VAE shared parameters: 1) kl divergence 2) reconstruction loss / feature matching loss 3) discriminator loss
        # 4. Discriminator: 1) adversarial loss

        # From VAE-GAN paper: The discriminator loss should only be applied to the VAE decoder, not the VAE encoder or discriminator

        # vae_opt_encoder, vae_opt_decoder, vae_opt_shared_params, discriminator_opt = self.optimizers()
        vae_opt, discriminator_opt = self.optimizers()
        losses = {}

        batch_inp = self.model.pre_processors_state(batch, in_place=False)  # shape = (bs, nens_input, multistep, 

        x_inp = batch_inp[..., self.data_indices.data.input.full]

        if batch_target is not None:
            batch_target = self.model.post_processors_state(batch_target, in_place=False, inverse=True)
            x_target = batch_target[
                ..., self.data_indices.data.output.full,
            ]
        else:

            batch_target = self.model.post_processors_state(batch, in_place=False, inverse=True)
            x_target = batch_target[
                ..., self.data_indices.data.output.full,
            ]

        # The mask represents 1 where we want to calculate loss and 0 where we don't, which also means 1 where the input is masked and 0 where the input is not masked
        input_mask, target_mask = self.masked_weather_modeller.generate_masks( x_target, x_inp )
        x_inp_masked = torch.where(input_mask == 1, torch.zeros_like(x_inp), x_inp  ) # NOTE: not using perceiver so need to make sure the 0 value is outside the valid range of the data and output of the model is bounded w/ Relu at a value m matching the standardization
        dict_model_outputs = self(x_inp_masked) #  z_mu, li_z_mu, z_logvar
        x_rec = dict_model_outputs.pop("x_rec")
        # NOTE: we only want to calculate loss on unmasked data so at the points where we don't want to calculate loss we set the target to the input
        x_rec_for_loss = torch.where(target_mask == 1, x_rec, x_target)

        # Scale losses by accumulation factor
        accumulation_factor = 1.0 / self.accum_grad_batches
        grad_update_step = (batch_idx + 1) % self.accum_grad_batches == 0
        first_grad_accum_step = (batch_idx % self.accum_grad_batches) == 0

        # Reconstruction loss
        map_vae_loss: dict = checkpoint(self.loss, x_rec_for_loss, x_target, **dict_model_outputs, squash=True, use_reentrant=False, mwm_mask=target_mask)
        reconstruction_loss = map_vae_loss[self.loss.name_reconstruction]
        latent_regularisation_loss = map_vae_loss[self.loss.name_latent] 

        # Check if we should start training the discriminator
        should_train_discriminator = self.trainer.global_step >= self.discriminator_training_start_step
        should_backprop_to_vae_dec = self.trainer.global_step >= self.discriminator_backprop_to_vae_dec_start_step
        
        # Discriminator loss calculation
        if should_train_discriminator:
            discriminator_loss, dict_discriminator_outputs = self._step_discriminator(x_inp, x_target, should_backprop_to_vae_dec)
            dict_model_outputs.update(dict_discriminator_outputs)
        else:
            # Zero loss and empty dict if not training discriminator yet
            discriminator_loss = torch.tensor(0.0, device=self.device)
            dict_discriminator_outputs = {}
        

        # Zero gradients only at start of accumulation cycle
        if first_grad_accum_step:
            vae_opt.zero_grad(set_to_none=True)
            if should_train_discriminator:
                discriminator_opt.zero_grad(set_to_none=True)

        # Alternative way is to do 1 manual backward - but then the discriminator should only receives a weighted gradient of discriminator_loss, but all other modules need to only recieve the weighted discriminator_loss
        
        if self.model.model.latent_representation_format == "continuous":
            total_loss = (
                reconstruction_loss + 
                latent_regularisation_loss * self.loss.divergence_loss_weight + 
                discriminator_loss * self.discriminator_loss_weight
            )

        elif self.model.model.latent_representation_format == "discrete":

            # NOTE: Currently this only supports commitment weight as regularisation loss - would need optionality of other regularisation losses being used
            total_loss = (
                reconstruction_loss +
                latent_regularisation_loss * self.loss.commitment_weight + #(the latent regularisation loss is already scaled by the vq_lucid rains class)
                discriminator_loss * self.discriminator_loss_weight
            )

            # vae_loss = (
            #     reconstruction_loss + 
            #     latent_regularisation_loss * self.loss.latent_regularisation_loss_weight + 
            #     discriminator_loss * self.discriminator_loss_weight
            # )
            
        

        if not validation_mode:
            
            # Use no_sync context manager to prevent gradient synchronization during non-update steps
            # context_manager = self.trainer.strategy.no_sync if hasattr(self.trainer.strategy, "no_sync") and not grad_update_step else nullcontext
            from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior
            context_manager = _block_parallel_sync_behavior(self.trainer.strategy, block=True) if not grad_update_step else nullcontext()

            with context_manager:
                self.manual_backward(total_loss * accumulation_factor)

            if grad_update_step:
                vae_scheduler, discriminator_scheduler = self.lr_schedulers()

                # VAE Encoder Gradient Updates
                self.clip_gradients(vae_opt, gradient_clip_val=self.config.training.gradient_clip.val, gradient_clip_algorithm=self.config.training.gradient_clip.algorithm)
                vae_opt.step()
                vae_scheduler.step(self.trainer.global_step)

                if should_train_discriminator:
                    self.clip_gradients(discriminator_opt, gradient_clip_val=self.config.training.gradient_clip_discriminator.val, gradient_clip_algorithm=self.config.training.gradient_clip_discriminator.algorithm)
                    # Hacky way to prevent the global step increasing after the first optimizer steps - o/w issues mentioned here occur: https://github.com/Lightning-AI/pytorch-lightning/blob/bdac61a11894264fcf2b40d174be99cb0c32e006/src/lightning/pytorch/loops/training_epoch_loop.py#L100
                    self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed -= 1
                    discriminator_opt.step()
                    discriminator_scheduler.step(self.trainer.global_step - self.discriminator_training_start_step)
                
        if validation_mode:
            # NOTE: validation metrics are calculated based on the full reconstruction
            dict_tensors = self.get_proc_and_unproc_data(x_rec, x_target, x_inp)
            metrics = self.calculate_val_metrics(**dict_tensors, x_quantized=dict_model_outputs.get("x_quantized"), x_latent=dict_model_outputs.get("x_latent"), validation_kwargs=validation_kwargs)
        else:
            dict_tensors = {}
            metrics = {}

        losses.update(map_vae_loss)
        losses.update({f"discriminator_loss_{self.discriminator_loss.name}": discriminator_loss})

        return losses, metrics, {**dict_tensors, **dict_model_outputs}
    

    def _step_discriminator(self, x_inp_unmasked: Tensor, x_target_unmasked: Tensor, should_backprop_to_vae_dec: bool = True):
        #NOTE: the vae.encoder should not be udpdated here, only the vae.decoder and the discriminator so the detach() should be after the vae.encoder
        context = torch.no_grad if not should_backprop_to_vae_dec else nullcontext        
        with context():
            dict_model_outputs = self.model.model.forward(x_inp_unmasked, model_comm_group=None, sample_from_latent_space=False, detach_encoder_outp=True)

        x_unmasked_rec = dict_model_outputs.pop("x_rec")


        _, t, ens, grid, _ = x_unmasked_rec.shape
        
        # inp = torch.cat([x_unmasked_rec, x_target_unmasked], dim=0)
        # dict_discriminator_outputs = self.discriminator(inp)
        # x_classified = dict_discriminator_outputs["x_classified"]
        # x_classified = einops.rearrange(x_classified, "(bs t ens grid) outp_dim -> bs t ens grid outp_dim", t=t, ens=ens, grid=grid)
        # x_fake_classified  = x_classified[0:bs_fake, ...] # shape (bs, ts, ens, nvars, latlon)
        # x_real_classified = x_classified[bs_fake:, ...] 


        dict_discriminator_outputs_fake = self.discriminator(x_unmasked_rec)
        dict_discriminator_outputs_real = self.discriminator(x_target_unmasked)
        x_classified_fake = dict_discriminator_outputs_fake["x_classified"]
        x_classified_real = dict_discriminator_outputs_real["x_classified"]
        x_classified_fake = einops.rearrange(x_classified_fake, "(bs t ens grid) outp_dim -> bs t ens grid outp_dim", t=t, ens=ens, grid=grid)
        x_classified_real = einops.rearrange(x_classified_real, "(bs t ens grid) outp_dim -> bs t ens grid outp_dim", t=t, ens=ens, grid=grid)
 

        discriminator_loss = self.discriminator_loss(x_classified_fake, x_classified_real, squash=True)

        return discriminator_loss, {}
        
        

    def get_proc_and_unproc_data(self, x_rec: Tensor, x_target: Tensor, x_inp: Tensor):

        x_rec_postprocessed = self.model.post_processors_state(x_rec, in_place=False, data_index=self.data_indices.data.output.full)
        x_target_postprocessed = self.model.post_processors_state(x_target, in_place=False, data_index=self.data_indices.data.output.full)
        x_inp_postprocessed = self.model.pre_processors_state(x_inp, in_place=False, data_index=self.data_indices.data.input.full, inverse=True)

        return {
            "x_rec": x_rec,
            "x_target": x_target,
            "x_inp": x_inp,
            "x_rec_postprocessed": x_rec_postprocessed,
            "x_target_postprocessed": x_target_postprocessed,
            "x_inp_postprocessed": x_inp_postprocessed,
            
        }

    def calculate_val_metrics(self, x_rec, x_target, x_inp, x_rec_postprocessed, x_target_postprocessed, time_index:list[int]|None=None, val_metrics:ModuleList|None=None, val_metric_ranges:dict|None=None, **kwargs):
        metric_vals = {}
        timesteps = x_rec.shape[1]

        if val_metrics is None:
            val_metrics = self.val_metrics

        if val_metric_ranges is None:
            val_metric_ranges = self.val_metric_ranges

        for metric in val_metrics:
            for mkey, indices in val_metric_ranges.items():
                # NOTE: feature_scale is turned off here

                # for single metrics do no variable scaling and non processed data
                # TOOD (rilwan-ade): Update logging to use get_time_step and report lead time in hours
                if len(indices) == 1:
                    _args = (x_rec_postprocessed[..., indices], x_target_postprocessed[..., indices])
                else:
                    _args = (x_rec[..., indices], x_target[..., indices])
                
                m_value = metric(*_args, feature_scale=False, squash=(-5, -3, -2, -1), feature_indices=indices,  **kwargs)

                if m_value.numel() == 0:
                    continue

                # Check if it is a time-dependent metric. 
                metric_reduced_time_dimension = m_value.shape[0] != timesteps
                if not metric_reduced_time_dimension: 

                    for i in range(timesteps):
                        if time_index is None or i in time_index:
                            metric_vals[f"{metric.name}_{mkey}_{i + 1}"] = m_value[i]
                else:
                    metric_vals[f"{metric.name}_{mkey}"] = m_value

        return metric_vals

    def instantiate_loss(self, config, graph_data):

        loss = instantiate_debug(
            config.training.loss,
            node_weights=self.node_weights,
            latent_node_weights=self.get_latent_node_weights(graph_data, config),
            feature_weights=self.feature_weights,
            data_indices_model_output=self.data_indices.model.output,
        )

        return loss
    
    def instantiate_val_metrics(self, config, graph_data):
        
        val_metrics = ModuleList(
            [
                instantiate_debug(
                    vm_cfg,
                    node_weights=self.node_weights,
                    feature_weights=self.feature_weights,
                    latent_node_weights=self.get_latent_node_weights(graph_data, config),
                    data_indices_model_output=self.data_indices.model.output,
                )
                for vm_cfg in config.training.validation_metrics
            ],
        )

        return val_metrics
 