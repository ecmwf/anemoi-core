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
from typing import Optional, Union, Any, Mapping

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.utils import grad_scaler
from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiLightningModule(pl.LightningModule):
    """Base Lightning Module for Anemoi models."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict = None,
        model_cls=None,
    ) -> None:
        """Initialize base Anemoi Lightning Module.

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
        supporting_arrays : dict, optional
            Supporting NumPy arrays to store in the checkpoint
        model_cls : class, optional
            Model class to instantiate
        """
        super().__init__()
        
        self.config = config
        self.data_indices = data_indices
        self.graph_data = graph_data
        self.statistics = statistics
        self.metadata = metadata
        self.supporting_arrays = supporting_arrays or {}
        
        self.save_hyperparameters()
        
        # Common configuration
        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled
        
        # Learning rate configuration
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.warmup_t = getattr(config.training.lr, "warmup_t", 1000)
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        
        # Distributed training configuration
        self.use_zero_optimizer = config.training.zero_optimizer
        self.model_comm_group = None
        self.reader_groups = None
        
        # Lazy init model and reader group info, will be set by the DDPGroupStrategy
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        
        self.reader_group_id = 0
        self.reader_group_rank = 0
        self.reader_group_size = 1

    @staticmethod
    def get_loss_function(
        config: DictConfig,
        scalars: Union[dict[str, tuple[Union[int, tuple[int, ...], torch.Tensor]]], None] = None,
        **kwargs,
    ) -> Union[BaseWeightedLoss, torch.nn.ModuleList]:
        """Get loss functions from config.

        Can be ModuleList if multiple losses are specified.

        Parameters
        ----------
        config : DictConfig
            Loss function configuration, should include `scalars` if scalars are to be added to the loss function.
        scalars : Union[dict[str, tuple[Union[int, tuple[int, ...], torch.Tensor]]], None], optional
            Scalars which can be added to the loss function. Defaults to None.
            If a scalar is to be added to the loss, ensure it is in `scalars` in the loss config
            E.g.
                If `scalars: ['variable']` is set in the config, and `variable` in `scalars`
                `variable` will be added to the scalar of the loss function.
        kwargs : Any
            Additional arguments to pass to the loss function

        Returns
        -------
        Union[BaseWeightedLoss, torch.nn.ModuleList]
            Loss function, or list of metrics

        Raises
        ------
        TypeError
            If not a subclass of `BaseWeightedLoss`
        ValueError
            If scalar is not found in valid scalars
        """
        config_container = OmegaConf.to_container(config, resolve=False)
        if isinstance(config_container, list):
            return torch.nn.ModuleList(
                [
                    AnemoiLightningModule.get_loss_function(
                        OmegaConf.create(loss_config),
                        scalars=scalars,
                        **kwargs,
                    )
                    for loss_config in config
                ],
            )

        loss_config = OmegaConf.to_container(config, resolve=True)
        scalars_to_include = loss_config.pop("scalars", [])

        # Instantiate the loss function with the loss_init_config
        loss_function = instantiate(loss_config, **kwargs)

        if not isinstance(loss_function, BaseWeightedLoss):
            error_msg = f"Loss must be a subclass of 'BaseWeightedLoss', not {type(loss_function)}"
            raise TypeError(error_msg)

        for key in scalars_to_include:
            if key not in scalars or []:
                error_msg = f"Scalar {key!r} not found in valid scalars: {list(scalars.keys())}"
                raise ValueError(error_msg)
            loss_function.add_scalar(*scalars[key], name=key)

        return loss_function

    def set_model_comm_group(
        self,
        model_comm_group: ProcessGroup,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        model_comm_group_size: int,
    ) -> None:
        """Set model communication group.

        Parameters
        ----------
        model_comm_group : ProcessGroup
            Process group for model communication
        model_comm_group_id : int
            ID of the model communication group
        model_comm_group_rank : int
            Rank within the model communication group
        model_comm_num_groups : int
            Number of model communication groups
        model_comm_group_size : int
            Size of the model communication group
        """
        self.model_comm_group = model_comm_group
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_size = model_comm_group_size

    def set_reader_groups(
        self,
        reader_groups: list[ProcessGroup],
        reader_group_id: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        """Set reader groups.

        Parameters
        ----------
        reader_groups : list[ProcessGroup]
            List of reader groups
        reader_group_id : int
            ID of the reader group
        reader_group_rank : int
            Rank within the reader group
        reader_group_size : int
            Size of the reader group
        """
        self.reader_groups = reader_groups
        self.reader_group_id = reader_group_id
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Allgather the batch-shards across the reader group.

        Parameters
        ----------
        batch : torch.Tensor
            Batch-shard of current reader rank

        Returns
        -------
        torch.Tensor
            Allgathered (full) batch
        """
        grid_size = len(self.latlons_data)  # number of points

        if grid_size == batch.shape[-2]:
            return batch  # already have the full grid

        grid_shard_size = grid_size // self.reader_group_size
        last_grid_shard_size = grid_size - (grid_shard_size * (self.reader_group_size - 1))

        # prepare tensor list with correct shapes for all_gather
        shard_shape = list(batch.shape)
        shard_shape[-2] = grid_shard_size
        last_shard_shape = list(batch.shape)
        last_shard_shape[-2] = last_grid_shard_size

        tensor_list = [torch.empty(tuple(shard_shape), device=self.device) for _ in range(self.reader_group_size - 1)]
        tensor_list.append(torch.empty(last_shard_shape, device=self.device))

        torch.distributed.all_gather(
            tensor_list,
            batch,
            group=self.reader_groups[self.reader_group_id],
        )

        return torch.cat(tensor_list, dim=-2)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        tuple[list[torch.optim.Optimizer], list[dict]]
            List of optimizers and list of scheduler configurations
        """
        if self.use_zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(),
                optimizer_class=torch.optim.AdamW,
                betas=(0.9, 0.95),
                lr=self.lr,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(),
                betas=(0.9, 0.95),
                lr=self.lr,
            )

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=self.warmup_t,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: None = None) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : CosineLRScheduler
            Learning rate scheduler object.
        metric : Optional[Any]
            Metric object for e.g. ReduceLRonPlateau. Default is None.
        """
        del metric
        scheduler.step(epoch=self.trainer.global_step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return self.model(x, self.model_comm_group) 