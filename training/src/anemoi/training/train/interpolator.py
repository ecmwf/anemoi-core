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
from typing import Optional
from typing import Union
from operator import itemgetter

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
from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.losses.utils import grad_scaler
from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask
from anemoi.utils.config import DotDict
from anemoi.training.train.forecaster import GraphForecaster

LOGGER = logging.getLogger(__name__)

class GraphInterpolator(GraphForecaster):
    """Graph neural network interpolator for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict
    ) -> None:
        """Initialize graph neural network interpolator.

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
        super().__init__(config = config, graph_data = graph_data, statistics = statistics, data_indices = data_indices, metadata = metadata, supporting_arrays = supporting_arrays)
        self.target_forcing_indices = itemgetter(*config.training.target_forcing.data)(data_indices.data.input.name_to_index)
        if type(self.target_forcing_indices) == int:
            self.target_forcing_indices = [self.target_forcing_indices]
        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index,data_index in enumerate(sorted_indices)}


    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        
        del batch_idx
        batch = self.allgather_batch(batch)

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []

        batch = self.model.pre_processors(batch, in_place = not validation_mode)

        if not self.updated_loss_mask:
            self.training_weights_for_imputed_variables(batch)

        x_bound = batch[:, itemgetter(*self.boundary_times)(self.imap)][..., self.data_indices.data.input.full] # (bs, time, ens, latlon, nvar)

        tfi = self.target_forcing_indices
        target_forcing = torch.empty(batch.shape[0], batch.shape[2], batch.shape[3], len(tfi)+1, device = self.device, dtype = batch.dtype)
        for interp_step in self.interp_times:
            #get the forcing information for the target interpolation time:
            target_forcing[..., :len(tfi)] = batch[:, self.imap[interp_step], :, :, tfi]
            target_forcing[..., -1] = (interp_step - self.boundary_times[1])/(self.boundary_times[1] - self.boundary_times[0])
            #TODO: make fraction time one of a config given set of arbitrary custom forcing functions.

            y_pred = self(x_bound, target_forcing)
            y = batch[:, self.imap[interp_step], :, :, self.data_indices.data.output.full]

            loss += checkpoint(self.loss, y_pred, y, use_reentrant=False)

            metrics_next = {}
            if validation_mode:
                metrics_next = self.calculate_val_metrics(y_pred, y, interp_step-1) #expects rollout but can be repurposed here.
            metrics.update(metrics_next)
            y_preds.extend(y_pred)

        loss *= 1.0 / len(self.interp_times)
        return loss, metrics, y_preds
    
    def forward(self, x: torch.Tensor, target_forcing: torch.Tensor) -> torch.Tensor:
        return self.model(x, target_forcing, self.model_comm_group)