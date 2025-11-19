# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.bounding import build_boundings
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.truncation import BaseTruncation
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class BaseGraphModel(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        truncation_data: dict,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()
        self._graph_data = graph_data
        self.data_indices = data_indices
        self.statistics = statistics
        self._truncation_data = truncation_data

        model_config = DotDict(model_config)
        self._graph_name_data = model_config.graph.data
        self._graph_name_hidden = model_config.graph.hidden
        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        # build networks
        self._build_networks(model_config)

        # build truncation
        self.truncation = BaseTruncation(self._truncation_data)

        # build boundings
        self.boundings = build_boundings(model_config, self.data_indices, self.statistics)

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = len(data_indices.model.input)
        self.num_output_channels = len(data_indices.model.output)
        self.num_input_channels_prognostic = len(data_indices.model.input.prognostic)
        self._internal_input_idx = data_indices.model.input.prognostic
        self._internal_output_idx = data_indices.model.output.prognostic
        self.input_dim = self._calculate_input_dim()
        self.input_dim_latent = self._calculate_input_dim_latent()

    def _assert_matching_indices(self, data_indices: dict) -> None:
        assert len(self._internal_output_idx) == len(data_indices.model.output.full) - len(
            data_indices.model.output.diagnostic
        ), (
            f"Mismatch between the internal data indices ({len(self._internal_output_idx)}) and "
            f"the output indices excluding diagnostic variables "
            f"({len(data_indices.model.output.full) - len(data_indices.model.output.diagnostic)})",
        )
        assert len(self._internal_input_idx) == len(
            self._internal_output_idx,
        ), f"Model indices must match {self._internal_input_idx} != {self._internal_output_idx}"

    def _assert_valid_sharding(
        self,
        batch_size: int,
        ensemble_size: int,
        in_out_sharded: bool,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> None:
        assert not (
            in_out_sharded and model_comm_group is None
        ), "If input is sharded, model_comm_group must be provided."

        if model_comm_group is not None:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded across GPUs"

            assert (
                model_comm_group.size() == 1 or ensemble_size == 1
            ), "Ensemble size per device must be 1 when model is sharded across GPUs"

    def _calculate_input_dim(self):
        return self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

    def _calculate_input_dim_latent(self):
        return self.node_attributes.attr_ndims[self._graph_name_hidden]

    @abstractmethod
    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the networks for the model."""
        pass

    @abstractmethod
    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):
        pass

    @abstractmethod
    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        pass

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : Tensor
            Input data
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None
        grid_shard_shapes : list, optional
            Shard shapes of the grid, by default None

        Returns
        -------
        Tensor
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        pass

    def predict_step(
        self,
        batch: torch.Tensor,
        pre_processors: nn.Module,
        post_processors: nn.Module,
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> Tensor:
        """Prediction step for the model.

        Base implementation applies pre-processing, performs a forward pass, and applies post-processing.
        Subclasses can override this for different behavior (e.g., sampling for diffusion models).

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data (before pre-processing)
        pre_processors : nn.Module,
            Pre-processing module
        post_processors : nn.Module,
            Post-processing module
        multi_step : int,
            Number of input timesteps
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        gather_out : bool
            Whether to gather output tensors across distributed processes
        **kwargs
            Additional arguments

        Returns
        -------
        Tensor
            Model output (after post-processing)
        """
        with torch.no_grad():

            assert (
                len(batch.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"
            # Dimensions are
            # batch, timesteps, grid, variables
            x = batch[:, 0:multi_step, None, ...]  # add dummy ensemble dimension as 3rd index

            # Handle distributed processing
            grid_shard_shapes = None
            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x, -2, model_comm_group=model_comm_group)
                grid_shard_shapes = [shape[-2] for shape in shard_shapes]
                x = shard_tensor(x, -2, shard_shapes, model_comm_group)

            x = pre_processors(x, in_place=False)

            # Perform forward pass
            y_hat = self.forward(x, model_comm_group=model_comm_group, grid_shard_shapes=grid_shard_shapes, **kwargs)

            # Apply post-processing
            y_hat = post_processors(y_hat, in_place=False)

            # Gather output if needed
            if gather_out and model_comm_group is not None:
                y_hat = gather_tensor(y_hat, -2, grid_shard_shapes, model_comm_group)

        return y_hat
