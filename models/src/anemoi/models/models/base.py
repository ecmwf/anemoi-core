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
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.bounding import build_boundings
from anemoi.models.layers.graph import NamedNodesAttributes
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
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        statistics : dict
            Data statistics
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()
        self._graph_data = graph_data
        self.data_indices = data_indices
        self.statistics = statistics

        model_config = DotDict(model_config)
        self._graph_name_data = (
            model_config.graph.data
        )  # assumed to be all the same because this is how we construct the graphs
        self._graph_name_hidden = (
            model_config.graph.hidden
        )  # assumed to be all the same because this is how we construct the graphs
        self.n_step_input = model_config.training.multistep_input
        self.n_step_output = model_config.training.multistep_output
        self.num_channels = model_config.model.num_channels
        self.latent_skip = model_config.model.get("latent_skip", True)

        self.node_attributes = torch.nn.ModuleDict()
        for dataset_name in self._graph_data.keys():
            self.node_attributes[dataset_name] = NamedNodesAttributes(
                model_config.model.trainable_parameters.hidden, self._graph_data[dataset_name]
            )

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)
        self._assert_consistent_hidden_graphs()

        # build networks
        self.has_processor = model_config.model.processor is not None
        self._build_networks(model_config)

        # build residual connection
        self._build_residual(model_config.model.residual)

        # build boundings
        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        # Multi-dataset: create ModuleDict with ModuleList per dataset
        self.boundings = build_boundings(model_config, self.data_indices, self.statistics)

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        # Multi-dataset: create dictionaries for each property
        self.num_input_channels = {}
        self.num_output_channels = {}
        self.num_input_channels_prognostic = {}
        self.num_input_channels_target_forcings = {}
        self._internal_input_idx = {}
        self._internal_output_idx = {}
        self._target_forcing_input_idx = {}
        self.input_dim = {}
        self.input_dim_latent = {}
        self.target_dim = {}
        self.output_dim = {}

        for dataset_name, dataset_indices in data_indices.items():
            self._internal_input_idx[dataset_name] = dataset_indices.model.input.prognostic
            self._internal_output_idx[dataset_name] = dataset_indices.model.output.prognostic
            self._forcing_input_idx[dataset_name] = [] # TODO: Where do we define the target forcings?

            self.num_input_channels[dataset_name] = len(dataset_indices.model.input)
            self.num_input_channels_target_forcings[dataset_name] = len(self._forcing_input_idx[dataset_name])
            self.num_output_channels[dataset_name] = len(dataset_indices.model.output)

            self.input_dim[dataset_name] = self._calculate_input_dim(dataset_name)
            self.input_dim_latent[dataset_name] = self._calculate_input_dim_latent(dataset_name)
            self.target_dim[dataset_name] = self._calculate_target_dim(dataset_name)
            self.output_dim[dataset_name] = self._calculate_output_dim(dataset_name)

    def _calculate_input_dim(self, dataset_name: str) -> int:
        return (
            self.n_step_input * self.num_input_channels[dataset_name]
            + self.node_attributes[dataset_name].attr_ndims[self._graph_name_data]
        )

    def _calculate_input_dim_latent(self, dataset_name: str) -> int:
        return self.node_attributes[dataset_name].attr_ndims[self._graph_name_hidden]
    
    def _calculate_target_dim(self, dataset_name: str) -> int:
        return (
            self.n_step_output * self.num_input_channels_target_forcings[dataset_name]
            + self.node_attributes[dataset_name].attr_ndims[self._graph_name_data]
        )
    def _calculate_output_dim(self, dataset_name: str) -> int:
        return self.n_step_output * self.num_output_channels[dataset_name]

    def _assert_matching_indices(self, data_indices: dict) -> None:
        # Multi-dataset: check assertions for each dataset
        for dataset_name, dataset_indices in data_indices.items():
            dataset_internal_output_idx = self._internal_output_idx[dataset_name]
            dataset_internal_input_idx = self._internal_input_idx[dataset_name]

            assert len(dataset_internal_output_idx) == len(dataset_indices.model.output.full) - len(
                dataset_indices.model.output.diagnostic
            ), (
                f"Dataset '{dataset_name}': Mismatch between the internal data indices ({len(dataset_internal_output_idx)}) and "
                f"the output indices excluding diagnostic variables "
                f"({len(dataset_indices.model.output.full) - len(dataset_indices.model.output.diagnostic)})",
            )
            assert len(dataset_internal_input_idx) == len(
                dataset_internal_output_idx,
            ), f"Dataset '{dataset_name}': Model indices must match {dataset_internal_input_idx} != {dataset_internal_output_idx}"

    def _assert_consistent_hidden_graphs(self) -> None:
        """Assert that all datasets have identical hidden-to-hidden graph structures.

        This is required because the processor is shared between datasets and operates
        on the hidden state, so all datasets must have the same hidden graph topology.
        """
        if isinstance(self._graph_data, dict) and len(self._graph_data) > 1:
            dataset_names = list(self._graph_data.keys())
            reference_dataset = dataset_names[0]
            reference_graph = self._graph_data[reference_dataset]
            reference_hidden_graph = reference_graph[(self._graph_name_hidden, "to", self._graph_name_hidden)]

            # Check hidden graph structure consistency across all datasets
            for dataset_name in dataset_names[1:]:
                dataset_graph = self._graph_data[dataset_name]
                dataset_hidden_graph = dataset_graph[(self._graph_name_hidden, "to", self._graph_name_hidden)]

                # Compare edge indices
                assert torch.equal(reference_hidden_graph.edge_index, dataset_hidden_graph.edge_index), (
                    f"Hidden-to-hidden graph edge structure mismatch between reference dataset '{reference_dataset}' "
                    f"and dataset '{dataset_name}'. All datasets must have identical hidden graph topology "
                    f"for the shared processor to work correctly."
                )

                # Compare number of nodes (should be same for hidden graphs)
                ref_num_hidden_nodes = self.node_attributes[reference_dataset].num_nodes[self._graph_name_hidden]
                dataset_num_hidden_nodes = self.node_attributes[dataset_name].num_nodes[self._graph_name_hidden]
                assert ref_num_hidden_nodes == dataset_num_hidden_nodes, (
                    f"Hidden node count mismatch between reference dataset '{reference_dataset}' ({ref_num_hidden_nodes} nodes) "
                    f"and dataset '{dataset_name}' ({dataset_num_hidden_nodes} nodes). "
                    f"All datasets must have the same number of hidden nodes for the shared processor."
                )

            LOGGER.info(
                "All datasets have consistent hidden-to-hidden graph structures (required for shared processor)"
            )

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

    def _get_consistent_dim(self, x: dict[str, Tensor], dim: int) -> int:
        dim_sizes = [_x.shape[dim] for _x in x.values()]
        # Assert all datasets have the same sizes
        assert all(bs == dim_sizes[0] for bs in dim_sizes), f"Dimensions must be the same across datasets: {dim_sizes}"

        return dim_sizes[0]

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

    def _build_residual(self, residual_config: DotDict) -> None:
        self.residual = torch.nn.ModuleDict()
        for dataset_name in self._graph_data.keys():
            self.residual[dataset_name] = instantiate(residual_config, graph=self._graph_data[dataset_name])

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
        batch: dict[str, torch.Tensor],
        pre_processors: nn.ModuleDict,
        post_processors: nn.ModuleDict,
        n_step_input: int,
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
        n_step_input : int,
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
            dataset_names = list(batch.keys())

            for dataset_name in dataset_names:
                assert (
                    len(batch[dataset_name].shape) == 4
                ), f"The {dataset_name} input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch[dataset_name].shape}!"
                # Dimensions are: batch, timesteps, grid, variables

            x = {}
            for dataset_name in dataset_names:
                x[dataset_name] = batch[dataset_name][
                    :, 0:n_step_input, None, ...
                ]  # add dummy ensemble dimension as 3rd index

            # Handle distributed processing
            grid_shard_shapes = None
            if model_comm_group is not None:
                grid_shard_shapes = {}
                for dataset_name in dataset_names:
                    shard_shapes = get_shard_shapes(x[dataset_name], -2, model_comm_group=model_comm_group)
                    grid_shard_shapes[dataset_name] = [shape[-2] for shape in shard_shapes]
                    x[dataset_name] = shard_tensor(x[dataset_name], -2, shard_shapes, model_comm_group)

            for dataset_name in dataset_names:
                x[dataset_name] = pre_processors[dataset_name](x[dataset_name], in_place=False)

            # Perform forward pass
            y_hat = self.forward(x, model_comm_group=model_comm_group, grid_shard_shapes=grid_shard_shapes, **kwargs)

            # Apply post-processing
            for dataset_name in dataset_names:
                y_hat[dataset_name] = post_processors[dataset_name](y_hat[dataset_name], in_place=False)

            # Gather output if needed
            if gather_out and model_comm_group is not None:
                for dataset_name in dataset_names:
                    y_hat_shard_shapes = apply_shard_shapes(y_hat[dataset_name], -2, grid_shard_shapes[dataset_name])
                    y_hat[dataset_name] = gather_tensor(y_hat[dataset_name], -2, y_hat_shard_shapes, model_comm_group)

        return y_hat

    @abstractmethod
    def fill_metadata(self, md_dict) -> None:
        """To be implemented in subclasses to fill model-specific metadata."""
        pass
