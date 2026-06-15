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
from omegaconf import ListConfig
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.models.data.batch import Batch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.bounding import build_boundings
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.utils.config import COORDS_DIM
from anemoi.models.utils.config import broadcast_config_keys
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


def split_graph_config(
    graph_config: DotDict,
    is_dataset_static: dict[str, bool],
    hidden_nodes_name: str | list[str],
) -> tuple[DotDict, DotDict]:
    """This function creates the static graph structure and returns the dictionary for the dynamic graph configuration.

    Parameters
    ----------
    graph_config : DotDict
        Graph configuration
    is_dataset_static : dict[str, bool]
        Dictionary indicating whether each dataset is static (e.g., static grid) or not.
    hidden_nodes_name : str or list of str
        Name(s) of the hidden nodes in the graph. They are considered to be static.
    """
    if isinstance(hidden_nodes_name, str):
        is_dataset_static[hidden_nodes_name] = True
    elif isinstance(hidden_nodes_name, list):
        for hidden_name in hidden_nodes_name:
            is_dataset_static[hidden_name] = True
    else:
        raise TypeError(f"Hidden nodes name must be a string or a list of strings, got {type(hidden_nodes_name)}")

    static_graph_config, dynamic_graph_config = {"nodes": {}, "edges": []}, {"nodes": {}, "edges": {}}
    for nodes_name, nodes_config in graph_config.nodes.items():
        if is_dataset_static[nodes_name]:
            static_graph_config["nodes"][nodes_name] = nodes_config
        else:
            dynamic_graph_config["nodes"][nodes_name] = nodes_config

    for edge_config in graph_config.edges:
        source_name = edge_config.source_name
        target_name = edge_config.target_name
        if is_dataset_static[source_name] and is_dataset_static[target_name]:
            static_graph_config["edges"].append(edge_config)
            dynamic_graph_config["edges"][(source_name, "to", target_name)] = {}
        else:
            dynamic_graph_config["edges"][(source_name, "to", target_name)] = {
                "edge_builders": edge_config.edge_builders,
                "attributes": edge_config.attributes,
            }

    return DotDict(static_graph_config), DotDict(dynamic_graph_config)


class BaseGraphModel(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        model_graph_config: DotDict,
        data_indices: dict[str, IndexCollection],
        statistics: dict[str, dict],
        is_dataset_static: dict[str, bool],
        n_step_input: int,
        n_step_output: int,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DictConfig
            Model configuration
        data_indices : dict
            Data indices
        statistics : dict
            Data statistics
        model_graph_config : DotDict
            Graph configuration
        """
        super().__init__()

        model_config = DotDict(model_config)
        model_graph_config = DotDict(model_graph_config)
        self._graph_name_hidden = model_config.model.model.hidden_nodes_name

        static_graph_config, dynamic_graph_config = split_graph_config(
            model_graph_config, is_dataset_static, self._graph_name_hidden
        )

        self._graph_data = GraphCreator(static_graph_config).create()
        self.data_indices = data_indices
        self.statistics = statistics
        self.n_step_input = n_step_input
        self.n_step_output = n_step_output

        self.dataset_names = list(data_indices.keys())
        self.is_dataset_static = is_dataset_static
        self._graph_name_hidden = model_config.model.model.hidden_nodes_name

        self.num_channels = model_config.model.num_channels
        self.latent_skip = model_config.model.model.latent_skip

        trainable_parameters = broadcast_config_keys(
            model_config.model.trainable_parameters,
            data=self.dataset_names,
            hidden=self._graph_name_hidden,
        )
        self.node_attributes = NamedNodesAttributes(trainable_parameters, self._graph_data)
        self.use_encoder_data_output = {"grid": True, "obs": False}

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)
        self._assert_hidden_nodes_name(self._graph_name_hidden)

        # build networks
        self._build_networks(model_config, self._graph_data, dynamic_graph_config.edges)

        # build residual connection
        self._build_residual(model_config.model.residual)

        # build boundings
        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = build_boundings(model_config.model.bounding, dataset_names=self.dataset_names)

    def _hidden_coordinates(self) -> torch.Tensor:
        return self._graph_data[self._graph_name_hidden].x

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        # Multi-dataset: create dictionaries for each property
        self.num_input_channels = {}
        self.num_output_channels = {}
        self.num_input_channels_prognostic = {}
        self.num_input_channels_decoding_forcings = {}
        self._internal_input_idx = {}
        self._internal_output_idx = {}
        self._decoding_forcing_input_idx = {}
        self.input_dim = {}
        self.input_dim_latent = self._calculate_input_dim_latent()
        self.target_dim = {}
        self.output_dim = {}

        for dataset_name, dataset_indices in data_indices.items():
            self._internal_input_idx[dataset_name] = dataset_indices.model.input.prognostic
            self._internal_output_idx[dataset_name] = dataset_indices.model.output.prognostic
            self._decoding_forcing_input_idx[dataset_name] = [
                dataset_indices.name_to_index[name] for name in dataset_indices.model._forcing
            ]

            self.num_input_channels[dataset_name] = len(dataset_indices.model.input)
            self.num_input_channels_prognostic[dataset_name] = len(dataset_indices.model.input.prognostic)
            self.num_input_channels_decoding_forcings[dataset_name] = len(
                self._decoding_forcing_input_idx[dataset_name]
            )
            self.num_output_channels[dataset_name] = len(dataset_indices.model.output)

            self.input_dim[dataset_name] = self._calculate_input_dim(dataset_name)
            self.target_dim[dataset_name] = self._calculate_target_dim(dataset_name)
            self.output_dim[dataset_name] = self._calculate_output_dim(dataset_name)

    def _calculate_input_dim(self, dataset_name: str) -> int:
        if self.is_dataset_static[dataset_name]:
            return (
                self.n_step_input * self.num_input_channels[dataset_name]
                + COORDS_DIM
                + self.node_attributes.num_trainable_parameters.get(dataset_name, 0)
            )

        # time is already part of the grid dimension
        return (
            self.num_input_channels[dataset_name]
            + COORDS_DIM
            + self.node_attributes.num_trainable_parameters.get(dataset_name, 0)
        )

    def _calculate_input_dim_latent(self) -> int:
        """Calculate the latent input dimension."""
        nodes_name = self._graph_name_hidden if isinstance(self._graph_name_hidden, str) else self._graph_name_hidden[0]
        return COORDS_DIM + self.node_attributes.num_trainable_parameters.get(nodes_name, 0)

    @staticmethod
    def _as_hidden_node_names(
        hidden_nodes_name: str | list[str] | ListConfig,
    ) -> list[str]:
        if isinstance(hidden_nodes_name, str):
            return [hidden_nodes_name]

        if isinstance(hidden_nodes_name, (list, ListConfig)):
            return list(hidden_nodes_name)

        raise TypeError(
            f"Hidden nodes name must be a string or a list of strings, got {type(hidden_nodes_name)}",
        )

    def _assert_hidden_nodes_name(self, hidden_nodes_name: str) -> None:
        for hidden_name in self._as_hidden_node_names(hidden_nodes_name):
            assert (
                hidden_name in self._graph_data.node_types
            ), f"Hidden nodes name '{hidden_name}' not found in graph data node types {self._graph_data.node_types}"

    def _calculate_target_dim(self, dataset_name: str) -> int:
        # Default behaviour is to pass the same input as to the encoder.
        # TODO: abstract different options into the base class
        if self.use_encoder_data_output[dataset_name]:
            return self._calculate_input_dim(dataset_name)
        else:
            return COORDS_DIM + self.node_attributes.num_trainable_parameters.get(dataset_name, 0)

    def _calculate_output_dim(self, dataset_name: str) -> int:
        if self.is_dataset_static[dataset_name]:
            return self.n_step_output * self.num_output_channels[dataset_name]

        # time is already part of the grid dimension
        return self.num_output_channels[dataset_name]

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

    def _resolve_in_out_sharded(
        self,
        dataset_names: list[str],
        grid_shard_sizes: DatasetShardSizes | None,
    ) -> dict[str, bool]:
        in_out_sharded: dict[str, bool] = {}
        for dataset_name in dataset_names:
            if grid_shard_sizes is None:
                in_out_sharded[dataset_name] = False
            else:
                in_out_sharded[dataset_name] = grid_shard_sizes[dataset_name] is not None

        return in_out_sharded

    def _get_consistent_dim(self, x: dict[str, "Tensor | list[Tensor]"], dim: int) -> int:
        """Return a dimension size that is consistent across all datasets.

        Sparse-observation datasets carry their per-sample tensors as
        ``list[Tensor]`` (the list itself is the batch axis — no leading
        batch dim is added on collation). For ``dim == 0`` (batch size)
        the list length plays the role of ``shape[0]``. For other
        dimensions the per-sample tensor uses a different layout
        (``(E, N, V)``) so we cannot map the gridded ``dim`` index onto
        it; in that case list entries are skipped and the size is
        derived from the tensor entries only.
        """
        dim_sizes: list[int] = []
        for _x in x.values():
            if isinstance(_x.data, list):
                if dim == 0:
                    dim_sizes.append(len(_x.data))
                # Other dims live in a different per-sample layout for
                # sparse datasets — skip and rely on tensor entries.
                continue
            dim_sizes.append(_x.data.shape[dim])

        assert dim_sizes, f"_get_consistent_dim: no entries available for dim={dim}"
        # Assert all datasets have the same sizes
        assert all(bs == dim_sizes[0] for bs in dim_sizes), f"Dimensions must be the same across datasets: {dim_sizes}"

        return dim_sizes[0]

    @abstractmethod
    def _build_networks(self, model_config: DotDict, static_graph: HeteroData, graph_config: DotDict) -> None:
        """Builds the networks for the model."""
        pass

    @abstractmethod
    def _assemble_input(
        self,
        x,
        batch_size,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
    ):
        pass

    @abstractmethod
    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        pass

    def _build_residual(self, residual_config: DotDict) -> None:
        self.residual = torch.nn.ModuleDict()
        fused = uses_fused_dataset_graph(self._graph_data, self.dataset_names)
        for dataset_name in self.dataset_names:
            if not self.is_dataset_static[dataset_name]:
                LOGGER.info(f"Skipping residual connection for static dataset: {dataset_name}")
                continue

            data_node_name = dataset_name if fused else DEFAULT_DATASET_NAME
            self.residual[dataset_name] = instantiate(
                residual_config,
                graph=self._graph_data,
                data_node_name=data_node_name,
                statistics=self.statistics[dataset_name],
                data_indices=self.data_indices[dataset_name],
                dataset_name=dataset_name,
            )

    @abstractmethod
    def forward(
        self,
        batch: Batch,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        batch : Batch
            Typed batch envelope carrying ``data`` (per-dataset input tensors)
            and ``coords`` (per-dataset coordinate tensors). Concrete model
            implementations unpack ``batch.data`` and ``batch.coordinates`` at the
            top of the method.
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None.
        grid_shard_sizes : DatasetShardSizes, optional
            Per-dataset shard sizes for the grid dimension. ``None`` means the
            corresponding dataset is replicated, not sharded.
        **kwargs
            Additional model-specific arguments.

        Returns
        -------
        dict[str, Tensor]
            Output of the model, with the same shape as the input (sharded if
            the corresponding input dataset is sharded).
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
    ) -> dict[str, torch.Tensor]:
        """Prediction step for the model.

        Base implementation applies pre-processing, performs a forward pass, and applies post-processing.
        Subclasses can override this for different behavior, such as transport sampling.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data (before pre-processing).
        pre_processors : nn.Module
            Pre-processing module.
        post_processors : nn.Module
            Post-processing module.
        n_step_input : int
            Number of input timesteps.
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training.
        gather_out : bool
            Whether to gather output tensors across distributed processes.
        **kwargs
            Additional arguments.

        Returns
        -------
        dict[str, torch.Tensor]
            Model output (after post-processing).
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
            grid_shard_sizes: DatasetShardSizes | None = None
            if model_comm_group is not None:
                grid_shard_sizes = {}
                for dataset_name in dataset_names:
                    grid_shard_sizes[dataset_name] = get_shard_sizes(
                        x[dataset_name], -2, model_comm_group=model_comm_group
                    )
                    x[dataset_name] = shard_tensor(
                        x[dataset_name], -2, grid_shard_sizes[dataset_name], model_comm_group
                    )

            for dataset_name in dataset_names:
                x[dataset_name] = pre_processors[dataset_name](x[dataset_name], in_place=False)

            # Wrap into a Batch (no coords available at inference today; the
            # static-grid path inside the model uses the node-attribute buffers).
            forward_batch = Batch(data=x)

            # Perform forward pass
            y_hat = self.forward(
                forward_batch, model_comm_group=model_comm_group, grid_shard_sizes=grid_shard_sizes, **kwargs
            )

            # Apply post-processing
            for dataset_name in dataset_names:
                y_hat[dataset_name] = post_processors[dataset_name](y_hat[dataset_name], in_place=False)

            # Gather output if needed
            if gather_out and model_comm_group is not None:
                assert grid_shard_sizes is not None
                for dataset_name in dataset_names:
                    y_hat[dataset_name] = gather_tensor(
                        y_hat[dataset_name], -2, grid_shard_sizes[dataset_name], model_comm_group
                    )

        return y_hat

    @abstractmethod
    def fill_metadata(self, md_dict) -> None:
        """To be implemented in subclasses to fill model-specific metadata."""
        pass
