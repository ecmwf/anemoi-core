# (C) Copyright 2025-2026 Anemoi contributors.
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
from omegaconf import DictConfig
from omegaconf import ListConfig
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.bounding import build_boundings
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.models.target_features import DecodingTargetFeature
from anemoi.models.models.target_features import create_decoding_target_features
from anemoi.models.utils.config import broadcast_config_keys
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class BaseGraphModel(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DictConfig,
        data_indices: dict,
        statistics: dict,
        n_step_input: int,
        n_step_output: int,
        graph_data: HeteroData,
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
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()
        self._graph_data = graph_data
        self.data_indices = data_indices
        self.statistics = statistics
        self.n_step_input = n_step_input
        self.n_step_output = n_step_output

        self.dataset_names = list(data_indices.keys())
        self._graph_name_hidden = model_config.model.model.hidden_nodes_name

        self.latent_skip = model_config.model.model.latent_skip

        trainable_parameters = broadcast_config_keys(
            model_config.model.node_trainable_parameters,
            data=self.dataset_names,
            hidden=self._graph_name_hidden,
        )
        self.node_attributes = NamedNodesAttributes(trainable_parameters, self._build_named_node_attributes_graph())

        self._build_encoder_routing(model_config.model.encoders)
        self._build_decoder_routing(model_config.model.decoders)

        self._calculate_shapes_and_indices(data_indices)

        self._assert_model_routing()
        self._assert_matching_indices(data_indices)
        self._assert_hidden_nodes_name(self._graph_name_hidden)

        # build networks
        self._build_networks(model_config.model)

        # build residual connection
        self._build_residual(
            get_multiple_datasets_config(model_config.model.residual),
            sparse_projector_config=model_config.model.get("sparse_projector", {}),
        )

        # build boundings
        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        # Multi-dataset: create ModuleDict with ModuleList per dataset
        self.boundings = build_boundings(
            get_multiple_datasets_config(model_config.model.get("bounding", [])),
            data_indices=self.data_indices,
            statistics=self.statistics,
        )

    def _build_encoder_routing(self, encoders_config: DotDict) -> None:
        """Builds the dataset routing for encoders."""
        self.dataset2encoder: dict[str, str] = {}
        self.encoder2datasets: dict[str, list[str]] = {}
        self.encoder_fusing_strategy: dict[str, str] = {}
        for encoder_name, encoder_config in encoders_config.items():
            datasets_to_encode = encoder_config["source_datasets"]
            self.encoder2datasets[encoder_name] = datasets_to_encode
            for d in datasets_to_encode:
                self.dataset2encoder[d] = encoder_name
            self.encoder_fusing_strategy[encoder_name] = encoder_config.dataset_fusing_strategy

        self.input_datasets = list(self.dataset2encoder.keys())

    def _build_decoder_routing(self, decoders_config: DotDict) -> None:
        """Builds the dataset routing for decoders."""
        self.dataset2decoder: dict[str, str] = {}
        self.decoder2datasets: dict[str, list[str]] = {}
        self.decoders_target_input: dict[str, DecodingTargetFeature] = {}
        for decoder_name, decoder_config in decoders_config.items():
            datasets_to_decode = decoder_config["target_datasets"]
            self.decoder2datasets[decoder_name] = datasets_to_decode
            assert len(datasets_to_decode) == 1, "Each decoder must be associated with exactly one dataset for now."
            for d in datasets_to_decode:
                self.dataset2decoder[d] = decoder_name

            self.decoders_target_input[decoder_name] = create_decoding_target_features(
                decoder_config.input_target_features, datasets_to_decode, self
            )

        self.target_datasets = list(self.dataset2decoder.keys())

    def _assert_model_routing(self) -> None:
        """Asserts that the model routing is valid."""
        not_input_datasets = set(self.input_datasets) - set(self.input_dim.keys())
        assert all(
            d in self.input_datasets for d in self.dataset2encoder.keys()
        ), f"Datasets {not_input_datasets} are in input_datasets but not in data_indices provided to the model. "

        not_target_datasets = set(self.target_datasets) - set(self.output_dim.keys())
        assert all(
            d in self.target_datasets for d in self.dataset2decoder.keys()
        ), f"Datasets {not_target_datasets} are in target_datasets but not in data_indices provided to the model. "

        for encoder_name, fusing_strategy in self.encoder_fusing_strategy.items():
            if fusing_strategy not in ("not_supported"):
                raise ValueError(f"Encoder '{encoder_name}' has unsupported fusing strategy '{fusing_strategy}'.")

        # Validated here. The target dimension may depend on the shapes computed in _calculate_shapes_and_indices
        for target_features in self.decoders_target_input.values():
            target_features.validate()

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        """Compute per-dataset input/output channel counts, dimensions and internal data indices."""
        # Multi-dataset: create dictionaries for each property
        self.num_input_channels = {}
        self.num_output_channels = {}
        self.num_input_channels_prognostic = {}
        self.num_input_channels_forcings = {}
        self.num_input_channels_decoding_forcings = {}
        self._internal_input_idx = {}
        self._internal_output_idx = {}
        self._forcing_input_idx = {}
        self.input_dim = {}
        self.input_dim_latent = self._calculate_input_dim_latent()
        self.target_dim = {}
        self.output_dim = {}

        for dataset_name, dataset_indices in data_indices.items():
            self._internal_input_idx[dataset_name] = dataset_indices.model.input.prognostic
            self._internal_output_idx[dataset_name] = dataset_indices.model.output.prognostic
            self._forcing_input_idx[dataset_name] = dataset_indices.model.input.forcing

            self.num_input_channels[dataset_name] = len(dataset_indices.model.input)
            self.num_input_channels_forcings[dataset_name] = len(dataset_indices.model.input.forcing)
            self.num_input_channels_prognostic[dataset_name] = len(dataset_indices.model.input.prognostic)
            self.num_output_channels[dataset_name] = len(dataset_indices.model.output)

            self.input_dim[dataset_name] = self._calculate_input_dim(dataset_name)
            self.target_dim[dataset_name] = self._calculate_target_dim(dataset_name)
            self.output_dim[dataset_name] = self._calculate_output_dim(dataset_name)

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

    def _calculate_input_dim(self, dataset_name: str) -> int:
        """Calculate the encoder input dimension for a given dataset."""
        return self.n_step_input * self.num_input_channels[dataset_name] + self.node_attributes.attr_ndims[dataset_name]

    def _calculate_input_dim_latent(self) -> int:
        """Calculate the latent input dimension."""
        nodes_name = self._graph_name_hidden if isinstance(self._graph_name_hidden, str) else self._graph_name_hidden[0]
        return self.node_attributes.attr_ndims[nodes_name]

    def _calculate_target_dim(self, dataset_name: str) -> int:
        """Calculate the decoder target input dimension for a given dataset.

        Decoder target features are per-node vectors attached to the destination nodes of the
        hidden-to-data decoder. The returned width is the sum
        of the feature blocks listed in ``decoders_target_input`` for this dataset's decoder.
        """
        if dataset_name not in self.dataset2decoder:
            LOGGER.warning(
                "Dataset '%s' does not have a decoder associated with it. Target dimension will be calculated as 0.",
                dataset_name,
            )
            return 0

        return self.decoders_target_input[self.dataset2decoder[dataset_name]].dim

    def _calculate_output_dim(self, dataset_name: str) -> int:
        """Calculate the decoder output dimension for a given dataset."""
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

    def _build_residual(self, residual_configs: dict[str, DotDict], sparse_projector_config: DotDict) -> None:
        """Instantiate the per-dataset residual connection modules."""
        self.residual = torch.nn.ModuleDict()
        sparse_projector_num_chunks = sparse_projector_config.get("num_chunks", 1)
        for dataset_name, residual_config in residual_configs.items():
            assert residual_config is not None, f"Residual config for dataset '{dataset_name}' is None."
            self.residual[dataset_name] = instantiate(
                residual_config,
                graph=self._graph_data,
                data_node_name=dataset_name,
                statistics=self.statistics[dataset_name],
                data_indices=self.data_indices[dataset_name],
                dataset_name=dataset_name,
                sparse_projector_num_chunks=sparse_projector_num_chunks,
            )

    def _build_named_node_attributes_graph(self) -> HeteroData:
        node_attributes_graph = HeteroData()
        for dataset_name in self.dataset_names:
            node_attributes_graph[dataset_name].x = self._graph_data[dataset_name].x
            node_attributes_graph[dataset_name].num_nodes = self._graph_data[dataset_name].num_nodes

        for hidden_name in self._as_hidden_node_names(self._graph_name_hidden):
            node_attributes_graph[hidden_name].x = self._graph_data[hidden_name].x
            node_attributes_graph[hidden_name].num_nodes = self._graph_data[hidden_name].num_nodes

        return node_attributes_graph

    @abstractmethod
    def forward(
        self,
        x: dict[str, Tensor],
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        x : dict[str, Tensor]
            Input data.
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

            # Perform forward pass
            y_hat = self.forward(
                x,
                model_comm_group=model_comm_group,
                grid_shard_sizes=grid_shard_sizes,
                **kwargs,
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
