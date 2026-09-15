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

from anemoi.graphs.projection_helpers import DEFAULT_EDGE_RELATION_NAME
from anemoi.graphs.projection_helpers import graph_participants
from anemoi.graphs.projection_helpers import participant_node_name
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.bounding import build_boundings
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.graph_provider import BaseGraphProvider
from anemoi.models.layers.graph_provider import ParticipantSwitchingGraphProvider
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.residual import TruncatedConnection
from anemoi.models.layers.target_features import DecodingTargetFeature
from anemoi.models.layers.target_features import create_decoding_target_features
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
        self._participants = self._infer_participants()
        self._active_participant: str | None = self._participants[0] if self._participants else None

        self.latent_skip = model_config.model.model.latent_skip

        self.node_attributes = NamedNodesAttributes(
            model_config.model.node_trainable_parameters,
            self._build_named_node_attributes_graph(),
            participants=self._participants,
        )
        self.node_attributes.set_active_participant(self._active_participant)

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

    def _infer_participants(self) -> list[str]:
        """Participants of the graph, empty for a single-domain (unsuffixed) graph.

        A multi-participant graph carries one node group per participant for every dataset
        and hidden node group (``data_west``, ``hidden_west``, ...); all of them must name
        the same participants.
        """
        node_names = self.dataset_names + self._as_hidden_node_names(self._graph_name_hidden)
        participants = graph_participants(self._graph_data, node_names[0])

        for node_name in node_names[1:]:
            other = graph_participants(self._graph_data, node_name)
            if other != participants:
                msg = (
                    f"Node group '{node_name}' has participants {other}, but '{node_names[0]}' has "
                    f"{participants}. All node groups must have the same participants."
                )
                raise ValueError(msg)

        return participants

    @property
    def participants(self) -> list[str]:
        """Participants this model can switch between, empty for a single-domain graph."""
        return list(self._participants)

    @property
    def active_participant(self) -> str | None:
        """Participant whose graph the model currently uses, ``None`` if there are none."""
        return self._active_participant

    def set_active_participant(self, participant: str | None) -> None:
        """Select the participant whose graph the model uses from now on.

        Participants sharing one graph are not represented in the graph, so this is a no-op
        for a single-domain graph and callers need not know which case they are in.
        """
        if not self._participants:
            return

        if participant not in self._participants:
            msg = f"Unknown participant '{participant}', expected one of {self._participants}."
            raise ValueError(msg)

        self._active_participant = participant
        self.node_attributes.set_active_participant(participant)
        for module in self.modules():
            if isinstance(module, ParticipantSwitchingGraphProvider):
                module.set_active_participant(participant)

    def node_name(self, name: str) -> str:
        """Return the node-group name of ``name`` for the active participant."""
        return participant_node_name(name, self._active_participant)

    def _participant_node_names(self, name: str) -> list[str]:
        """Return the node-group names of ``name``, one per participant."""
        if not self._participants:
            return [name]
        return [participant_node_name(name, participant) for participant in self._participants]

    def _create_graph_provider(
        self,
        src_name: str,
        dst_name: str,
        edge_attributes: Optional[list[str]],
        trainable_size: int,
        trainable_size_key: str,
    ) -> BaseGraphProvider:
        """Create the edge provider for ``src_name -> dst_name``, one per participant if needed."""
        if not self._participants:
            return create_graph_provider(
                graph=self._graph_data[(src_name, DEFAULT_EDGE_RELATION_NAME, dst_name)],
                edge_attributes=edge_attributes,
                src_size=self.node_attributes.num_nodes[src_name],
                dst_size=self.node_attributes.num_nodes[dst_name],
                trainable_size=trainable_size,
            )

        if trainable_size:
            # A trainable tensor per participant changes the set of parameters receiving a
            # gradient between steps, which DDP rejects under static_graph=True.
            msg = (
                f"{trainable_size_key} must be 0 when training on the participants {self._participants} "
                f"(got {trainable_size}): the participants have different edges, so a trainable edge "
                "tensor would change the set of parameters receiving gradients between steps."
            )
            raise ValueError(msg)

        providers = {}
        for participant in self._participants:
            src = participant_node_name(src_name, participant)
            dst = participant_node_name(dst_name, participant)
            providers[participant] = create_graph_provider(
                graph=self._graph_data[(src, DEFAULT_EDGE_RELATION_NAME, dst)],
                edge_attributes=edge_attributes,
                src_size=self.node_attributes.num_nodes[src],
                dst_size=self.node_attributes.num_nodes[dst],
                trainable_size=0,
            )

        provider = ParticipantSwitchingGraphProvider(providers)
        provider.set_active_participant(self._active_participant)
        return provider

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
                decoder_config.target_node_features, datasets_to_decode, self
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

    def _build_latent_aggregator(self, aggregator_config: DotDict) -> None:
        """Build the latent aggregator."""
        latent_aggregator_channels = {
            dataset_name: self.encoder[self.dataset2encoder[dataset_name]].hidden_dim
            for dataset_name in self.input_datasets
        }

        self.latent_aggregator = instantiate(
            aggregator_config,
            _recursive_=False,
            input_channels=self.input_dim_latent,
            source_channels=latent_aggregator_channels,
        )

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
            node_name = self.node_name(hidden_name)
            assert (
                node_name in self._graph_data.node_types
            ), f"Hidden nodes name '{node_name}' not found in graph data node types {self._graph_data.node_types}"

    def _calculate_input_dim(self, dataset_name: str) -> int:
        """Calculate the encoder input dimension for a given dataset."""
        return (
            self.n_step_input * self.num_input_channels[dataset_name]
            + self.node_attributes.attr_ndims[self.node_name(dataset_name)]
        )

    def _calculate_input_dim_latent(self) -> int:
        """Calculate the latent input dimension."""
        nodes_name = self._graph_name_hidden if isinstance(self._graph_name_hidden, str) else self._graph_name_hidden[0]
        return self.node_attributes.attr_ndims[self.node_name(nodes_name)]

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
                data_node_name=self.node_name(dataset_name),
                statistics=self.statistics[dataset_name],
                data_indices=self.data_indices[dataset_name],
                dataset_name=dataset_name,
                sparse_projector_num_chunks=sparse_projector_num_chunks,
            )
            if self._participants and isinstance(self.residual[dataset_name], TruncatedConnection):
                # A truncated connection is grid-sized, so it would have to be built once per
                # participant; only participant-invariant residuals are supported so far.
                msg = (
                    f"model.residual for dataset '{dataset_name}' is a TruncatedConnection, which is not "
                    f"supported when training on the participants {self._participants}."
                )
                raise NotImplementedError(msg)

    def _build_named_node_attributes_graph(self) -> HeteroData:
        node_attributes_graph = HeteroData()
        for dataset_name in self.dataset_names:
            for node_name in self._participant_node_names(dataset_name):
                node_attributes_graph[node_name].x = self._graph_data[node_name].x
                node_attributes_graph[node_name].num_nodes = self._graph_data[node_name].num_nodes

        for hidden_name in self._as_hidden_node_names(self._graph_name_hidden):
            for node_name in self._participant_node_names(hidden_name):
                node_attributes_graph[node_name].x = self._graph_data[node_name].x
                node_attributes_graph[node_name].num_nodes = self._graph_data[node_name].num_nodes

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
