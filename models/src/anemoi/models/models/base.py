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
from collections.abc import Mapping
from typing import Optional

import torch
from hydra.utils import get_class
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import ListConfig
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.data.batch import Batch
from anemoi.models.data.sources import BaseTemplate
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.utils import model_is_distributed
from anemoi.models.layers.bounding import build_boundings
from anemoi.models.layers.graph import NodeTrainableParameters
from anemoi.models.layers.node_embedder import is_noop_node_embedder_config
from anemoi.models.models.target_features import DecodingTargetFeature
from anemoi.models.models.target_features import create_decoding_target_features
from anemoi.models.models.target_features import time_steps_per_node
from anemoi.models.utils.config import COORDS_DIM
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.models.utils.graphs import hidden_node_names
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

# Encoder dataset-fusing strategies currently implemented.
# - "none": no fusion; the node embedder (``model.encoders.<name>.node_embedder``) is a NoOpNodeEmbedder, so the
#   mapper takes a source dataset's assembled input directly. Normally one source dataset per encoder; several
#   are accepted only if they have the same number of input channels and at most one of them is present per batch
#   (e.g. multi-domain training).
# - "sequential": a projecting node embedder (e.g. LinearNodeEmbedder) projects every source dataset onto a
#   common number of input channels; each is then encoded in its own pass with the shared mapper weights, and the
#   per-dataset latents are combined by the latent aggregator.
SUPPORTED_FUSING_STRATEGIES = {"none", "sequential"}

# Strategies that are documented but not implemented yet.
PLANNED_FUSING_STRATEGIES = {"joint"}


class BaseGraphModel(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        static_graph_data: HeteroData,
        dynamic_graph_config: DotDict,
        data_indices: dict[str, IndexCollection],
        statistics: dict[str, dict],
        n_step_input: dict[str, int],
        n_step_output: dict[str, int],
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
        static_graph_data : HeteroData
            Static graph: the nodes and edges of the gridded datasets and the hidden meshes.
        dynamic_graph_config : DotDict
            Builders of the nodes and edges that touch tabular datasets, built per batch. Its ``edges``
            have an entry for every edge type, with no builders for the static ones.
        n_step_input : dict[str, int]
            Number of input time steps (windows, for tabular datasets) for each dataset. Gridded datasets embed
            all of them per node; tabular datasets stack their windows on the node axis, so each node carries one
            (see ``target_features.time_steps_per_node``).
        n_step_output : dict[str, int]
            Number of output time steps (windows, for tabular datasets) for each dataset, with the same per-node
            convention as ``n_step_input``.
        """
        super().__init__()

        model_config = DotDict(model_config)
        dynamic_graph_config = DotDict(dynamic_graph_config)
        self._graph_name_hidden = model_config.model.hidden_nodes_name
        self._graph_data = static_graph_data
        self.data_indices = data_indices
        self.statistics = statistics
        self.n_step_input = n_step_input
        self.n_step_output = n_step_output

        self.dataset_names = list(data_indices.keys())
        # Gridded datasets live in the static graph; tabular ones are built per batch.
        self.is_dataset_static = {name: name in static_graph_data.node_types for name in self.dataset_names}
        self._build_hidden_node_routing()

        self.latent_skip = model_config.model.latent_skip

        self.node_attributes = NodeTrainableParameters(model_config.node_trainable_parameters, self._graph_data)

        self.dynamic_node_attributes: dict[str, dict[str, object]] = {}
        self.dynamic_node_attribute_dims: dict[str, int] = {}

        self._configure_dynamic_node_attributes(dynamic_graph_config.nodes)

        self._build_encoder_routing(model_config.encoders)
        self._build_decoder_routing(model_config.decoders)

        self._calculate_shapes_and_indices(data_indices)

        self._assert_model_routing()
        self._assert_matching_indices(data_indices)
        self._assert_hidden_nodes_name()

        # build networks
        self._build_networks(model_config, self._graph_data, dynamic_graph_config.edges)

        # build residual connection
        self._build_residual(
            get_multiple_datasets_config(model_config.residual),
            sparse_projector_config=model_config.get("sparse_projector", {}),
        )

        # build boundings
        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        # Multi-dataset: create ModuleDict with ModuleList per dataset
        self.boundings = build_boundings(
            get_multiple_datasets_config(model_config.get("bounding", [])),
            data_indices=self.data_indices,
            statistics=self.statistics,
        )

    def __setstate__(self, state: dict) -> None:
        """Fill in the hidden-mesh routing of models pickled before it existed.

        Inference checkpoints pickle the whole model, so unpickling skips ``__init__``.
        Models pickled before the hidden mesh was routed per dataset have neither
        ``hidden_names`` nor ``dataset2hidden``, and a single processor graph provider
        instead of one per hidden mesh.

        Parameters
        ----------
        state : dict
            The pickled ``__dict__`` of the model.
        """
        super().__setstate__(state)
        if "dataset2hidden" in self.__dict__:
            return
        self._build_hidden_node_routing()
        provider = self._modules.get("processor_graph_provider")
        if isinstance(self._graph_name_hidden, str) and isinstance(provider, nn.Module):
            if not isinstance(provider, nn.ModuleDict):
                self._modules["processor_graph_provider"] = nn.ModuleDict({self._graph_name_hidden: provider})

    def _configure_dynamic_node_attributes(self, dynamic_node_config: DotDict) -> None:
        """Configure runtime node attributes for models that support them."""
        del dynamic_node_config

    def _hidden_coordinates(self, hidden_name: str | None = None) -> torch.Tensor:
        hidden_name = self.hidden_names[0] if hidden_name is None else hidden_name
        return self._graph_data[hidden_name].x

    def _build_hidden_node_routing(self) -> None:
        """Build and validate the dataset routing for hidden nodes."""
        hidden_nodes_name = self._graph_name_hidden
        if isinstance(hidden_nodes_name, str):
            hidden_nodes_name = dict.fromkeys(self.dataset_names, hidden_nodes_name)

        if isinstance(hidden_nodes_name, (dict, DictConfig)):
            self.dataset2hidden = dict(hidden_nodes_name)
            missing_datasets = set(self.dataset_names) - set(self.dataset2hidden)
            unknown_datasets = set(self.dataset2hidden) - set(self.dataset_names)
            assert not missing_datasets, f"Hidden node names are missing for datasets {sorted(missing_datasets)}."
            assert (
                not unknown_datasets
            ), f"Hidden node names were configured for unknown datasets {sorted(unknown_datasets)}."
            self.hidden_names = list(dict.fromkeys(self.dataset2hidden.values()))
        elif isinstance(hidden_nodes_name, (list, ListConfig)):
            # Hierarchical levels: every dataset is encoded into the first level.
            self.hidden_names = list(hidden_nodes_name)
            self.dataset2hidden = dict.fromkeys(self.dataset_names, self.hidden_names[0])
        else:
            raise TypeError(
                f"Hidden nodes name must be a string, a list of strings, or a dataset mapping, got {type(hidden_nodes_name)}",
            )

    def _build_encoder_routing(self, encoders_config: DotDict) -> None:
        """Builds the dataset routing for encoders."""
        self.dataset2encoder: dict[str, str] = {}
        self.encoder2datasets: dict[str, list[str]] = {}
        self.encoder_fusing_strategy: dict[str, str] = {}
        # Width of the source latent returned by encoders whose mapper updates its source nodes
        # (e.g. GNNForwardMapper); other mappers pass the source input through.
        self.encoder_src_latent_dim: dict[str, int] = {}
        for encoder_name, encoder_config in encoders_config.items():
            mapper_target = encoder_config.get("mapper", {}).get("_target_")
            if mapper_target is not None and getattr(get_class(mapper_target), "returns_src_latent", False):
                self.encoder_src_latent_dim[encoder_name] = encoder_config.mapper.num_channels

            datasets_to_encode = list(encoder_config["source_datasets"])
            self.encoder2datasets[encoder_name] = datasets_to_encode
            for d in datasets_to_encode:
                if d in self.dataset2encoder:
                    raise ValueError(
                        f"Dataset '{d}' is listed under source_datasets of both encoder "
                        f"'{self.dataset2encoder[d]}' and encoder '{encoder_name}'. Each dataset must "
                        "be encoded by exactly one encoder; list it once, and use "
                        "dataset_fusing_strategy to control how that encoder combines its sources."
                    )
                self.dataset2encoder[d] = encoder_name
            self.encoder_fusing_strategy[encoder_name] = self._validate_fusing_strategy(encoder_name, encoder_config)

        self.input_datasets = list(self.dataset2encoder.keys())

    @staticmethod
    def _validate_fusing_strategy(encoder_name: str, encoder_config: DotDict) -> str:
        """Return the encoder's fusing strategy after checking it against its node embedder setting.

        Also enforced by the config schema, but repeated here so that models built without
        config validation fail just as early.
        """
        strategy = encoder_config.get("dataset_fusing_strategy", "none")
        node_embedder_config = encoder_config.get("node_embedder")
        if strategy in PLANNED_FUSING_STRATEGIES:
            msg = (
                f"Encoder '{encoder_name}' sets dataset_fusing_strategy '{strategy}', which is planned but "
                f"not yet supported. Valid options are: {sorted(SUPPORTED_FUSING_STRATEGIES)}."
            )
            raise ValueError(msg)
        if strategy not in SUPPORTED_FUSING_STRATEGIES:
            msg = (
                f"Encoder '{encoder_name}' has unsupported dataset_fusing_strategy '{strategy}'. "
                f"Valid options are: {sorted(SUPPORTED_FUSING_STRATEGIES)}."
            )
            raise ValueError(msg)
        if strategy == "sequential" and is_noop_node_embedder_config(node_embedder_config):
            msg = (
                f"Encoder '{encoder_name}' uses dataset_fusing_strategy 'sequential', which needs a projecting "
                "node embedder (not NoOpNodeEmbedder), e.g.\n"
                f"    model.encoders.{encoder_name}.node_embedder:\n"
                "      _target_: anemoi.models.layers.node_embedder.LinearNodeEmbedder\n"
                "      num_channels: 512"
            )
            raise ValueError(msg)
        if strategy == "none" and not is_noop_node_embedder_config(node_embedder_config):
            LOGGER.warning(
                "Encoder '%s' uses dataset_fusing_strategy 'none'; its node embedder %s is ignored and "
                "NoOpNodeEmbedder is used instead.",
                encoder_name,
                node_embedder_config.get("_target_"),
            )
        return strategy

    def _build_decoder_routing(self, decoders_config: DotDict) -> None:
        """Builds the dataset routing for decoders."""
        self.dataset2decoder: dict[str, str] = {}
        self.decoder2datasets: dict[str, list[str]] = {}
        self.decoders_target_input: dict[str, DecodingTargetFeature] = {}
        for decoder_name, decoder_config in decoders_config.items():
            datasets_to_decode = decoder_config["target_datasets"]
            self.decoder2datasets[decoder_name] = datasets_to_decode
            for d in datasets_to_decode:
                self.dataset2decoder[d] = decoder_name

            self.decoders_target_input[decoder_name] = create_decoding_target_features(
                decoder_config.target_node_features, datasets_to_decode, self
            )

        self.target_datasets = list(self.dataset2decoder.keys())

    def _assert_model_routing(self) -> None:
        """Asserts that the model routing is valid."""
        not_input_datasets = set(self.dataset2encoder) - set(self.input_dim)
        assert (
            not not_input_datasets
        ), f"Datasets {not_input_datasets} are referenced by encoders but missing from data_indices provided to the model. "

        not_target_datasets = set(self.dataset2decoder) - set(self.output_dim)
        assert (
            not not_target_datasets
        ), f"Datasets {not_target_datasets} are referenced by decoders but missing from data_indices provided to the model. "

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

    def _assert_hidden_nodes_name(self) -> None:
        for hidden_name in self.hidden_names:
            assert (
                hidden_name in self._graph_data.node_types
            ), f"Hidden nodes name '{hidden_name}' not found in graph data node types {self._graph_data.node_types}"

    def _calculate_input_dim(self, dataset_name: str) -> int:
        return (
            time_steps_per_node(self, dataset_name, self.n_step_input) * self.num_input_channels[dataset_name]
            + COORDS_DIM
            + self.node_attributes.num_trainable_parameters.get(dataset_name, 0)
            + self.dynamic_node_attribute_dims.get(dataset_name, 0)
        )

    def _calculate_input_dim_latent(self) -> int:
        """Calculate the latent input dimension of the hidden meshes the datasets are encoded into.

        Hierarchical models size their deeper levels themselves, so only the first level counts there.
        """
        hidden_dims = {
            hidden_name: COORDS_DIM + self.node_attributes.num_trainable_parameters.get(hidden_name, 0)
            for hidden_name in dict.fromkeys(self.dataset2hidden.values())
        }
        assert (
            len(set(hidden_dims.values())) == 1
        ), f"All hidden meshes must have the same node attribute dimension, got {hidden_dims}."
        return next(iter(hidden_dims.values()))

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
        return time_steps_per_node(self, dataset_name, self.n_step_output) * self.num_output_channels[dataset_name]

    @staticmethod
    def _as_hidden_node_names(
        hidden_nodes_name: str | list[str] | ListConfig | Mapping[str, str],
    ) -> list[str]:
        return hidden_node_names(hidden_nodes_name)

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

    def _resolve_in_out_sharded(self, batch: Batch) -> dict[str, bool]:
        """Per-dataset flag indicating whether the dataset is grid-sharded.

        Sharding metadata is carried by each source, which exposes it via
        ``.flatten().shard_sizes``.

        ``None`` means that dataset is replicated, not sharded.
        """
        return {dataset_name: source.shard_sizes is not None for dataset_name, source in batch.items()}

    def _get_active_hidden_name(self, dataset_names: list[str]) -> str:
        """Return the single hidden mesh used by the datasets in the batch."""
        active_hidden_names = list(dict.fromkeys(self.dataset2hidden[dataset_name] for dataset_name in dataset_names))
        if len(active_hidden_names) != 1:
            raise ValueError(f"All datasets in a batch must use the same hidden mesh, got {active_hidden_names}.")
        return active_hidden_names[0]

    @abstractmethod
    def _build_networks(self, model_config: DotDict, static_graph: HeteroData, graph_config: DotDict) -> None:
        """Builds the networks for the model."""
        pass

    @abstractmethod
    def _assemble_input(
        self,
        x,
        batch_size,
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
            # Only gridded datasets have a static grid, so a dynamic one is tabular.
            if not self.is_dataset_static.get(dataset_name, True):
                msg = (
                    f"model.residual configures a residual connection for dataset '{dataset_name}', which is "
                    "tabular. Residual connections need a gridded dataset with an explicit "
                    f"time axis; remove model.residual.datasets.{dataset_name}."
                )
                raise ValueError(msg)

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

        for hidden_name in self.hidden_names:
            node_attributes_graph[hidden_name].x = self._graph_data[hidden_name].x
            node_attributes_graph[hidden_name].num_nodes = self._graph_data[hidden_name].num_nodes

        return node_attributes_graph

    @abstractmethod
    def forward(
        self,
        batch: Batch,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        batch : Batch
            Typed batch envelope carrying ``data`` (per-dataset input tensors)
            and ``coords`` (per-dataset coordinate tensors). Concrete model
            implementations unpack ``batch.data`` and ``batch.coordinates`` at the
            top of the method. Per-dataset grid sharding is carried by the batch
            (``batch.shard_sizes``) and read through the source views.
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None.
        **kwargs
            Additional model-specific arguments.

        Returns
        -------
        dict[str, Tensor]
            Output of the model, with the same shape as the input (sharded if
            the corresponding input dataset is sharded).
        """
        pass

    @staticmethod
    def _shard_target_template(
        target_template: dict[str, BaseTemplate],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> dict[str, BaseTemplate]:
        """Split the target templates across ``model_comm_group``, as the inputs are split for prediction.

        The decoder reads its target nodes from the template, so with a distributed model each rank's
        template must describe its own shard of the nodes.
        """
        if not model_is_distributed(model_comm_group):
            return target_template
        return {dataset_name: template.shard(model_comm_group) for dataset_name, template in target_template.items()}

    def _prepare_prediction_inputs(
        self,
        x: Batch,
        target_forcing: Optional[Batch],
        pre_processors: nn.ModuleDict,
        model_comm_group: Optional[ProcessGroup] = None,
        spatial_pre_processors: Optional[nn.ModuleDict] = None,
        **kwargs,
    ) -> tuple[Batch, Batch]:
        """Shard, project and normalise the inputs and the decoder conditioning of a prediction step.

        Parameters
        ----------
        x : Batch
            Input batched data (before pre-processing).
        target_forcing : Optional[Batch]
            Decoder conditioning (before pre-processing): the forcing variables at the output valid times.
            ``None`` means there are no forcings.
        pre_processors : nn.ModuleDict
            Pre-processing module.
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training.
        spatial_pre_processors : Optional[nn.ModuleDict]
            Spatial preprocessors keyed by dataset name (e.g. CrossGridProjector).
            Applied after grid sharding and before normalisation, as in training.
        **kwargs
            Additional arguments for the pre-processors.

        Returns
        -------
        tuple[Batch, Batch]
            The normalised inputs and the normalised decoder conditioning, split across
            ``model_comm_group`` when the model is distributed.
        """
        dataset_names = list(x.keys())
        if target_forcing is None:
            target_forcing = Batch({})

        if model_is_distributed(model_comm_group):
            for dataset_name in dataset_names:
                x = x.replace(dataset_name, x[dataset_name].shard(model_comm_group))
            for dataset_name in target_forcing.dataset_names:
                target_forcing = target_forcing.replace(
                    dataset_name, target_forcing[dataset_name].shard(model_comm_group)
                )

        # Spatial preprocessing: applied after grid sharding and, as in training, before
        # normalisation, so projectors see raw values. The projected source is on the
        # dataset's graph node set.
        for dataset_name in dataset_names:
            if spatial_pre_processors is not None and dataset_name in spatial_pre_processors:
                projected = spatial_pre_processors[dataset_name].project_source(
                    x[dataset_name],
                    self._graph_data[dataset_name].x,
                    model_comm_group=model_comm_group,
                )
                x = x.replace(dataset_name, projected)

        processed_batch = x
        for dataset_name in dataset_names:
            processed_batch = processed_batch.replace(
                dataset_name,
                pre_processors[dataset_name](x[dataset_name], in_place=False, **kwargs),
            )

        # The target forcings condition the decoder, and need to go through the input processors
        processed_forcing = target_forcing
        for dataset_name in target_forcing.dataset_names:
            if dataset_name not in pre_processors:
                continue
            processed_forcing = processed_forcing.replace(
                dataset_name,
                pre_processors[dataset_name](target_forcing[dataset_name], in_place=False, **kwargs),
            )

        return processed_batch, processed_forcing

    def predict_step(
        self,
        x: Batch,
        target_template: dict[str, BaseTemplate],
        pre_processors: nn.ModuleDict,
        post_processors: nn.ModuleDict,
        target_forcing: Optional[Batch] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        spatial_pre_processors: Optional[nn.ModuleDict] = None,
        **kwargs,
    ) -> Batch:
        """Prediction step for the model.

        Base implementation applies pre-processing, performs a forward pass, and applies post-processing.
        Subclasses can override this for different behavior, such as transport sampling.

        Parameters
        ----------
        x : Batch
            Input batched data (before pre-processing).
        target_template : dict[str, BaseTemplate]
            What to decode, one template per decoded dataset: the output nodes, layout,
            variables and time steps.
        pre_processors : nn.ModuleDict
            Pre-processing module.
        post_processors : nn.ModuleDict
            Post-processing module.
        target_forcing : Optional[Batch]
            Decoder conditioning (before pre-processing): the forcing variables at the output valid times.
            ``None`` means there are no forcings.
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training.
        gather_out : bool
            Whether to gather output tensors across distributed processes.
        spatial_pre_processors : Optional[nn.ModuleDict]
            Spatial preprocessors keyed by dataset name (e.g. CrossGridProjector).
            Applied after grid sharding and before normalisation, as in training.
        **kwargs
            Additional arguments.

        Returns
        -------
        Batch
            Model output (after post-processing), built from the target information.
        """
        with torch.no_grad():
            target_template = self._shard_target_template(target_template, model_comm_group)
            processed_batch, processed_target = self._prepare_prediction_inputs(
                x,
                target_forcing,
                pre_processors,
                model_comm_group=model_comm_group,
                spatial_pre_processors=spatial_pre_processors,
                **kwargs,
            )

            # Perform forward pass
            y_hat = self.forward(
                processed_batch,
                target_forcings=processed_target,
                target_template=target_template,
                model_comm_group=model_comm_group,
                **kwargs,
            )

            # Apply post-processing
            for dataset_name in y_hat.dataset_names:
                if dataset_name not in post_processors:
                    continue
                y_hat = y_hat.replace(
                    dataset_name,
                    post_processors[dataset_name](y_hat[dataset_name], in_place=False),
                )

            # Gather the output if needed
            if gather_out:
                for dataset_name in y_hat.dataset_names:
                    y_hat = y_hat.replace(dataset_name, y_hat[dataset_name].allgather(model_comm_group))

        return y_hat

    @abstractmethod
    def fill_metadata(self, md_dict) -> None:
        """To be implemented in subclasses to fill model-specific metadata."""
        pass
