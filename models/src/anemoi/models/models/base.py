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
from collections import defaultdict
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import ListConfig
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphCreator

# from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
# from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.models.data.batch import Batch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.bounding import build_boundings
from anemoi.models.layers.graph import NodeTrainableParameters
from anemoi.models.utils.config import COORDS_DIM
from anemoi.models.models.target_features import DecodingTargetFeature
from anemoi.models.models.target_features import create_decoding_target_features
from anemoi.models.utils.config import broadcast_config_keys
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

# Encoder dataset-fusing strategies currently implemented.
# - "none": no fusion. The natural choice for a single-source encoder; several source datasets
#   are only accepted if they share an input dimension, and each is encoded independently.
# - "sequential": one encoder pass per source dataset with shared weights; the per-dataset
#   latents are combined by the latent aggregator.
# - "joint": a single encoder pass over the union of all source nodes.
SUPPORTED_FUSING_STRATEGIES = {"none", "sequential", "joint"}

# Strategies for which per-dataset "thin" source projections are built, so that source
# datasets of differing feature widths (channel counts) can share one encoder.
PROJECTING_FUSING_STRATEGIES = {"sequential", "joint"}


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

        self.latent_skip = model_config.model.model.latent_skip

        trainable_parameters = broadcast_config_keys(
            model_config.model.trainable_parameters,
            data=self.dataset_names,
            hidden=self._graph_name_hidden,
        )
        self.node_attributes = NodeTrainableParameters(trainable_parameters, self._graph_data)

        # HACK: returns True for "data" and "grid" labels, False for everything else (obs)
        # TODO: this info should come through the config, not be hardcoded here.
        # The model should not know about the dataset names.
        self.use_encoder_data_output = defaultdict(
            bool, {"data": True, "grid": True, "era5": True, "cerra": True, "operan": True}
        )
        self.dynamic_node_attributes: dict[str, dict[str, object]] = {}
        self.dynamic_node_attribute_dims: dict[str, int] = {}

        self._configure_dynamic_node_attributes(dynamic_graph_config.nodes)

        self._build_encoder_routing(model_config.model.encoders)
        self._build_decoder_routing(model_config.model.decoders)

        self._calculate_shapes_and_indices(data_indices)

        self._assert_model_routing()
        self._assert_matching_indices(data_indices)
        self._assert_hidden_nodes_name(self._graph_name_hidden)

        # build networks
        self._build_networks(model_config.model, self._graph_data, dynamic_graph_config.edges)

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

    def _configure_dynamic_node_attributes(self, dynamic_node_config: DotDict) -> None:
        """Configure runtime node attributes for models that support them."""
        del dynamic_node_config

    def _hidden_coordinates(self) -> torch.Tensor:
        return self._graph_data[self._graph_name_hidden].x

    def _build_encoder_routing(self, encoders_config: DotDict) -> None:
        """Builds the dataset routing for encoders."""
        self.dataset2encoder: dict[str, str] = {}
        self.encoder2datasets: dict[str, list[str]] = {}
        self.encoder_fusing_strategy: dict[str, str] = {}
        for encoder_name, encoder_config in encoders_config.items():
            datasets_to_encode = list(encoder_config["source_datasets"])
            self.encoder2datasets[encoder_name] = datasets_to_encode
            for d in datasets_to_encode:
                assert d not in self.dataset2encoder, (
                    f"Dataset '{d}' is claimed by encoders '{self.dataset2encoder[d]}' and "
                    f"'{encoder_name}'. Each dataset must be encoded by exactly one encoder."
                )
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
        not_input_datasets = set(self.dataset2encoder) - set(self.input_dim)
        assert (
            not not_input_datasets
        ), f"Datasets {not_input_datasets} are referenced by encoders but missing from data_indices provided to the model. "

        not_target_datasets = set(self.dataset2decoder) - set(self.output_dim)
        assert (
            not not_target_datasets
        ), f"Datasets {not_target_datasets} are referenced by decoders but missing from data_indices provided to the model. "

        for encoder_name, fusing_strategy in self.encoder_fusing_strategy.items():
            if fusing_strategy not in SUPPORTED_FUSING_STRATEGIES:
                raise ValueError(
                    f"Encoder '{encoder_name}' has unsupported fusing strategy '{fusing_strategy}'. "
                    + f"Valid options are: {SUPPORTED_FUSING_STRATEGIES}"
                )

            source_datasets = self.encoder2datasets[encoder_name]
            if len(source_datasets) == 1:
                continue

            if fusing_strategy == "joint":
                # Gridded and tabular sources disagree on the destination index space: tabular
                # sources are encoded against a batch-repeated hidden node set, gridded ones
                # against the plain hidden nodes. They cannot share one joint encoder pass.
                static_flags = {d: self.is_dataset_static[d] for d in source_datasets}
                if len(set(static_flags.values())) > 1:
                    raise ValueError(
                        f"Encoder '{encoder_name}' fuses gridded and non-gridded datasets jointly "
                        f"({static_flags}), which is not supported because they use different "
                        "destination index spaces. Use dataset_fusing_strategy: 'sequential' instead."
                    )

                # Joint encoding of several sources needs the merged source index space and edge
                # fusion, which is not implemented yet. Fail here rather than encoding the sources
                # one by one, which would silently drop all but the last of them.
                raise NotImplementedError(
                    f"Encoder '{encoder_name}' requests dataset_fusing_strategy: 'joint' over "
                    f"{source_datasets}, which is not implemented yet. "
                    "Use 'sequential' to encode each source dataset with shared encoder weights."
                )

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
            + self.dynamic_node_attribute_dims.get(dataset_name, 0)
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

    def _resolve_in_out_sharded(self, batch: Batch) -> dict[str, bool]:
        """Per-dataset flag indicating whether the dataset is grid-sharded.

        Sharding metadata is carried by the batch itself (``batch.shard_sizes``),
        which the per-dataset source views expose via ``flatten().shard_sizes``.
        ``None`` means the corresponding dataset is replicated, not sharded.
        """
        return {dataset_name: batch.shard_sizes.get(dataset_name) is not None for dataset_name in batch.keys()}

    # Canonical gridded axis order: the integer passed to _get_consistent_dim indexes this tuple
    # TODO: Should this be defined here?
    _AXIS_BY_POSITION = ("batch", "time", "ensemble", "grid", "variables")

    def _get_consistent_dim(self, x: dict[str, Tensor | list[Tensor]], dim: int) -> int:
        """Return a logical dimension size that is consistent across all datasets.

        dim is a gridded physical position (e.g. 0==batch, 2==ensemble). It gets
        resolved to a logical axis name and looked up through each dataset's own TensorLayout

        - batch:
          for tabular datasets the outer list is the batch axis, i.e. len(data)
          for gridded datasets it is shape[layout.batch]
        - other axes (e.g. "ensemble") are read from the (per-sample) tensor at the layout
          position. Tabular observation datasets contribute an implicit singleton size==1.
        """
        axis_name = self._AXIS_BY_POSITION[dim]
        dim_sizes: list[int] = []
        for _x in x.values():
            layout = _x.layout

            if axis_name == "batch":
                if isinstance(_x.data, list):
                    # Tabular observations: the outer list is the batch axis
                    dim_sizes.append(len(_x.data))
                elif layout.batch is not None:
                    dim_sizes.append(_x.data.shape[layout.batch])
                continue

            axis_pos = getattr(layout, axis_name)
            if axis_pos is None:
                # Axis not materialised for this dataset -> implicit singleton.
                dim_sizes.append(1)
                continue

            if isinstance(_x.data, list):
                if len(_x.data) > 0:
                    dim_sizes.append(_x.data[0].shape[axis_pos])
            else:
                dim_sizes.append(_x.data.shape[axis_pos])

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
                for dataset_name in dataset_names:  # TODO: make this compatible with tabular
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
            # Sharding is carried by the batch so the model can read it off the views.
            forward_batch = Batch(data=x, shard_sizes=grid_shard_sizes or {})

            # Perform forward pass
            y_hat = self.forward(forward_batch, model_comm_group=model_comm_group, **kwargs)

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
