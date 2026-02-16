# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Optional

import torch
from hydra.utils import instantiate
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.bounding import build_boundings
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.models import AnemoiModelAutoEncoder
from anemoi.utils.config import DotDict


class AnemoiModelHierarchicalAutoEncoder(AnemoiModelAutoEncoder):
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

        nn.Module.__init__(self)
        self._graph_data = graph_data
        self.data_indices = data_indices
        self.statistics = statistics

        model_config = DotDict(model_config)
        self._graph_name_hidden = model_config.graph.hidden

        self.n_step_input = model_config.training.multistep_input
        self.n_step_output = model_config.training.multistep_output

        self.num_channels = model_config.model.num_channels

        # hidden_dims is the dimentionality of features at each depth
        self.hidden_dims = {hidden: self.num_channels * (2**i) for i, hidden in enumerate(self._graph_name_hidden)}

        # Unpack config for hierarchical graph
        self.level_process = model_config.model.enable_hierarchical_level_processing

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters, self._graph_data)

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        # build networks
        self._build_networks(model_config)

        # build residual connection
        self._build_residual(model_config.model.residual)

        # build boundings
        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        # Multi-dataset: create ModuleDict with ModuleList per dataset
        self.boundings = build_boundings(model_config, self.data_indices, self.statistics)

    def _calculate_input_dim_latent(self, dataset_name: str) -> int:
        return self.node_attributes.attr_ndims[self._graph_name_hidden[0]]

    def _build_networks(self, model_config):

        # note that this is called by the super class init
        # self.hidden_dims is the dimentionality of features at each depth
        self.hidden_dims = {hidden: self.num_channels * (2**i) for i, hidden in enumerate(self._graph_name_hidden)}
        self.num_hidden = len(self._graph_name_hidden)

        # Encoder data -> hidden
        self.encoder_graph_provider = nn.ModuleDict()
        self.encoder = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            self.encoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[(dataset_name, "to", self._graph_name_hidden[0])],
                edge_attributes=model_config.model.encoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes.num_nodes[dataset_name],
                dst_size=self.node_attributes.num_nodes[self._graph_name_hidden[0]],
                trainable_size=model_config.model.encoder.get("trainable_size", 0),
            )
            self.encoder[dataset_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.input_dim[dataset_name],
                in_channels_dst=self.input_dim_latent[dataset_name],
                hidden_dim=self.hidden_dims[self._graph_name_hidden[0]],
                edge_dim=self.encoder_graph_provider[dataset_name].edge_dim,
            )

        # Level processors
        if self.level_process:
            self.down_level_processor = nn.ModuleDict()
            self.down_level_processor_graph_providers = nn.ModuleDict()
            self.up_level_processor = nn.ModuleDict()
            self.up_level_processor_graph_providers = nn.ModuleDict()

            for i in range(0, self.num_hidden - 1):
                nodes_names = self._graph_name_hidden[i]

                # Create graph providers for down level processor
                self.down_level_processor_graph_providers[nodes_names] = create_graph_provider(
                    graph=self._graph_data[(nodes_names, "to", nodes_names)],
                    edge_attributes=model_config.model.processor.get("sub_graph_edge_attributes"),
                    src_size=self.node_attributes.num_nodes[nodes_names],
                    dst_size=self.node_attributes.num_nodes[nodes_names],
                    trainable_size=model_config.model.processor.get("trainable_size", 0),
                )

                self.down_level_processor[nodes_names] = instantiate(
                    model_config.model.processor,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    num_channels=self.hidden_dims[nodes_names],
                    edge_dim=self.down_level_processor_graph_providers[nodes_names].edge_dim,
                    num_layers=model_config.model.level_process_num_layers,
                )

                # Create graph providers for up level processor
                self.up_level_processor_graph_providers[nodes_names] = create_graph_provider(
                    graph=self._graph_data[(nodes_names, "to", nodes_names)],
                    edge_attributes=model_config.model.processor.get("sub_graph_edge_attributes"),
                    src_size=self.node_attributes.num_nodes[nodes_names],
                    dst_size=self.node_attributes.num_nodes[nodes_names],
                    trainable_size=model_config.model.processor.get("trainable_size", 0),
                )

                self.up_level_processor[nodes_names] = instantiate(
                    model_config.model.processor,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    num_channels=self.hidden_dims[nodes_names],
                    edge_dim=self.up_level_processor_graph_providers[nodes_names].edge_dim,
                    num_layers=model_config.model.level_process_num_layers,
                )

        # Downscale
        self.downscale = nn.ModuleDict()
        self.downscale_graph_providers = nn.ModuleDict()

        for i in range(0, self.num_hidden - 1):
            src_nodes_name = self._graph_name_hidden[i]
            dst_nodes_name = self._graph_name_hidden[i + 1]

            self.downscale_graph_providers[src_nodes_name] = create_graph_provider(
                graph=self._graph_data[(src_nodes_name, "to", dst_nodes_name)],
                edge_attributes=model_config.model.encoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes.num_nodes[src_nodes_name],
                dst_size=self.node_attributes.num_nodes[dst_nodes_name],
                trainable_size=model_config.model.encoder.get("trainable_size", 0),
            )

            self.downscale[src_nodes_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[src_nodes_name],
                in_channels_dst=self.node_attributes.attr_ndims[dst_nodes_name],
                hidden_dim=self.hidden_dims[dst_nodes_name],
                edge_dim=self.downscale_graph_providers[src_nodes_name].edge_dim,
            )

        # Upscale
        self.upscale = nn.ModuleDict()
        self.upscale_graph_providers = nn.ModuleDict()

        for i in range(1, self.num_hidden):
            src_nodes_name = self._graph_name_hidden[i]
            dst_nodes_name = self._graph_name_hidden[i - 1]

            self.upscale_graph_providers[src_nodes_name] = create_graph_provider(
                graph=self._graph_data[(src_nodes_name, "to", dst_nodes_name)],
                edge_attributes=model_config.model.decoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes.num_nodes[src_nodes_name],
                dst_size=self.node_attributes.num_nodes[dst_nodes_name],
                trainable_size=model_config.model.decoder.get("trainable_size", 0),
            )

            self.upscale[src_nodes_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[src_nodes_name],
                in_channels_dst=self.hidden_dims[dst_nodes_name],
                hidden_dim=self.hidden_dims[src_nodes_name],
                out_channels_dst=self.hidden_dims[dst_nodes_name],
                edge_dim=self.upscale_graph_providers[src_nodes_name].edge_dim,
            )

        # Decoder hidden -> data
        self.decoder_graph_provider = nn.ModuleDict()
        self.decoder = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            self.decoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[(self._graph_name_hidden[0], "to", dataset_name)],
                edge_attributes=model_config.model.decoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes.num_nodes[self._graph_name_hidden[0]],
                dst_size=self.node_attributes.num_nodes[dataset_name],
                trainable_size=model_config.model.decoder.get("trainable_size", 0),
            )

            self.decoder[dataset_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[self._graph_name_hidden[0]],
                in_channels_dst=self.target_dim[dataset_name],
                hidden_dim=self.hidden_dims[self._graph_name_hidden[0]],
                out_channels_dst=self.output_dim[dataset_name],
                edge_dim=self.decoder_graph_provider[dataset_name].edge_dim,
            )

    def forward(
        self,
        x: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, Optional[list]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        x : dict[str, Tensor]
            Input data
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None
        grid_shard_shapes : list, optional
            Shard shapes of the grid, by default None

        Returns
        -------
        dict[str, Tensor]
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        dataset_names = list(x.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        # Get all trainable parameters for the hidden layers -> initialisation of each hidden, which becomes trainable bias
        x_hidden_latents = {}
        for hidden in self._graph_name_hidden:
            x_hidden_latents[hidden] = self.node_attributes(hidden, batch_size=batch_size)

        # Get data and hidden shapes for sharding
        shard_shapes_hidden_dict = {}
        for hidden, x_latent in x_hidden_latents.items():
            shard_shapes_hidden_dict[hidden] = get_shard_shapes(x_latent, 0, model_comm_group=model_comm_group)

        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_data_latent_dict = {}
        shard_shapes_data_dict = {}
        x_encoded_latents_dict = {}

        for dataset_name in dataset_names:
            x_data_latent, shard_shapes_data = self._assemble_input(
                x[dataset_name],
                batch_size=batch_size,
                grid_shard_shapes=grid_shard_shapes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            shard_shapes_data_dict[dataset_name] = shard_shapes_data

            # Compute encoder edges at model level
            encoder_edge_attr, encoder_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )

            # Encoder for this dataset
            x_data_latent, x_latent = self.encoder[dataset_name](
                (x_data_latent, x_hidden_latents[self._graph_name_hidden[0]]),
                batch_size=batch_size,
                shard_shapes=(
                    shard_shapes_data_dict[dataset_name],
                    shard_shapes_hidden_dict[self._graph_name_hidden[0]],
                ),
                edge_attr=encoder_edge_attr,
                edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
                x_dst_is_sharded=False,  # x_latent does not come sharded
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
                edge_shard_shapes=enc_edge_shard_shapes,
            )
            x_data_latent_dict[dataset_name] = x_data_latent

            x_encoded_latents_dict[dataset_name] = {}

            ## Downscale
            for i in range(0, self.num_hidden - 1):
                src_hidden_name = self._graph_hidden_names[i]
                dst_hidden_name = self._graph_hidden_names[i + 1]

                ## Processing at same level
                if self.level_process:
                    # Compute edges for down level processor
                    (
                        down_level_edge_attr,
                        down_level_edge_index,
                        down_edge_shard_shapes,
                    ) = self.down_level_processor_graph_providers[src_hidden_name].get_edges(
                        batch_size=batch_size,
                        model_comm_group=model_comm_group,
                    )

                    x_latent = self.down_level_processor[src_hidden_name](
                        x_latent,
                        batch_size=batch_size,
                        shard_shapes=shard_shapes_hidden_dict[src_hidden_name],
                        edge_attr=down_level_edge_attr,
                        edge_index=down_level_edge_index,
                        model_comm_group=model_comm_group,
                        edge_shard_shapes=down_edge_shard_shapes,
                    )

                # Compute edges for downscale mapper
                downscale_edge_attr, downscale_edge_index, ds_edge_shard_shapes = self.downscale_graph_providers[
                    src_hidden_name
                ].get_edges(
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                )

                # Encode to next hidden level
                x_encoded_latents_dict[dataset_name][src_hidden_name], x_latent = self.downscale[src_hidden_name](
                    (x_latent, x_hidden_latents[dst_hidden_name]),
                    batch_size=batch_size,
                    shard_shapes=(shard_shapes_hidden_dict[src_hidden_name], shard_shapes_hidden_dict[dst_hidden_name]),
                    edge_attr=downscale_edge_attr,
                    edge_index=downscale_edge_index,
                    model_comm_group=model_comm_group,
                    x_src_is_sharded=True,
                    x_dst_is_sharded=False,  # x_latent does not come sharded
                    keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
                    edge_shard_shapes=ds_edge_shard_shapes,
                )

            dataset_latents[dataset_name] = x_latent

        # Combine all dataset latents in the innermost layer
        x_latent = sum(dataset_latents.values())

        # Decoder
        x_out_dict = {}
        for dataset_name in dataset_names:
            ## Upscale
            for i in range(self.num_hidden - 1, 0, -1):
                src_hidden_name = self._graph_name_hidden[i]
                dst_hidden_name = self._graph_name_hidden[i - 1]

                # Compute edges for upscale mapper
                upscale_edge_attr, upscale_edge_index, us_edge_shard_shapes = self.upscale_graph_providers[
                    src_hidden_name
                ].get_edges(
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                )

                # Decode to next level
                x_latent = self.upscale[src_hidden_name](
                    (x_latent, x_encoded_latents_dict[dataset_name][dst_hidden_name]),
                    batch_size=batch_size,
                    shard_shapes=(shard_shapes_hidden_dict[src_hidden_name], shard_shapes_hidden_dict[dst_hidden_name]),
                    edge_attr=upscale_edge_attr,
                    edge_index=upscale_edge_index,
                    model_comm_group=model_comm_group,
                    x_src_is_sharded=in_out_sharded,
                    x_dst_is_sharded=in_out_sharded,
                    keep_x_dst_sharded=in_out_sharded,
                    edge_shard_shapes=us_edge_shard_shapes,
                )

                # Processing at same level
                if self.level_process:
                    # Compute edges for up level processor
                    (
                        up_level_edge_attr,
                        up_level_edge_index,
                        up_edge_shard_shapes,
                    ) = self.up_level_processor_graph_providers[dst_hidden_name].get_edges(
                        batch_size=batch_size,
                        model_comm_group=model_comm_group,
                    )

                    x_latent = self.up_level_processor[dst_hidden_name](
                        x_latent,
                        edge_attr=up_level_edge_attr,
                        edge_index=up_level_edge_index,
                        batch_size=batch_size,
                        shard_shapes=shard_shapes_hidden_dict[dst_hidden_name],
                        model_comm_group=model_comm_group,
                        edge_shard_shapes=up_edge_shard_shapes,
                    )

            # Do not pass x_data_latent to the decoder
            # In autoencoder training this would cause the model to discard everything else and just keep the values they were before
            # Only pass data and forcing coordinates to the decoder
            x_target_latent, shard_shapes_target = self._assemble_forcings(
                x[dataset_name], batch_size, grid_shard_shapes, model_comm_group, dataset_name
            )

            # Compute decoder edges
            decoder_edge_attr, decoder_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )

            x_out = self.decoder[dataset_name](
                (x_latent, x_target_latent),
                batch_size=batch_size,
                shard_shapes=(
                    shard_shapes_hidden_dict[self._graph_name_hidden[0]],
                    shard_shapes_target,
                ),
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,  # x_latent always comes sharded
                x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
                keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
                edge_shard_shapes=dec_edge_shard_shapes,
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out,
                batch_size,
                ensemble_size,
                x[dataset_name].dtype,
                dataset_name,
            )

        return x_out_dict
