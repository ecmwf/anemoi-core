# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import torch
from hydra.utils import instantiate
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models import AnemoiModelEncProcDec

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDecHierarchical(AnemoiModelEncProcDec):
    """Message passing hierarchical graph neural network."""

    def _calculate_input_dim_latent(self):
        return self.node_attributes.attr_ndims[self._graph_name_hidden[0]]

    def _build_networks(self, model_config):
        """Builds the model components."""
        # note that this is called by the super class init
        # self.hidden_dims is the dimentionality of features at each depth
        self.hidden_dims = {hidden: self.num_channels * (2**i) for i, hidden in enumerate(self._graph_name_hidden)}
        self.num_hidden = len(self._graph_name_hidden)

        # Encoder data -> hidden
        self.encoder = torch.nn.ModuleDict()
        for dataset_name in self._graph_data.keys():
            self.encoder[dataset_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.input_dim[dataset_name],
                in_channels_dst=self.input_dim_latent[dataset_name],
                hidden_dim=self.hidden_dims[self._graph_name_hidden[0]],
                sub_graph=self._graph_data[dataset_name][(self._graph_name_data, "to", self._graph_name_hidden[0])],
                src_grid_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_data],
                dst_grid_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_hidden[0]],
            )

        # Processor hidden -> hidden (shared across all datasets)
        first_dataset_name = next(iter(self._graph_data.keys()))

        # Level processors
        self.level_process = model_config.model.enable_hierarchical_level_processing
        if self.level_process:
            self.down_level_processor = nn.ModuleDict()
            self.up_level_processor = nn.ModuleDict()

            for i in range(0, self.num_hidden - 1):
                nodes_names = self._graph_name_hidden[i]

                self.down_level_processor[nodes_names] = instantiate(
                    model_config.model.processor,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    num_channels=self.hidden_dims[nodes_names],
                    sub_graph=self._graph_data[first_dataset_name][(nodes_names, "to", nodes_names)],
                    src_grid_size=self.node_attributes[first_dataset_name].num_nodes[nodes_names],
                    dst_grid_size=self.node_attributes[first_dataset_name].num_nodes[nodes_names],
                    num_layers=model_config.model.level_process_num_layers,
                )

                self.up_level_processor[nodes_names] = instantiate(
                    model_config.model.processor,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    num_channels=self.hidden_dims[nodes_names],
                    sub_graph=self._graph_data[first_dataset_name][(nodes_names, "to", nodes_names)],
                    src_grid_size=self.node_attributes[first_dataset_name].num_nodes[nodes_names],
                    dst_grid_size=self.node_attributes[first_dataset_name].num_nodes[nodes_names],
                    num_layers=model_config.model.level_process_num_layers,
                )

        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.hidden_dims[self._graph_name_hidden[self.num_hidden - 1]],
            sub_graph=self._graph_data[first_dataset_name][
                (self._graph_name_hidden[0], "to", self._graph_name_hidden[0])
            ],
            src_grid_size=self.node_attributes[first_dataset_name].num_nodes[self._graph_name_hidden[0]],
            dst_grid_size=self.node_attributes[first_dataset_name].num_nodes[self._graph_name_hidden[0]],
        )

        # Downscale
        self.downscale = nn.ModuleDict()

        for i in range(0, self.num_hidden - 1):
            src_nodes_name = self._graph_name_hidden[i]
            dst_nodes_name = self._graph_name_hidden[i + 1]

            self.downscale[src_nodes_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[src_nodes_name],
                in_channels_dst=self.node_attributes[first_dataset_name].attr_ndims[dst_nodes_name],
                hidden_dim=self.hidden_dims[dst_nodes_name],
                sub_graph=self._graph_data[first_dataset_name][(src_nodes_name, "to", dst_nodes_name)],
                src_grid_size=self.node_attributes[first_dataset_name].num_nodes[src_nodes_name],
                dst_grid_size=self.node_attributes[first_dataset_name].num_nodes[dst_nodes_name],
            )

        # Upscale
        self.upscale = nn.ModuleDict()

        for i in range(1, self.num_hidden):
            src_nodes_name = self._graph_name_hidden[i]
            dst_nodes_name = self._graph_name_hidden[i - 1]

            self.upscale[src_nodes_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[src_nodes_name],
                in_channels_dst=self.hidden_dims[dst_nodes_name],
                hidden_dim=self.hidden_dims[src_nodes_name],
                out_channels_dst=self.hidden_dims[dst_nodes_name],
                sub_graph=self._graph_data[first_dataset_name][(src_nodes_name, "to", dst_nodes_name)],
                src_grid_size=self.node_attributes[first_dataset_name].num_nodes[src_nodes_name],
                dst_grid_size=self.node_attributes[first_dataset_name].num_nodes[dst_nodes_name],
            )

        # Decoder hidden -> data
        self.decoder = torch.nn.ModuleDict()
        for dataset_name in self._graph_data.keys():
            self.decoder[dataset_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.hidden_dims[self._graph_name_hidden[0]],
                in_channels_dst=self.input_dim[dataset_name],
                hidden_dim=self.hidden_dims[self._graph_name_hidden[0]],
                out_channels_dst=self.num_output_channels[dataset_name],
                sub_graph=self._graph_data[dataset_name][(self._graph_name_hidden[0], "to", self._graph_name_data)],
                src_grid_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_hidden[0]],
                dst_grid_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_data],
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
        x : Tensor
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

        # Extract and validate batch sizes across datasets
        batch_sizes = [x[dataset_name].shape[0] for dataset_name in dataset_names]
        ensemble_sizes = [x[dataset_name].shape[2] for dataset_name in dataset_names]

        # Assert all datasets have the same batch and ensemble sizes
        assert all(
            bs == batch_sizes[0] for bs in batch_sizes
        ), f"Batch sizes must be the same across datasets: {batch_sizes}"
        assert all(
            es == ensemble_sizes[0] for es in ensemble_sizes
        ), f"Ensemble sizes must be the same across datasets: {ensemble_sizes}"

        batch_size = batch_sizes[0]
        ensemble_size = ensemble_sizes[0]
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        # Get all trainable parameters for the hidden layers -> initialisation of each hidden, which becomes trainable bias
        x_hidden_latents = {}
        for hidden in self._graph_name_hidden:
            x_hidden_latents[hidden] = self.node_attributes[dataset_names[0]](hidden, batch_size=batch_size)

        # Get data and hidden shapes for sharding
        shard_shapes_hidden_dict = {}
        for hidden, x_latent in x_hidden_latents.items():
            shard_shapes_hidden_dict[hidden] = get_shard_shapes(x_latent, 0, model_comm_group=model_comm_group)

        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_shapes_data_dict = {}

        for dataset_name in dataset_names:
            x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
                x[dataset_name],
                batch_size=batch_size,
                grid_shard_shapes=grid_shard_shapes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            x_skip_dict[dataset_name] = x_skip
            x_data_latent_dict[dataset_name] = x_data_latent
            shard_shapes_data_dict[dataset_name] = shard_shapes_data

            # Encoder for this dataset
            x_data_latent, x_latent = self.encoder[dataset_name](
                (x_data_latent, x_hidden_latents[self._graph_name_hidden[0]]),
                batch_size=batch_size,
                shard_shapes=(
                    shard_shapes_data_dict[dataset_name],
                    shard_shapes_hidden_dict[self._graph_name_hidden[0]],
                ),
                model_comm_group=model_comm_group,
                x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
                x_dst_is_sharded=False,  # x_latent does not come sharded
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
            )
            dataset_latents[dataset_name] = x_latent

        # Combine all dataset latents
        x_latent = sum(dataset_latents.values())

        x_encoded_latents = {}
        skip_connections = {}

        ## Downscale
        for i in range(0, self.num_hidden - 1):
            src_hidden_name = self._graph_name_hidden[i]
            dst_hidden_name = self._graph_name_hidden[i + 1]

            # Processing at same level
            if self.level_process:
                x_latent = self.down_level_processor[src_hidden_name](
                    x_latent,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes_hidden_dict[src_hidden_name],
                    model_comm_group=model_comm_group,
                )

            # store latents for skip connections
            skip_connections[src_hidden_name] = x_latent

            # Encode to next hidden level
            x_encoded_latents[src_hidden_name], x_latent = self.downscale[src_hidden_name](
                (x_latent, x_hidden_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden_dict[src_hidden_name], shard_shapes_hidden_dict[dst_hidden_name]),
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=False,  # x_latent does not come sharded
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
            )

        # Processing hidden-most level
        x_latent = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden_dict[self._graph_hidden_names[-1]],
            model_comm_group=model_comm_group,
        )

        ## Upscale
        for i in range(self.num_hidden - 1, 0, -1):
            src_hidden_name = self._graph_name_hidden[i]
            dst_hidden_name = self._graph_name_hidden[i - 1]

            # Decode to next level
            x_latent = self.upscale[src_hidden_name](
                (x_latent, x_encoded_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden_dict[src_hidden_name], shard_shapes_hidden_dict[dst_hidden_name]),
                model_comm_group=model_comm_group,
                x_src_is_sharded=in_out_sharded,
                x_dst_is_sharded=in_out_sharded,
                keep_x_dst_sharded=in_out_sharded,
            )

            # Add skip connections
            x_latent = x_latent + skip_connections[dst_hidden_name]

            # Processing at same level
            if self.level_process:
                x_latent = self.up_level_processor[dst_hidden_name](
                    x_latent,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes_hidden_dict[dst_hidden_name],
                    model_comm_group=model_comm_group,
                )

        # Run decoder
        x_out_dict = {}
        for dataset_name in dataset_names:
            x_out = self.decoder[dataset_name](
                (x_latent, x_data_latent_dict[dataset_name]),
                batch_size=batch_size,
                shard_shapes=(
                    shard_shapes_hidden_dict[self._graph_name_hidden[0]],
                    shard_shapes_data_dict[dataset_name],
                ),
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,  # x_latent always comes sharded
                x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
                keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out,
                x_skip_dict[dataset_name],
                batch_size,
                ensemble_size,
                x[dataset_name].dtype,
                dataset_name,
            )

        return x_out_dict
