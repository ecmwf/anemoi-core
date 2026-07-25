# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional

import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.models import AnemoiModelAutoEncoder
from anemoi.models.models.base import BaseGraphModel


class AnemoiModelHierarchicalAutoEncoder(AnemoiModelAutoEncoder):
    """Hierarchical auto-encoder (no deep processor).

    All components (encoder, downscale/upscale mappers, optional per-level processors and
    the decoder, with their graph providers) are built by a ``ModelBuilder`` and injected;
    this class stores them and derives ``hidden_dims`` from ``num_channels``. It uses the
    container base directly (there is no single main processor to inject).
    """

    def __init__(
        self,
        *,
        encoder: nn.ModuleDict,
        encoder_graph_provider: nn.ModuleDict,
        decoder: nn.ModuleDict,
        decoder_graph_provider: nn.ModuleDict,
        downscale: nn.ModuleDict,
        downscale_graph_providers: nn.ModuleDict,
        upscale: nn.ModuleDict,
        upscale_graph_providers: nn.ModuleDict,
        level_process: bool,
        down_level_processor: nn.ModuleDict | None = None,
        down_level_processor_graph_providers: nn.ModuleDict | None = None,
        up_level_processor: nn.ModuleDict | None = None,
        up_level_processor_graph_providers: nn.ModuleDict | None = None,
        **base_kwargs,
    ) -> None:
        # Skip the enc-proc-dec container (it requires a main processor which this model
        # does not have) and use the shared base container directly.
        BaseGraphModel.__init__(self, **base_kwargs)
        self.hidden_dims = {hidden: self.num_channels * (2**i) for i, hidden in enumerate(self._graph_name_hidden)}
        self.num_hidden = len(self._graph_name_hidden)

        self.encoder_graph_provider = encoder_graph_provider
        self.encoder = encoder

        self.level_process = level_process
        if level_process:
            self.down_level_processor = down_level_processor
            self.down_level_processor_graph_providers = down_level_processor_graph_providers
            self.up_level_processor = up_level_processor
            self.up_level_processor_graph_providers = up_level_processor_graph_providers

        self.downscale = downscale
        self.downscale_graph_providers = downscale_graph_providers
        self.upscale = upscale
        self.upscale_graph_providers = upscale_graph_providers

        self.decoder_graph_provider = decoder_graph_provider
        self.decoder = decoder

    def forward(
        self,
        x: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        x : dict[str, Tensor]
            Input data
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None
        grid_shard_sizes : DatasetShardSizes, optional
            Per-dataset shard sizes for the grid dimension. ``None`` means the
            corresponding dataset is replicated, not sharded.

        Returns
        -------
        dict[str, Tensor]
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        dataset_names = list(x.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = self._resolve_in_out_sharded(
            dataset_names=dataset_names,
            grid_shard_sizes=grid_shard_sizes,
        )
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        # Get all trainable parameters for the hidden layers -> initialisation of each hidden, which becomes trainable bias
        x_hidden_latents = {}
        for hidden in self._graph_name_hidden:
            x_hidden_latents[hidden] = self.node_attributes(hidden, batch_size=batch_size)

        # Get data and hidden shapes for sharding
        shard_sizes_hidden_dict = {}
        for hidden, x_latent in x_hidden_latents.items():
            shard_sizes_hidden_dict[hidden] = get_shard_sizes(x_latent, 0, model_comm_group=model_comm_group)
            x_hidden_latents[hidden] = shard_tensor(x_latent, 0, shard_sizes_hidden_dict[hidden], model_comm_group)

        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_data_latent_dict = {}
        shard_sizes_data_dict = {}
        x_encoded_latents_dict: dict[str, dict[str, torch.Tensor]] = {}

        for dataset_name in dataset_names:
            x_data_latent, shard_sizes_data = self._assemble_input(
                x[dataset_name],
                batch_size=batch_size,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            shard_sizes_data_dict[dataset_name] = shard_sizes_data

            # Compute encoder edges at model level
            (
                encoder_edge_attr,
                encoder_edge_index,
                enc_edge_shard_sizes,
            ) = self.encoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )

            enc_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_data_dict[dataset_name],  # None if not sharded
                dst_nodes=shard_sizes_hidden_dict[self._graph_name_hidden[0]],
                edges=enc_edge_shard_sizes,
            )

            # Encoder for this dataset
            x_data_latent, x_latent = self.encoder[dataset_name](
                (x_data_latent, x_hidden_latents[self._graph_name_hidden[0]]),
                batch_size=batch_size,
                shard_info=enc_shard_info,
                edge_attr=encoder_edge_attr,
                edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
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
                        down_edge_shard_sizes,
                    ) = self.down_level_processor_graph_providers[src_hidden_name].get_edges(
                        batch_size=batch_size,
                        model_comm_group=model_comm_group,
                    )

                    x_latent = self.down_level_processor[src_hidden_name](
                        x_latent,
                        batch_size=batch_size,
                        shard_info=GraphShardInfo(
                            nodes=shard_sizes_hidden_dict[src_hidden_name],
                            edges=down_edge_shard_sizes,
                        ),
                        edge_attr=down_level_edge_attr,
                        edge_index=down_level_edge_index,
                        model_comm_group=model_comm_group,
                    )

                # Compute edges for downscale mapper
                (
                    downscale_edge_attr,
                    downscale_edge_index,
                    ds_edge_shard_sizes,
                ) = self.downscale_graph_providers[src_hidden_name].get_edges(
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                )

                ds_shard_info = BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_hidden_dict[src_hidden_name],
                    dst_nodes=shard_sizes_hidden_dict[dst_hidden_name],
                    edges=ds_edge_shard_sizes,
                )

                # Encode to next hidden level
                x_encoded_latents_dict[dataset_name][src_hidden_name], x_latent = self.downscale[src_hidden_name](
                    (x_latent, x_hidden_latents[dst_hidden_name]),
                    batch_size=batch_size,
                    shard_info=ds_shard_info,
                    edge_attr=downscale_edge_attr,
                    edge_index=downscale_edge_index,
                    model_comm_group=model_comm_group,
                    keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
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
                (
                    upscale_edge_attr,
                    upscale_edge_index,
                    us_edge_shard_sizes,
                ) = self.upscale_graph_providers[src_hidden_name].get_edges(
                    batch_size=batch_size,
                    model_comm_group=model_comm_group,
                )

                us_shard_info = BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_hidden_dict[src_hidden_name],
                    dst_nodes=shard_sizes_hidden_dict[dst_hidden_name],
                    edges=us_edge_shard_sizes,
                )

                # Decode to next level
                x_latent = self.upscale[src_hidden_name](
                    (x_latent, x_encoded_latents_dict[dataset_name][dst_hidden_name]),
                    batch_size=batch_size,
                    shard_info=us_shard_info,
                    edge_attr=upscale_edge_attr,
                    edge_index=upscale_edge_index,
                    model_comm_group=model_comm_group,
                    keep_x_dst_sharded=True,
                )

                # Processing at same level
                if self.level_process:
                    # Compute edges for up level processor
                    (
                        up_level_edge_attr,
                        up_level_edge_index,
                        up_edge_shard_sizes,
                    ) = self.up_level_processor_graph_providers[dst_hidden_name].get_edges(
                        batch_size=batch_size,
                        model_comm_group=model_comm_group,
                    )

                    x_latent = self.up_level_processor[dst_hidden_name](
                        x_latent,
                        edge_attr=up_level_edge_attr,
                        edge_index=up_level_edge_index,
                        batch_size=batch_size,
                        shard_info=GraphShardInfo(
                            nodes=shard_sizes_hidden_dict[dst_hidden_name],
                            edges=up_edge_shard_sizes,
                        ),
                        model_comm_group=model_comm_group,
                    )

            # Do not pass x_data_latent to the decoder
            # In autoencoder training this would cause the model to discard everything else and just keep the values they were before
            # Only pass data and forcing coordinates to the decoder
            x_target_latent, shard_sizes_target = self._assemble_forcings(
                x[dataset_name], batch_size, grid_shard_sizes, model_comm_group, dataset_name
            )

            # Compute decoder edges
            (
                decoder_edge_attr,
                decoder_edge_index,
                dec_edge_shard_sizes,
            ) = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden_dict[self._graph_name_hidden[0]],
                dst_nodes=shard_sizes_target,  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            x_out = self.decoder[dataset_name](
                (x_latent, x_target_latent),
                batch_size=batch_size,
                shard_info=dec_shard_info,
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],  # keep x_out sharded iff in_out_sharded
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out,
                batch_size,
                ensemble_size,
                x[dataset_name].dtype,
                dataset_name,
            )

        return x_out_dict
