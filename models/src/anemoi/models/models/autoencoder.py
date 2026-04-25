# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import einops
import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec

LOGGER = logging.getLogger(__name__)


class AnemoiModelAutoEncoder(AnemoiModelEncProcDec):
    """AutoEncoder"""

    def _calculate_target_dim(self, dataset_name: str) -> int:
        return (
            self.n_step_output * self.num_input_channels_decoding_forcings[dataset_name]
            + self.node_attributes.attr_ndims[dataset_name]
        )

    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None, dataset_name=None):
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes(dataset_name, batch_size=batch_size)
        grid_shard_shapes = grid_shard_shapes[dataset_name] if grid_shard_shapes is not None else None
        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        x_input = x[:, : self.n_step_input, ...]
        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x_input, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
        )

        return x_data_latent, shard_shapes_data

    def _assemble_output(self, x_out, batch_size, ensemble_size, dtype, dataset_name=None):
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) (time vars) -> batch time ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
                time=self.n_step_output,
            )
            .to(dtype=dtype)
            .clone()
        )
        assert dataset_name is not None, "dataset_name must be provided for multi-dataset case"

        for bounding in self.boundings[dataset_name]:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def _assemble_forcings(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None, dataset_name=None):
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_target = self.node_attributes(dataset_name, batch_size=batch_size)
        grid_shard_shapes = grid_shard_shapes[dataset_name] if grid_shard_shapes is not None else None
        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_target, 0, grid_shard_shapes, model_comm_group
            )
            node_attributes_target = shard_tensor(node_attributes_target, 0, shard_shapes_nodes, model_comm_group)

        x_forcing = x[:, : self.n_step_output, ...]
        # normalize and add data positional info (lat/lon)
        x_target_latent = torch.cat(
            (
                einops.rearrange(
                    x_forcing[..., self._decoding_forcing_input_idx[dataset_name]],
                    "batch time ensemble grid vars -> (batch ensemble grid) (time vars)",
                ),
                node_attributes_target,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_target = get_or_apply_shard_shapes(x_target_latent, 0, grid_shard_shapes, model_comm_group)
        return x_target_latent, shard_shapes_target

    def forward(
        self,
        x: Tensor,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, Optional[list]] | None = None,
        **kwargs,
    ) -> Tensor:
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
        Tensor
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """

        dataset_names = list(x.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = self._resolve_in_out_sharded(
            dataset_names=dataset_names,
            grid_shard_shapes=grid_shard_shapes,
        )
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        shard_shapes_data_dict = {}

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)
        for dataset_name in dataset_names:
            x_data_latent, shard_shapes_data = self._assemble_input(
                x[dataset_name], batch_size, grid_shard_shapes, model_comm_group, dataset_name
            )
            shard_shapes_data_dict[dataset_name] = shard_shapes_data

            encoder_edge_attr, encoder_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )
            # Encoder for this dataset
            x_data_latent, x_latent = self.encoder[dataset_name](
                (x_data_latent, x_hidden_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_data, shard_shapes_hidden),
                edge_attr=encoder_edge_attr,
                edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=in_out_sharded[dataset_name],  # x_data_latent comes sharded iff in_out_sharded
                x_dst_is_sharded=False,  # x_latent does not come sharded
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
                edge_shard_shapes=enc_edge_shard_shapes,
            )

            dataset_latents[dataset_name] = x_latent

        # Combine all dataset latents
        x_latent = sum(dataset_latents.values())

        # Decoder
        x_out_dict = {}
        for dataset_name in dataset_names:

            # Do not pass x_data_latent to the decoder
            # In autoencoder training this would cause the model to discard everything else and just keep the values they were before
            # Only pass data and forcing coordinates to the decoder
            x_target_latent, shard_shapes_target = self._assemble_forcings(
                x[dataset_name], batch_size, grid_shard_shapes, model_comm_group, dataset_name
            )

            # Compute decoder edges using updated latent representation
            decoder_edge_attr, decoder_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[
                dataset_name
            ].get_edges(batch_size=batch_size, model_comm_group=model_comm_group)

            x_out = self.decoder[dataset_name](
                (x_latent, x_target_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_target),
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,  # x_latent always comes sharded
                x_dst_is_sharded=in_out_sharded[dataset_name],  # x_data_latent comes sharded iff in_out_sharded
                keep_x_dst_sharded=in_out_sharded[dataset_name],  # keep x_out sharded iff in_out_sharded
                edge_shard_shapes=dec_edge_shard_shapes,
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out, batch_size, ensemble_size, x[dataset_name].dtype, dataset_name
            )

        return x_out_dict
