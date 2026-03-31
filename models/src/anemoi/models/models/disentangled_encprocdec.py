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
from hydra.utils import instantiate
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.utils.config import DotDict

from .autoencoder import AnemoiModelAutoEncoder

LOGGER = logging.getLogger(__name__)


class AnemoiModelDisentangledEncProcDec(AnemoiModelAutoEncoder):
    """Disentangled graph network with multi-dataset support.

    Encodes each input timestep independently for each dataset, accumulates all
    latent representations, blends them with a learned latent_blender, then runs
    a processor in latent space. On rollout step > 0, performs latent rollout by
    shifting the latent buffer instead of re-encoding from data space.
    """

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
    ) -> None:
        model_config = DotDict(model_config)
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )
        self.latent_rollout = model_config.model.model.latent_rollout

    def _calculate_input_dim(self, dataset_name: str) -> int:
        # Encode one timestep at a time — no n_step_input multiplier
        return self.num_input_channels[dataset_name] + self.node_attributes.attr_ndims[dataset_name]

    def _calculate_target_dim(self, dataset_name: str) -> int:
        return (
            self.n_step_output * self.num_input_channels_decoding_forcings[dataset_name]
            + self.node_attributes.attr_ndims[dataset_name]
        )

    def _build_networks(self, model_config: DotDict) -> None:
        # Build encoder, processor, decoder (and their graph providers) via parent
        super()._build_networks(model_config)

        # Latent blender: maps accumulated latents from all datasets
        # (n_datasets * n_step_input * num_channels) -> hidden
        # Uses the same hidden->hidden edges as the processor graph provider
        self.latent_blender = instantiate(
            model_config.model.encoder,
            _recursive_=False,
            in_channels_src=self.num_channels * self.n_step_input * len(self.dataset_names),
            in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
            hidden_dim=self.num_channels,
            edge_dim=self.processor_graph_provider.edge_dim,
        )

    def _assemble_single_input(
        self,
        x_t: Tensor,
        batch_size: int,
        grid_shard_shapes: dict | None,
        model_comm_group,
        dataset_name: str,
    ) -> tuple[Tensor, list]:
        """Assemble input for a single timestep (no n_step_input slicing)."""
        node_attributes_data = self.node_attributes(dataset_name, batch_size=batch_size)
        ds_grid_shard_shapes = grid_shard_shapes[dataset_name] if grid_shard_shapes is not None else None
        if ds_grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=ds_grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        x_data_latent = torch.cat(
            (
                einops.rearrange(x_t, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,
        )
        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=ds_grid_shard_shapes, model_comm_group=model_comm_group
        )
        return x_data_latent, shard_shapes_data

    def forward(
        self,
        x: dict[str, Tensor],
        rollout_step: Optional[int] = 0,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : dict[str, Tensor]
            Input data keyed by dataset name.
        rollout_step : int, optional
            Current rollout step. On 0, encode all input timesteps for all datasets.
            On > 0, do latent rollout.
        model_comm_group : ProcessGroup, optional
        grid_shard_shapes : dict, optional
        """
        batch_size = x[self.dataset_names[0]].shape[0]
        ensemble_size = x[self.dataset_names[0]].shape[2]

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

        proc_edge_attr, proc_edge_index, proc_edge_shard_shapes = self.processor_graph_provider.get_edges(
            batch_size=batch_size, model_comm_group=model_comm_group
        )

        if rollout_step == 0:
            # Encode each timestep of each dataset independently and accumulate latents
            latents = []
            for dataset_name in self.dataset_names:
                x_ds = x[dataset_name]
                in_out_sharded = grid_shard_shapes is not None and dataset_name in grid_shard_shapes
                self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

                enc_edge_attr, enc_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[
                    dataset_name
                ].get_edges(batch_size=batch_size, model_comm_group=model_comm_group)

                for i in range(self.n_step_input):
                    x_t = x_ds[:, i : i + 1, ...]  # [B, 1, E, G, D]
                    x_data_latent, shard_shapes_data = self._assemble_single_input(
                        x_t, batch_size, grid_shard_shapes, model_comm_group, dataset_name
                    )
                    _, x_latent = self.encoder[dataset_name](
                        (x_data_latent, x_hidden_latent),
                        batch_size=batch_size,
                        shard_shapes=(shard_shapes_data, shard_shapes_hidden),
                        edge_attr=enc_edge_attr,
                        edge_index=enc_edge_index,
                        model_comm_group=model_comm_group,
                        x_src_is_sharded=in_out_sharded,
                        x_dst_is_sharded=False,
                        keep_x_dst_sharded=True,
                        edge_shard_shapes=enc_edge_shard_shapes,
                    )
                    latents.append(x_latent)

            # Stack: [B*E*G_hidden, n_datasets * n_step_input, num_channels]
            self.x_accum = torch.stack(latents, dim=1)
            x_skip = latents[-1]  # last latent (last timestep of last dataset) for residual

        else:
            # Latent rollout: shift buffer and replace last slot with previous processor output
            self.x_accum = self.x_accum.roll(-1, dims=1)
            self.x_accum[:, -1, ...] = self.x_latent_proc
            x_skip = self.x_latent_proc

        # Flatten accumulated latents for blender input:
        # [B*E*G, n_datasets * n_step_input * num_channels]
        x_accum_flat = einops.rearrange(self.x_accum, "bg n c -> bg (n c)")

        # Latent blending: blend accumulated latents into a single hidden representation
        # Uses same hidden->hidden edges as the processor
        _, blended_x = self.latent_blender(
            (x_accum_flat, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_hidden),
            edge_attr=proc_edge_attr,
            edge_index=proc_edge_index,
            model_comm_group=model_comm_group,
            x_src_is_sharded=False,
            x_dst_is_sharded=False,
            keep_x_dst_sharded=True,
            edge_shard_shapes=proc_edge_shard_shapes,
        )

        # Processor: hidden -> hidden
        x_latent_proc = self.processor(
            x=blended_x,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            edge_attr=proc_edge_attr,
            edge_index=proc_edge_index,
            model_comm_group=model_comm_group,
            edge_shard_shapes=proc_edge_shard_shapes,
        )

        # Latent skip connection
        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_skip
        self.x_latent_proc = x_latent_proc

        # Decode each dataset independently
        outputs = {}
        for dataset_name in self.dataset_names:
            x_ds = x[dataset_name]
            in_out_sharded = grid_shard_shapes is not None and dataset_name in grid_shard_shapes

            dec_edge_attr, dec_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[
                dataset_name
            ].get_edges(batch_size=batch_size, model_comm_group=model_comm_group)

            x_target_latent, shard_shapes_target = self._assemble_forcings(
                x_ds, batch_size, grid_shard_shapes, model_comm_group, dataset_name=dataset_name
            )

            x_out = self.decoder[dataset_name](
                (x_latent_proc, x_target_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_target),
                edge_attr=dec_edge_attr,
                edge_index=dec_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=in_out_sharded,
                keep_x_dst_sharded=in_out_sharded,
                edge_shard_shapes=dec_edge_shard_shapes,
            )

            outputs[dataset_name] = self._assemble_output(
                x_out, batch_size, ensemble_size, x_ds.dtype, dataset_name=dataset_name
            )

        return outputs
