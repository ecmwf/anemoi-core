# (C) Copyright 2025- Anemoi contributors.
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
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.models.disentangled_encprocdec import DisentangledMixin
from anemoi.models.models.hierarchical import AnemoiModelEncProcDecHierarchical
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelDisentangledHierarchicalEncProcDec(DisentangledMixin, AnemoiModelEncProcDecHierarchical):
    """Disentangled hierarchical forecaster with optional latent rollout.

    Re-uses a hierarchical autoencoder (an ``AnemoiModelEncProcDecHierarchical``
    trained with the ``Autoencoder`` task and ``multistep_input = 1``). The
    submodules keep the names of the parent class (``encoder``, ``upscale``,
    ``downscale``, ``decoder``, level processors, ``processor``), so pretrained
    weights transfer by name and can be frozen via ``training.submodules_to_freeze``.

    Each input timestep is encoded independently and taken down the hierarchy
    (``upscale`` path: fine -> coarse) to the deepest level. The ``n_step_input``
    bottleneck latents are fused by ``latent_blender`` and advanced by the
    deepest-level ``processor``; the result is brought back up (``downscale``
    path: coarse -> fine) and decoded.

    The U-Net skip connections of the up path come from the **most recent input
    timestep** (there are no skips for the predicted time). With
    ``latent_rollout`` these skips stay those of the last *encoded* input, so
    they get older with every rollout step; this is the main approximation to
    monitor.
    """

    def __init__(self, *, model_config: DotDict, **kwargs) -> None:
        super().__init__(model_config=model_config, **kwargs)
        self._init_latent_rollout(model_config)

    def _build_networks(self, model_config: DotDict) -> None:
        super()._build_networks(model_config)
        bottleneck = self.hidden_levels[-1]
        self._build_latent_blender(
            model_config,
            num_channels=self.hidden_dims[bottleneck],
            hidden_name=bottleneck,
            edge_dim=self.processor_graph_provider.edge_dim,
        )

    def _encode_to_bottleneck(
        self,
        x: dict[str, Tensor],
        t: int,
        x_hidden_latents: dict[str, Tensor],
        shard_sizes_hidden: dict,
        batch_size: int,
        grid_shard_sizes: DatasetShardSizes | None,
        model_comm_group: Optional[ProcessGroup],
    ) -> tuple[Tensor, dict, dict, dict]:
        """Encode timestep ``t`` of every dataset down to the deepest level.

        Returns the bottleneck latent, the per-level skips, the per-level upscale
        source outputs and the data-node encoder outputs.
        """
        dataset_latents, encoded_data, shard_sizes_data = {}, {}, {}
        for dataset_name in x:
            if dataset_name not in self.input_datasets:
                continue
            x_data_latent, shard_sizes_data[dataset_name] = self._assemble_step_input(
                x[dataset_name][:, t : t + 1], batch_size, grid_shard_sizes, model_comm_group, dataset_name
            )
            enc_edge_attr, enc_edge_index, enc_edge_shard_sizes = self.encoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size, model_comm_group=model_comm_group
            )
            encoded_data[dataset_name], dataset_latents[dataset_name] = self.encoder[
                self.dataset2encoder[dataset_name]
            ](
                (x_data_latent, x_hidden_latents[self.hidden_levels[0]]),
                batch_size=batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_data[dataset_name],
                    dst_nodes=shard_sizes_hidden[self.hidden_levels[0]],
                    edges=enc_edge_shard_sizes,
                ),
                edge_attr=enc_edge_attr,
                edge_index=enc_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,
            )

        x_latent = self.latent_aggregator(x_hidden_latents[self.hidden_levels[0]], dataset_latents)

        level_skips, level_encoded = {}, {}
        for i in range(self.num_hidden - 1):
            src, dst = self.hidden_levels[i], self.hidden_levels[i + 1]
            if self.level_process:
                edge_attr, edge_index, edge_shard_sizes = self.down_level_processor_graph_providers[src].get_edges(
                    batch_size=batch_size, model_comm_group=model_comm_group
                )
                x_latent = self.down_level_processor[src](
                    x_latent,
                    batch_size=batch_size,
                    shard_info=GraphShardInfo(nodes=shard_sizes_hidden[src], edges=edge_shard_sizes),
                    edge_attr=edge_attr,
                    edge_index=edge_index,
                    model_comm_group=model_comm_group,
                )
            level_skips[src] = x_latent

            edge_attr, edge_index, edge_shard_sizes = self.upscale_graph_providers[src].get_edges(
                batch_size=batch_size, model_comm_group=model_comm_group
            )
            level_encoded[src], x_latent = self.upscale[src](
                (x_latent, x_hidden_latents[dst]),
                batch_size=batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_hidden[src],
                    dst_nodes=shard_sizes_hidden[dst],
                    edges=edge_shard_sizes,
                ),
                edge_attr=edge_attr,
                edge_index=edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,
            )

        return x_latent, level_skips, level_encoded, encoded_data

    def forward(
        self,
        x: dict[str, Tensor],
        rollout_step: int = 0,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        dataset_names = list(x.keys())
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = self._resolve_in_out_sharded(dataset_names=dataset_names, grid_shard_sizes=grid_shard_sizes)
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        x_hidden_latents, shard_sizes_hidden = {}, {}
        for hidden in self.hidden_levels:
            latent = self.node_attributes(hidden, batch_size=batch_size)
            shard_sizes_hidden[hidden] = get_shard_sizes(latent, 0, model_comm_group=model_comm_group)
            x_hidden_latents[hidden] = shard_tensor(latent, 0, shard_sizes_hidden[hidden], model_comm_group)
        bottleneck = self.hidden_levels[-1]

        proc_edge_attr, proc_edge_index, proc_edge_shard_sizes = self.processor_graph_provider.get_edges(
            batch_size=batch_size, model_comm_group=model_comm_group
        )

        if self._use_latent_buffer(rollout_step):
            z_skip = self._advance_latent_buffer()
        else:
            step_latents = []
            for t in range(self.n_step_input):
                z_t, level_skips, level_encoded, encoded_data = self._encode_to_bottleneck(
                    x, t, x_hidden_latents, shard_sizes_hidden, batch_size, grid_shard_sizes, model_comm_group
                )
                step_latents.append(z_t)
            # Skips and data-node encodings of the most recent input timestep.
            self._cached_encoder_outputs = {
                "level_skips": level_skips,
                "level_encoded": level_encoded,
                "encoded_data": encoded_data,
            }
            self._latent_buffer = torch.stack(step_latents, dim=1)
            z_skip = step_latents[-1]

        z = self._blend(
            self._latent_buffer,
            x_hidden_latents[bottleneck],
            shard_sizes_hidden[bottleneck],
            proc_edge_attr,
            proc_edge_index,
            proc_edge_shard_sizes,
            batch_size,
            model_comm_group,
        )
        z_proc = self.processor(
            z,
            batch_size=batch_size,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden[bottleneck], edges=proc_edge_shard_sizes),
            edge_attr=proc_edge_attr,
            edge_index=proc_edge_index,
            model_comm_group=model_comm_group,
        )
        if self.latent_skip:
            z_proc = z_proc + z_skip
        self._last_latent = z_proc

        cache = self._cached_encoder_outputs
        x_latent = z_proc
        for i in reversed(range(1, self.num_hidden)):
            src, dst = self.hidden_levels[i], self.hidden_levels[i - 1]
            edge_attr, edge_index, edge_shard_sizes = self.downscale_graph_providers[dst].get_edges(
                batch_size=batch_size, model_comm_group=model_comm_group
            )
            x_latent = self.downscale[dst](
                (x_latent, cache["level_encoded"][dst]),
                batch_size=batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_hidden[src],
                    dst_nodes=shard_sizes_hidden[dst],
                    edges=edge_shard_sizes,
                ),
                edge_attr=edge_attr,
                edge_index=edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,
            )
            x_latent = x_latent + cache["level_skips"][dst]
            if self.level_process:
                edge_attr, edge_index, edge_shard_sizes = self.up_level_processor_graph_providers[dst].get_edges(
                    batch_size=batch_size, model_comm_group=model_comm_group
                )
                x_latent = self.up_level_processor[dst](
                    x_latent,
                    edge_attr=edge_attr,
                    edge_index=edge_index,
                    batch_size=batch_size,
                    shard_info=GraphShardInfo(nodes=shard_sizes_hidden[dst], edges=edge_shard_sizes),
                    model_comm_group=model_comm_group,
                )

        x_out_dict = {}
        for dataset_name in dataset_names:
            if dataset_name not in self.target_datasets:
                continue
            # Decoder targets as configured by target_node_features; ``encoded_data``
            # comes from the most recent encoded input timestep.
            x_target_latent, shard_sizes_target = self._assemble_targets(
                x[dataset_name],
                cache["encoded_data"].get(dataset_name),
                batch_size,
                grid_shard_sizes,
                model_comm_group,
                dataset_name,
            )
            dec_edge_attr, dec_edge_index, dec_edge_shard_sizes = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size, model_comm_group=model_comm_group
            )
            x_out = self.decoder[self.dataset2decoder[dataset_name]](
                (x_latent, x_target_latent),
                batch_size=batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_hidden[self.hidden_levels[0]],
                    dst_nodes=shard_sizes_target,
                    edges=dec_edge_shard_sizes,
                ),
                edge_attr=dec_edge_attr,
                edge_index=dec_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],
            )
            x_out_dict[dataset_name] = self._assemble_output(
                x_out,
                self._residual(x[dataset_name], grid_shard_sizes, model_comm_group, dataset_name),
                batch_size,
                ensemble_size,
                x[dataset_name].dtype,
                dataset_name,
            )

        return x_out_dict
