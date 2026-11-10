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

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class DisentangledMixin:
    """Shared pieces of the disentangled (per-timestep encoding) forecasters.

    A disentangled forecaster re-uses the encoder/decoder of an autoencoder
    trained with ``multistep_input = 1`` (the ``Autoencoder`` task). Each input
    timestep is encoded independently through that encoder, the per-timestep
    latents are fused by a trainable ``latent_blender`` and advanced in time by
    the processor.

    With ``model.latent_rollout: True`` the model keeps a buffer of latents
    between calls: on ``rollout_step == 0`` it encodes all input timesteps, on
    later steps it shifts the buffer and appends the previous processor output
    instead of re-encoding from data space. This needs ``rollout_step`` passed to
    ``forward`` (see ``anemoi.training.train.methods.LatentRolloutTraining``).
    Without it every call re-encodes, which is always correct.
    """

    def _init_latent_rollout(self, model_config: DotDict) -> None:
        self.latent_rollout = bool(model_config.model.get("latent_rollout", False))
        self._latent_buffer: Optional[Tensor] = None
        self._last_latent: Optional[Tensor] = None
        self._cached_encoder_outputs: dict = {}

    def _calculate_input_dim(self, dataset_name: str) -> int:
        # One timestep per encoder call: no n_step_input multiplier. This matches an
        # encoder trained with multistep_input=1, so its weights load by name.
        return self.num_input_channels[dataset_name] + self.node_attributes.attr_ndims[dataset_name]

    def _build_latent_blender(self, model_config: DotDict, num_channels: int, hidden_name: str, edge_dim: int) -> None:
        """Mapper hidden -> hidden fusing ``n_step_input`` latents into one state.

        Uses ``model.latent_blender`` if given, otherwise the mapper config of the
        first encoder. It runs over the processor's hidden -> hidden edges.
        """
        blender_config = model_config.get("latent_blender") or next(iter(model_config.encoders.values())).mapper
        self.latent_blender = instantiate(
            blender_config,
            _recursive_=False,
            in_channels_src=num_channels * self.n_step_input,
            in_channels_dst=self.node_attributes.attr_ndims[hidden_name],
            num_channels=num_channels,
            edge_dim=edge_dim,
        )

    def _assemble_step_input(
        self,
        x_t: Tensor,
        batch_size: int,
        grid_shard_sizes: DatasetShardSizes | None,
        model_comm_group: Optional[ProcessGroup],
        dataset_name: str,
    ) -> tuple[Tensor, ShardSizes]:
        """Encoder source features for one timestep ``x_t`` of shape (batch, 1, ensemble, grid, vars)."""
        node_attributes_data = self.node_attributes(dataset_name, batch_size=batch_size)
        ds_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None
        if ds_shard_sizes is not None:
            node_attributes_data = shard_tensor(node_attributes_data, 0, ds_shard_sizes, model_comm_group)

        x_data_latent = torch.cat(
            (
                einops.rearrange(x_t, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,
        )
        return x_data_latent, ds_shard_sizes

    def _residual(
        self,
        x: Tensor,
        grid_shard_sizes: DatasetShardSizes | None,
        model_comm_group: Optional[ProcessGroup],
        dataset_name: str,
    ) -> Tensor:
        """Residual skip computed from the full (current) input window."""
        return self.residual[dataset_name](
            x,
            grid_shard_sizes=grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None,
            model_comm_group=model_comm_group,
            n_step_output=self.n_step_output,
        )

    def _use_latent_buffer(self, rollout_step: int) -> bool:
        return self.latent_rollout and rollout_step > 0 and self._latent_buffer is not None

    def _advance_latent_buffer(self) -> Tensor:
        """Shift the latent buffer by one step, appending the previous processor output."""
        self._latent_buffer = torch.cat((self._latent_buffer[:, 1:], self._last_latent.unsqueeze(1)), dim=1)
        return self._last_latent

    def _blend(
        self,
        latents: Tensor,
        x_hidden_latent: Tensor,
        shard_sizes_hidden: ShardSizes,
        edge_attr: Tensor,
        edge_index: Tensor,
        edge_shard_sizes,
        batch_size: int,
        model_comm_group: Optional[ProcessGroup],
    ) -> Tensor:
        """Fuse (nodes, n_step_input, channels) latents into a single (nodes, channels) state."""
        _, blended = self.latent_blender(
            (einops.rearrange(latents, "nodes time channels -> nodes (time channels)"), x_hidden_latent),
            batch_size=batch_size,
            shard_info=BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_hidden,
                edges=edge_shard_sizes,
            ),
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
            keep_x_dst_sharded=True,
        )
        return blended


class AnemoiModelDisentangledEncProcDec(DisentangledMixin, AnemoiModelEncProcDec):
    """Disentangled encoder-processor-decoder with optional latent rollout.

    Each input timestep of each dataset is encoded independently; the datasets of
    one timestep are combined by the ``latent_aggregator``. The resulting
    ``n_step_input`` latents are fused by ``latent_blender`` and the processor
    predicts the next latent state, which is decoded per dataset.
    """

    def __init__(self, *, model_config: DotDict, **kwargs) -> None:
        super().__init__(model_config=model_config, **kwargs)
        self._init_latent_rollout(model_config)

    def _build_networks(self, model_config: DotDict) -> None:
        super()._build_networks(model_config)
        assert len(self.hidden_names) == 1, "The disentangled model supports a single hidden mesh."
        self._build_latent_blender(
            model_config,
            num_channels=self.latent_aggregator.hidden_dim,
            hidden_name=self.hidden_names[0],
            edge_dim=self.processor_graph_provider[self.hidden_names[0]].edge_dim,
        )

    def forward(
        self,
        x: dict[str, Tensor],
        rollout_step: int = 0,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : dict[str, Tensor]
            Input per dataset, shaped (batch, time, ensemble, grid, vars).
        rollout_step : int, optional
            Rollout step. With ``latent_rollout`` enabled, steps > 0 advance the
            latent buffer instead of re-encoding ``x``.
        model_comm_group : ProcessGroup, optional
        grid_shard_sizes : DatasetShardSizes, optional
        """
        dataset_names = list(x.keys())
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = self._resolve_in_out_sharded(dataset_names=dataset_names, grid_shard_sizes=grid_shard_sizes)
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        hidden_name = self._get_active_hidden_name(dataset_names)
        x_hidden_latent = self.node_attributes(hidden_name, batch_size=batch_size)
        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)

        proc_edge_attr, proc_edge_index, proc_edge_shard_sizes = self.processor_graph_provider[hidden_name].get_edges(
            batch_size=batch_size, model_comm_group=model_comm_group
        )

        if self._use_latent_buffer(rollout_step):
            x_latent_skip = self._advance_latent_buffer()
        else:
            step_latents = []
            encoded_data = {}
            for t in range(self.n_step_input):
                dataset_latents = {}
                for dataset_name in dataset_names:
                    if dataset_name not in self.input_datasets:
                        continue
                    x_data_latent, shard_sizes_data = self._assemble_step_input(
                        x[dataset_name][:, t : t + 1],
                        batch_size,
                        grid_shard_sizes,
                        model_comm_group,
                        dataset_name,
                    )
                    enc_edge_attr, enc_edge_index, enc_edge_shard_sizes = self.encoder_graph_provider[
                        dataset_name
                    ].get_edges(batch_size=batch_size, model_comm_group=model_comm_group)
                    x_data_latent, dataset_latents[dataset_name] = self.encoder[self.dataset2encoder[dataset_name]](
                        (x_data_latent, x_hidden_latent),
                        batch_size=batch_size,
                        shard_info=BipartiteGraphShardInfo(
                            src_nodes=shard_sizes_data,
                            dst_nodes=shard_sizes_hidden,
                            edges=enc_edge_shard_sizes,
                        ),
                        edge_attr=enc_edge_attr,
                        edge_index=enc_edge_index,
                        model_comm_group=model_comm_group,
                        keep_x_dst_sharded=True,
                    )
                    if t == self.n_step_input - 1:
                        encoded_data[dataset_name] = x_data_latent
                step_latents.append(self.latent_aggregator(x_hidden_latent, dataset_latents))

            # Data-node encoder outputs of the most recent input step, used as decoder
            # targets when target_node_features includes ``encoded_data``.
            self._cached_encoder_outputs = encoded_data
            self._latent_buffer = torch.stack(step_latents, dim=1)
            x_latent_skip = step_latents[-1]

        x_latent = self._blend(
            self._latent_buffer,
            x_hidden_latent,
            shard_sizes_hidden,
            proc_edge_attr,
            proc_edge_index,
            proc_edge_shard_sizes,
            batch_size,
            model_comm_group,
        )
        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=batch_size,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden, edges=proc_edge_shard_sizes),
            edge_attr=proc_edge_attr,
            edge_index=proc_edge_index,
            model_comm_group=model_comm_group,
        )
        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent_skip
        self._last_latent = x_latent_proc

        x_out_dict = {}
        for dataset_name in dataset_names:
            if dataset_name not in self.target_datasets:
                continue
            x_target_latent, shard_sizes_target = self._assemble_targets(
                x[dataset_name],
                self._cached_encoder_outputs.get(dataset_name),
                batch_size,
                grid_shard_sizes,
                model_comm_group,
                dataset_name,
            )
            dec_edge_attr, dec_edge_index, dec_edge_shard_sizes = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size, model_comm_group=model_comm_group
            )
            x_out = self.decoder[self.dataset2decoder[dataset_name]](
                (x_latent_proc, x_target_latent),
                batch_size=batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_hidden,
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
                batch_size=batch_size,
                ensemble_size=ensemble_size,
                dtype=x[dataset_name].dtype,
                dataset_name=dataset_name,
            )

        return x_out_dict
