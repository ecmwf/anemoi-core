# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from dataclasses import dataclass
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


@dataclass
class _ForwardContext:
    """Sizes, sharding and hidden-mesh tensors shared by the steps of one forward pass."""

    dataset_names: list[str]
    batch_size: int
    ensemble_size: int
    in_out_sharded: dict[str, bool]
    x_hidden_latent: Tensor
    shard_sizes_hidden: ShardSizes
    proc_edge_attr: Tensor
    proc_edge_index: Tensor
    proc_edge_shard_sizes: object


class AnemoiModelDisentangledEncProcDec(AnemoiModelEncProcDec):
    """Disentangled encoder-processor-decoder.

    Re-uses the encoder/decoder of an autoencoder trained with
    ``multistep_input = 1`` (the ``Autoencoder`` task); submodule names match
    ``AnemoiModelEncProcDec``, so its weights load by name and can be frozen.

    Each input timestep of each dataset is encoded independently and the
    datasets of one timestep are combined by the ``latent_aggregator``. The
    ``n_step_input`` latents are fused by a trainable ``latent_blender`` and the
    processor predicts the next latent state, which is decoded per dataset.
    """

    def _build_networks(self, model_config: DotDict) -> None:
        super()._build_networks(model_config)
        assert len(self.hidden_names) == 1, "The disentangled model supports a single hidden mesh."
        hidden_name = self.hidden_names[0]
        num_channels = self.latent_aggregator.hidden_dim

        # Mapper hidden -> hidden fusing the n_step_input latents into one state, over the
        # processor edges. Uses ``latent_blender`` if configured, else the first encoder's mapper.
        blender_config = model_config.get("latent_blender") or next(iter(model_config.encoders.values())).mapper
        self.latent_blender = instantiate(
            blender_config,
            _recursive_=False,
            in_channels_src=num_channels * self.n_step_input,
            in_channels_dst=self.node_attributes.attr_ndims[hidden_name],
            num_channels=num_channels,
            edge_dim=self.processor_graph_provider[hidden_name].edge_dim,
        )

    def _calculate_input_dim(self, dataset_name: str) -> int:
        # One timestep per encoder call: no n_step_input multiplier. This matches an
        # encoder trained with multistep_input=1, so its weights load by name.
        return self.num_input_channels[dataset_name] + self.node_attributes.attr_ndims[dataset_name]

    def _forward_context(
        self,
        x: dict[str, Tensor],
        model_comm_group: Optional[ProcessGroup],
        grid_shard_sizes: DatasetShardSizes | None,
    ) -> _ForwardContext:
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
        return _ForwardContext(
            dataset_names=dataset_names,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            in_out_sharded=in_out_sharded,
            x_hidden_latent=x_hidden_latent,
            shard_sizes_hidden=shard_sizes_hidden,
            proc_edge_attr=proc_edge_attr,
            proc_edge_index=proc_edge_index,
            proc_edge_shard_sizes=proc_edge_shard_sizes,
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

    def _encode(
        self,
        x: dict[str, Tensor],
        ctx: _ForwardContext,
        model_comm_group: Optional[ProcessGroup],
        grid_shard_sizes: DatasetShardSizes | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Encode every input timestep independently.

        Returns the latents of shape (nodes, n_step_input, channels) and the data-node
        encoder outputs of the most recent input timestep (decoder targets for
        ``target_node_features: [encoded_data]``).
        """
        step_latents = []
        encoded_data = {}
        for t in range(self.n_step_input):
            dataset_latents = {}
            for dataset_name in ctx.dataset_names:
                if dataset_name not in self.input_datasets:
                    continue
                x_data_latent, shard_sizes_data = self._assemble_step_input(
                    x[dataset_name][:, t : t + 1],
                    ctx.batch_size,
                    grid_shard_sizes,
                    model_comm_group,
                    dataset_name,
                )
                enc_edge_attr, enc_edge_index, enc_edge_shard_sizes = self.encoder_graph_provider[
                    dataset_name
                ].get_edges(batch_size=ctx.batch_size, model_comm_group=model_comm_group)
                x_data_latent, dataset_latents[dataset_name] = self.encoder[self.dataset2encoder[dataset_name]](
                    (x_data_latent, ctx.x_hidden_latent),
                    batch_size=ctx.batch_size,
                    shard_info=BipartiteGraphShardInfo(
                        src_nodes=shard_sizes_data,
                        dst_nodes=ctx.shard_sizes_hidden,
                        edges=enc_edge_shard_sizes,
                    ),
                    edge_attr=enc_edge_attr,
                    edge_index=enc_edge_index,
                    model_comm_group=model_comm_group,
                    keep_x_dst_sharded=True,
                )
                if t == self.n_step_input - 1:
                    encoded_data[dataset_name] = x_data_latent
            step_latents.append(self.latent_aggregator(ctx.x_hidden_latent, dataset_latents))

        return torch.stack(step_latents, dim=1), encoded_data

    def _process(
        self,
        latents: Tensor,
        ctx: _ForwardContext,
        model_comm_group: Optional[ProcessGroup],
    ) -> Tensor:
        """Blend the (nodes, n_step_input, channels) latents and predict the next latent state."""
        _, x_latent = self.latent_blender(
            (einops.rearrange(latents, "nodes time channels -> nodes (time channels)"), ctx.x_hidden_latent),
            batch_size=ctx.batch_size,
            shard_info=BipartiteGraphShardInfo(
                src_nodes=ctx.shard_sizes_hidden,
                dst_nodes=ctx.shard_sizes_hidden,
                edges=ctx.proc_edge_shard_sizes,
            ),
            edge_attr=ctx.proc_edge_attr,
            edge_index=ctx.proc_edge_index,
            model_comm_group=model_comm_group,
            keep_x_dst_sharded=True,
        )
        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=ctx.batch_size,
            shard_info=GraphShardInfo(nodes=ctx.shard_sizes_hidden, edges=ctx.proc_edge_shard_sizes),
            edge_attr=ctx.proc_edge_attr,
            edge_index=ctx.proc_edge_index,
            model_comm_group=model_comm_group,
        )
        if self.latent_skip:
            # Skip connection from the most recent latent.
            x_latent_proc = x_latent_proc + latents[:, -1]
        return x_latent_proc

    def _decode(
        self,
        x_latent: Tensor,
        x: dict[str, Tensor],
        encoded_data: dict[str, Tensor],
        ctx: _ForwardContext,
        model_comm_group: Optional[ProcessGroup],
        grid_shard_sizes: DatasetShardSizes | None,
    ) -> dict[str, Tensor]:
        """Decode the latent state per dataset and add the residual of the current input."""
        x_out_dict = {}
        for dataset_name in ctx.dataset_names:
            if dataset_name not in self.target_datasets:
                continue
            x_target_latent, shard_sizes_target = self._assemble_targets(
                x[dataset_name],
                encoded_data.get(dataset_name),
                ctx.batch_size,
                grid_shard_sizes,
                model_comm_group,
                dataset_name,
            )
            dec_edge_attr, dec_edge_index, dec_edge_shard_sizes = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=ctx.batch_size, model_comm_group=model_comm_group
            )
            x_out = self.decoder[self.dataset2decoder[dataset_name]](
                (x_latent, x_target_latent),
                batch_size=ctx.batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=ctx.shard_sizes_hidden,
                    dst_nodes=shard_sizes_target,
                    edges=dec_edge_shard_sizes,
                ),
                edge_attr=dec_edge_attr,
                edge_index=dec_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=ctx.in_out_sharded[dataset_name],
            )
            x_skip = self.residual[dataset_name](
                x[dataset_name],
                grid_shard_sizes=grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None,
                model_comm_group=model_comm_group,
                n_step_output=self.n_step_output,
            )
            x_out_dict[dataset_name] = self._assemble_output(
                x_out,
                x_skip,
                batch_size=ctx.batch_size,
                ensemble_size=ctx.ensemble_size,
                dtype=x[dataset_name].dtype,
                dataset_name=dataset_name,
            )
        return x_out_dict

    def forward(
        self,
        x: dict[str, Tensor],
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Encode each input timestep, blend, process and decode.

        Parameters
        ----------
        x : dict[str, Tensor]
            Input per dataset, shaped (batch, time, ensemble, grid, vars).
        model_comm_group : ProcessGroup, optional
            Model communication group.
        grid_shard_sizes : DatasetShardSizes, optional
            Per-dataset shard sizes of the grid dimension.
        **kwargs
            Ignored.

        Returns
        -------
        dict[str, Tensor]
            Prediction per dataset.
        """
        ctx = self._forward_context(x, model_comm_group, grid_shard_sizes)
        latents, encoded_data = self._encode(x, ctx, model_comm_group, grid_shard_sizes)
        x_latent = self._process(latents, ctx, model_comm_group)
        return self._decode(x_latent, x, encoded_data, ctx, model_comm_group, grid_shard_sizes)
