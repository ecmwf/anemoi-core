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
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.utils.config import DotDict

from .hierarchical_autoencoder import AnemoiModelHierarchicalAutoEncoder

LOGGER = logging.getLogger(__name__)


class AnemoiModelDisentangledHierarchicalEncProcDec(AnemoiModelHierarchicalAutoEncoder):
    """Disentangled hierarchical forecaster.

    Re-uses a pretrained *hierarchical* autoencoder
    (:class:`AnemoiModelHierarchicalAutoEncoder`) as a frozen-able
    representation: the ``encoder`` + downscale path map data to a bottleneck
    latent, and the upscale path + ``decoder`` map a bottleneck latent back to
    data. Forecasting dynamics are learned by two *new* trainable modules that
    operate purely at the bottleneck mesh:

    - ``latent_blender``: a learned mapper that fuses the per-input-timestep
      bottleneck latents (e.g. t-1, t) into a single bottleneck state.
    - ``processor``: a learned processor at the deepest hidden level that maps
      the blended latent to the predicted next-step bottleneck latent (mirrors
      the main processor of :class:`AnemoiModelEncProcDecHierarchical`).

    Design notes
    ------------
    * Each input timestep is encoded **independently through the same encoder**
      (so a frozen, reconstruction-trained encoder is used exactly as it was
      trained — this requires the AE to have been trained with
      ``multistep_input = 1``). Datasets are summed at the bottleneck, exactly as
      the autoencoder does.
    * The hierarchical AE is a U-Net with **skip connections**: the upscale path
      consumes per-level latents captured on the way down. A forecaster has no
      t+1 skips, so we re-use the skip latents of the **most recent input
      timestep** (frozen). This injects persisted fine-scale detail into the
      reconstruction; it is the main approximation to monitor.
    * ``model.latent_skip`` (default True) adds the last input bottleneck latent
      as a residual around the processor (latent-space increment learning).
    * ``model.latent_rollout`` (default False) advances the bottleneck buffer in
      latent space on rollout steps > 0 instead of re-encoding. Note this re-uses
      stale skip latents and feeds processor-output latents back into a blender
      trained on encoder-output latents — keep it off until rollout=1 training is
      validated.

    Freeze the representation via ``training.submodules_to_freeze``, e.g.
    ``[encoder, downscale, upscale, decoder]`` (add ``down_level_processor`` /
    ``up_level_processor`` only if the AE used level processing). The new
    ``latent_blender`` and ``processor`` always stay trainable.
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
        self.latent_rollout = model_config.model.model.get("latent_rollout", False)

    def _calculate_input_dim(self, dataset_name: str) -> int:
        # Encode one timestep at a time -> no n_step_input multiplier.
        # This matches an autoencoder trained with multistep_input=1, so the
        # pretrained encoder's first-layer weights load by name.
        return self.num_input_channels[dataset_name] + self.node_attributes.attr_ndims[dataset_name]

    def _build_networks(self, model_config: DotDict) -> None:
        # Builds encoder / downscale / upscale / decoder (+ optional level
        # processors) with the SAME submodule names as the hierarchical AE, so
        # pretrained AE weights transfer by name.
        super()._build_networks(model_config)

        hidden_last = self._graph_name_hidden[self.num_hidden - 1]
        bottleneck_dim = self.hidden_dims[hidden_last]

        # Trainable bottleneck processor at the deepest hidden level
        # (mirrors AnemoiModelEncProcDecHierarchical main processor).
        self.processor_graph_provider = create_graph_provider(
            graph=self._graph_data[(hidden_last, "to", hidden_last)],
            edge_attributes=model_config.model.processor.get("sub_graph_edge_attributes"),
            src_size=self.node_attributes.num_nodes[hidden_last],
            dst_size=self.node_attributes.num_nodes[hidden_last],
            trainable_size=model_config.model.processor.get("trainable_size", 0),
        )
        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,
            num_channels=bottleneck_dim,
            edge_dim=self.processor_graph_provider.edge_dim,
        )

        # Trainable latent blender: fuses the n_step_input per-timestep bottleneck
        # latents (datasets already summed) into a single bottleneck state. Uses
        # the same deepest-level hidden->hidden edges as the processor.
        self.latent_blender = instantiate(
            model_config.model.encoder,
            _recursive_=False,
            in_channels_src=bottleneck_dim * self.n_step_input,
            in_channels_dst=self.node_attributes.attr_ndims[hidden_last],
            hidden_dim=bottleneck_dim,
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
        """Assemble encoder input for a single timestep (no n_step_input slicing)."""
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

    def _encode_to_bottleneck(
        self,
        x_t: Tensor,
        x_hidden_latents: dict,
        shard_shapes_hidden_dict: dict,
        batch_size: int,
        grid_shard_shapes: dict | None,
        in_out_sharded: bool,
        model_comm_group,
        dataset_name: str,
    ) -> tuple[Tensor, dict]:
        """Encode one timestep of one dataset to the bottleneck latent.

        Mirrors the down path of the hierarchical autoencoder. Returns the
        bottleneck latent and the per-level skip latents captured on the way down.
        """
        x_data_latent, shard_shapes_data = self._assemble_single_input(
            x_t, batch_size, grid_shard_shapes, model_comm_group, dataset_name
        )

        enc_edge_attr, enc_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[dataset_name].get_edges(
            batch_size=batch_size, model_comm_group=model_comm_group
        )
        _, x_latent = self.encoder[dataset_name](
            (x_data_latent, x_hidden_latents[self._graph_name_hidden[0]]),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden_dict[self._graph_name_hidden[0]]),
            edge_attr=enc_edge_attr,
            edge_index=enc_edge_index,
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,
            x_dst_is_sharded=False,
            keep_x_dst_sharded=True,
            edge_shard_shapes=enc_edge_shard_shapes,
        )

        skips = {}
        for i in range(0, self.num_hidden - 1):
            src_hidden_name = self._graph_name_hidden[i]
            dst_hidden_name = self._graph_name_hidden[i + 1]

            if self.level_process:
                dl_edge_attr, dl_edge_index, dl_edge_shard_shapes = self.down_level_processor_graph_providers[
                    src_hidden_name
                ].get_edges(batch_size=batch_size, model_comm_group=model_comm_group)
                x_latent = self.down_level_processor[src_hidden_name](
                    x_latent,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes_hidden_dict[src_hidden_name],
                    edge_attr=dl_edge_attr,
                    edge_index=dl_edge_index,
                    model_comm_group=model_comm_group,
                    edge_shard_shapes=dl_edge_shard_shapes,
                )

            ds_edge_attr, ds_edge_index, ds_edge_shard_shapes = self.downscale_graph_providers[
                src_hidden_name
            ].get_edges(batch_size=batch_size, model_comm_group=model_comm_group)
            skips[src_hidden_name], x_latent = self.downscale[src_hidden_name](
                (x_latent, x_hidden_latents[dst_hidden_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden_dict[src_hidden_name], shard_shapes_hidden_dict[dst_hidden_name]),
                edge_attr=ds_edge_attr,
                edge_index=ds_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=False,
                keep_x_dst_sharded=True,
                edge_shard_shapes=ds_edge_shard_shapes,
            )

        return x_latent, skips

    def forward(
        self,
        x: dict[str, Tensor],
        rollout_step: Optional[int] = 0,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        dataset_names = list(x.keys())
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = self._resolve_in_out_sharded(
            dataset_names=dataset_names,
            grid_shard_shapes=grid_shard_shapes,
        )
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        # Trainable hidden node embeddings + shard shapes at every level
        x_hidden_latents = {h: self.node_attributes(h, batch_size=batch_size) for h in self._graph_name_hidden}
        shard_shapes_hidden_dict = {
            h: get_shard_shapes(v, 0, model_comm_group=model_comm_group) for h, v in x_hidden_latents.items()
        }
        hidden_last = self._graph_name_hidden[self.num_hidden - 1]

        proc_edge_attr, proc_edge_index, proc_edge_shard_shapes = self.processor_graph_provider.get_edges(
            batch_size=batch_size, model_comm_group=model_comm_group
        )

        if rollout_step == 0 or not self.latent_rollout:
            # Encode every input timestep -> per-timestep bottleneck latent
            # (summed over datasets). Keep the LAST input timestep's skips.
            per_step_bottleneck = []
            skip_latents = {}
            for t in range(self.n_step_input):
                dataset_bottlenecks = {}
                for dataset_name in dataset_names:
                    x_t = x[dataset_name][:, t : t + 1, ...]
                    z_ds, skips_ds = self._encode_to_bottleneck(
                        x_t,
                        x_hidden_latents,
                        shard_shapes_hidden_dict,
                        batch_size,
                        grid_shard_shapes,
                        in_out_sharded[dataset_name],
                        model_comm_group,
                        dataset_name,
                    )
                    dataset_bottlenecks[dataset_name] = z_ds
                    if t == self.n_step_input - 1:
                        skip_latents[dataset_name] = skips_ds
                per_step_bottleneck.append(sum(dataset_bottlenecks.values()))

            self.skip_latents = skip_latents
            # [n_grid_hidden_last(_sharded), n_step_input, bottleneck_dim]
            self.z_accum = torch.stack(per_step_bottleneck, dim=1)
            z_skip = per_step_bottleneck[-1]
        else:
            # Latent rollout: shift buffer, insert previous processor output.
            self.z_accum = self.z_accum.roll(-1, dims=1)
            self.z_accum[:, -1, ...] = self.z_latent_proc
            z_skip = self.z_latent_proc

        # Blend the per-timestep bottleneck latents into a single bottleneck state.
        z_flat = einops.rearrange(self.z_accum, "bg n c -> bg (n c)")
        _, z_blend = self.latent_blender(
            (z_flat, x_hidden_latents[hidden_last]),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden_dict[hidden_last], shard_shapes_hidden_dict[hidden_last]),
            edge_attr=proc_edge_attr,
            edge_index=proc_edge_index,
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,
            x_dst_is_sharded=False,
            keep_x_dst_sharded=True,
            edge_shard_shapes=proc_edge_shard_shapes,
        )

        # Bottleneck processor: predict next-step bottleneck latent.
        z_proc = self.processor(
            z_blend,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden_dict[hidden_last],
            edge_attr=proc_edge_attr,
            edge_index=proc_edge_index,
            model_comm_group=model_comm_group,
            edge_shard_shapes=proc_edge_shard_shapes,
        )
        if self.latent_skip:
            z_proc = z_proc + z_skip
        self.z_latent_proc = z_proc

        # Up path + decoder per dataset, re-using the most-recent input skips.
        outputs = {}
        for dataset_name in dataset_names:
            x_latent = z_proc
            for i in range(self.num_hidden - 1, 0, -1):
                src_hidden_name = self._graph_name_hidden[i]
                dst_hidden_name = self._graph_name_hidden[i - 1]

                us_edge_attr, us_edge_index, us_edge_shard_shapes = self.upscale_graph_providers[
                    src_hidden_name
                ].get_edges(batch_size=batch_size, model_comm_group=model_comm_group)
                x_latent = self.upscale[src_hidden_name](
                    (x_latent, self.skip_latents[dataset_name][dst_hidden_name]),
                    batch_size=batch_size,
                    shard_shapes=(
                        shard_shapes_hidden_dict[src_hidden_name],
                        shard_shapes_hidden_dict[dst_hidden_name],
                    ),
                    edge_attr=us_edge_attr,
                    edge_index=us_edge_index,
                    model_comm_group=model_comm_group,
                    x_src_is_sharded=True,
                    x_dst_is_sharded=True,
                    keep_x_dst_sharded=True,
                    edge_shard_shapes=us_edge_shard_shapes,
                )

                if self.level_process:
                    ul_edge_attr, ul_edge_index, ul_edge_shard_shapes = self.up_level_processor_graph_providers[
                        dst_hidden_name
                    ].get_edges(batch_size=batch_size, model_comm_group=model_comm_group)
                    x_latent = self.up_level_processor[dst_hidden_name](
                        x_latent,
                        edge_attr=ul_edge_attr,
                        edge_index=ul_edge_index,
                        batch_size=batch_size,
                        shard_shapes=shard_shapes_hidden_dict[dst_hidden_name],
                        model_comm_group=model_comm_group,
                        edge_shard_shapes=ul_edge_shard_shapes,
                    )

            x_target_latent, shard_shapes_target = self._assemble_forcings(
                x[dataset_name], batch_size, grid_shard_shapes, model_comm_group, dataset_name
            )

            dec_edge_attr, dec_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size, model_comm_group=model_comm_group
            )
            x_out = self.decoder[dataset_name](
                (x_latent, x_target_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden_dict[self._graph_name_hidden[0]], shard_shapes_target),
                edge_attr=dec_edge_attr,
                edge_index=dec_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=in_out_sharded[dataset_name],
                keep_x_dst_sharded=in_out_sharded[dataset_name],
                edge_shard_shapes=dec_edge_shard_shapes,
            )

            outputs[dataset_name] = self._assemble_output(
                x_out, batch_size, ensemble_size, x[dataset_name].dtype, dataset_name
            )

        return outputs
