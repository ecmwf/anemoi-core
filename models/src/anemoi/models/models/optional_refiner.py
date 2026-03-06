from __future__ import annotations

from typing import Any

import torch
from hydra.utils import instantiate
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.utils.config import DotDict


class OptionalRefinerMixin:
    """Shared helper for optional data-grid refinement."""

    def _configure_optional_refiner(self, model_config: DotDict) -> None:
        refiner_config = model_config.model.get("refiner")
        self.refiner_config = refiner_config
        self.refiner_enabled = refiner_config is not None
        self.refiner_channels = None if refiner_config is None else int(refiner_config.refiner_channels)
        self.num_refiners = 0 if refiner_config is None else int(refiner_config.num_refiners)
        if self.refiner_enabled and self.num_refiners < 1:
            raise ValueError(f"num_refiners must be >= 1, got {self.num_refiners}")
        self.refiner_graph_provider = None
        self.refiners = None

    def _build_optional_refiner_networks(self, model_config: DotDict) -> None:
        self.refiner_graph_provider = torch.nn.ModuleDict()
        self.refiners = torch.nn.ModuleDict()

        if not self.refiner_enabled:
            return

        for dataset_name in self._graph_data.keys():
            self.decoder[dataset_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,
                in_channels_src=self.num_channels,
                in_channels_dst=self.input_dim[dataset_name],
                hidden_dim=self.num_channels,
                out_channels_dst=self.refiner_channels,
                edge_dim=self.decoder_graph_provider[dataset_name].edge_dim,
            )

            refiner_edge_key = (self._graph_name_data, "to", self._graph_name_data)
            if refiner_edge_key not in self._graph_data[dataset_name].edge_types:
                msg = (
                    f"Missing refiner edge type {refiner_edge_key} for dataset '{dataset_name}'. "
                    "Please add a data->data edge block in the graph config."
                )
                raise ValueError(msg)

            self.refiner_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[dataset_name][refiner_edge_key],
                edge_attributes=model_config.model.refiner.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_data],
                dst_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_data],
                trainable_size=model_config.model.refiner.get("trainable_size", 0),
            )

            refiners_for_dataset = []
            for idx in range(self.num_refiners):
                out_channels = self.output_dim[dataset_name] if idx == self.num_refiners - 1 else None
                refiners_for_dataset.append(
                    instantiate(
                        model_config.model.refiner,
                        _recursive_=False,
                        in_channels_src=self.refiner_channels,
                        in_channels_dst=self.refiner_channels,
                        hidden_dim=self.refiner_channels,
                        out_channels_dst=out_channels,
                        edge_dim=self.refiner_graph_provider[dataset_name].edge_dim,
                    ),
                )
            self.refiners[dataset_name] = torch.nn.ModuleList(refiners_for_dataset)

    def _decode_with_optional_refiner(
        self,
        *,
        dataset_name: str,
        x_latent_proc: torch.Tensor,
        x_data_latent: torch.Tensor,
        batch_size: int,
        shard_shapes_hidden: list[list[int]],
        shard_shapes_data: list[list[int]],
        in_out_sharded: bool,
        model_comm_group: ProcessGroup | None = None,
        decoder_kwargs: dict[str, Any] | None = None,
        refiner_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        decoder_edge_attr, decoder_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[
            dataset_name
        ].get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        x_out = self.decoder[dataset_name](
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            edge_attr=decoder_edge_attr,
            edge_index=decoder_edge_index,
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,
            x_dst_is_sharded=in_out_sharded,
            keep_x_dst_sharded=in_out_sharded,
            edge_shard_shapes=dec_edge_shard_shapes,
            **(decoder_kwargs or {}),
        )

        if not self.refiner_enabled:
            return x_out

        refiner_edge_attr, refiner_edge_index, refiner_edge_shard_shapes = self.refiner_graph_provider[
            dataset_name
        ].get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        for refiner in self.refiners[dataset_name]:
            x_out = refiner(
                (x_out, x_out),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_data, shard_shapes_data),
                edge_attr=refiner_edge_attr,
                edge_index=refiner_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=in_out_sharded,
                x_dst_is_sharded=in_out_sharded,
                keep_x_dst_sharded=in_out_sharded,
                edge_shard_shapes=refiner_edge_shard_shapes,
                **(refiner_kwargs or {}),
            )

        return x_out
