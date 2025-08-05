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

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.projection import GraphNodeEmbedder
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiMultiModel(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        sample_provider: "SampleProvider",
        model_config: DotDict,
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()
        model_config = DotDict(model_config)
        self._graph_data = graph_data

        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        self.latent_residual_connection = True
        self._define_residual_connection_indices(
            sample_provider.name_to_index["input"],
            sample_provider.name_to_index["target"],
            residual_connections=model_config.model.get("residual_connections", []),
        )

        def num_channels(name_to_index, **kwargs):
            return len(name_to_index["variables"]) * len(name_to_index["offset"])

        sample_provider = sample_provider.apply(num_channels)
        self.num_input_channels: dict[str, int] = sample_provider["input"].num_channels
        self.num_target_channels: dict[str, int] = sample_provider["target"].num_channels
        self.input_names: list[str] = model_config.model.model.encoder_input_names
        self.hidden_name: str = model_config.model.model.hidden_name
        self.target_names: list[str] = model_config.model.model.decoder_target_names

        # Embedding layers
        self.graph_node_embedder = GraphNodeEmbedder(
            num_input_channels=self.num_target_channels | self.num_input_channels, out_channels=self.num_channels
        )
        # self.unembed_data = NodeProjector(
        #    model_config.model.emb_data,
        #    in_features=self.num_channels,
        #    num_output_channels=self.num_target_channels
        # )

        # Encoder data -> hidden
        self.encoders = nn.ModuleDict({})
        for input_name in self.input_names:
            self.encoders[input_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,
                in_channels_src=self.num_input_channels[input_name] + 4,
                in_channels_dst=4,
                hidden_dim=self.num_channels,
                edge_dim=3,
            )

        # Processor hidden -> hidden
        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,
            num_channels=self.num_channels,
            edge_dim=3,
        )

        # Decoder hidden -> data
        self.decoders = nn.ModuleDict({})
        for target_name in self.target_names:
            self.decoders[target_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,
                in_channels_src=self.num_channels,
                in_channels_dst=self.num_input_channels[target_name] + 4,
                hidden_dim=self.num_channels,
                out_channels_dst=self.num_target_channels[target_name],
                edge_dim=3,
            )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        # TODO: Bring bounding back
        self.boundings = nn.ModuleList([])
        # self.boundings = nn.ModuleList(
        #    [
        #        instantiate(
        #            cfg,
        #            name_to_index=sample_provider.target.name_to_index,
        #            statistics=sample_provider.input.statistics,
        #            name_to_index_stats=sample_provider.input.name_to_index,
        #        )
        #        for cfg in getattr(model_config.model, "bounding", [])
        #    ]
        # )

    def _define_residual_connection_indices(
        self,
        input_name_to_index,
        target_name_to_index,
        residual_connections: list[str],
    ):
        self.residual_connection_indices = {}
        for source in residual_connections:
            input_vars = input_name_to_index[source]["variables"]
            target_vars = target_name_to_index[source]["variables"]
            common_vars = set(target_vars).intersection(input_vars)
            target_idx = [input_vars[v] for v in common_vars]
            input_idx = [target_vars[v] for v in common_vars]
            self.residual_connection_indices[source] = target_idx, input_idx

    def _assemble_input(self, name: str, x: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x.shape: (batch_size, multi_step, ens_dim, grid_size, num_vars)

        # x_src_data_latent = self.graph_node_embedder(
        #   graph[name],
        #   einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
        #   name,
        #   batch_size=batch_size
        # )

        # normalize and add data positional info (lat/lon)
        x_src_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(name, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_skip = x[:, -1, :, :, :]

        return x_src_data_latent, x_skip

    def _assemble_target(self, name: str, x: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x_dst_data_latent = self.graph_node_embedder(
        #   graph[name],
        #   name,
        #   batch_size=batch_size
        # )
        x_dst_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.node_attributes(name, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )
        return x_dst_data_latent

    def _disassemble_target(
        self,
        x: torch.Tensor,
        batch_size: int,
        ensemble_size: int,
    ) -> dict[str, torch.Tensor]:
        return (
            einops.rearrange(
                x,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=torch.float32)
            .clone()
        )

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: tuple[Tensor],
        sub_graph,
        batch_size: int,
        shard_shapes: tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Run mapper with activation checkpoint.

        Parameters
        ----------
        mapper : nn.Module
            Which processor to use
        data : tuple[Tensor]
            tuple of data to pass in
        sub_graph
            Sub graph to use for the mapper
        batch_size: int,
            Batch size
        shard_shapes : tuple[tuple[int, int], tuple[int, int]]
            Shard shapes for the data
        model_comm_group : ProcessGroup
            model communication group, specifies which GPUs work together
            in one model instance
        use_reentrant : bool, optional
            Use reentrant, by default False

        Returns
        -------
        Tensor
            Mapped data
        """
        return checkpoint(
            mapper,
            data,
            sub_graph,
            batch_size=batch_size,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )

    def encode(
        self,
        x: dict[str, torch.Tensor],
        graph: HeteroData,
        shard_shapes_hidden: tuple[list],
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        x_data, x_hidden = x

        x_data_latent, x_hidden_latent, x_data_skip = {}, {}, {}
        for name in self.input_names:
            x_input_data, x_data_skip[name] = self._assemble_input(name, x_data[name], batch_size)

            shard_shapes_input_data = get_shard_shapes(x_input_data, 0, model_comm_group)

            x_data_latent[name], x_hidden_latent[name] = self._run_mapper(
                self.encoders[name],
                (x_input_data, x_hidden),
                sub_graph=graph[(name, "to", self.hidden_name)].to(x_input_data.device),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_input_data, shard_shapes_hidden),
                model_comm_group=model_comm_group,
            )

        return x_data_latent, x_hidden_latent, x_data_skip

    def merge_latents(self, latents: dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: implement different strategies: sum, average, learnable, ...
        return latents[list(latents.keys())[0]]

    def decode(
        self,
        x: tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]],
        graph: HeteroData,
        shard_shapes_hidden: tuple[list],
        batch_size: int,
        ensemble_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ):
        x_hidden_latent, x_encoded_latents, x_raw_target = x

        x_out = {}
        for name in self.target_names:
            if name in x_encoded_latents:
                x_target_latent = x_encoded_latents[name]
            else:
                if name in x_raw_target:
                    x_target_latent = self._assemble_target(name, x_raw_target[name], batch_size)
                else:
                    x_target_latent = self.node_attributes(name, batch_size=batch_size)

            shard_shapes_target_data = get_shard_shapes(
                x_target_latent, 0, model_comm_group
            )  # This may be passed when name in x_target_data

            x_out[name] = self._run_mapper(
                self.decoders[name],
                (x_hidden_latent, x_target_latent),
                sub_graph=graph[(self.hidden_name, "to", name)].to(x_hidden_latent.device),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_target_data),
                model_comm_group=model_comm_group,
            )

            x_out[name] = self._disassemble_target(x_out[name], batch_size, ensemble_size)

        return x_out

    def residual_connection(
        self, y: dict[str, torch.Tensor], x_skips: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for source, (target_idx, input_idx) in self.residual_connection_indices.items():
            # residual connection (just for the prognostic variables)
            y[source][..., target_idx] += x_skips[source][..., input_idx]

        return y

    def bound(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        y = {}
        for tensor_name, pred in x.items():
            for bounding in self.boundings:
                # bounding performed in the order specified in the config file
                pred = bounding(pred)

            y[tensor_name] = pred

        return y

    def forward(
        self, x: dict[str, Tensor], graph: HeteroData, *, model_comm_group: Optional[ProcessGroup] = None, **kwargs
    ) -> dict[str, Tensor]:
        batch_size = x[list(x.keys())[0]].shape[0]
        ensemble_size = 1

        x_hidden = self.node_attributes(self.hidden_name, batch_size=batch_size)
        # x_hidden = self.graph_node_embedder(
        #   graph[self.hidden_name], None, self.hidden_name, batch_size=batch_size
        # )
        shard_shapes_hidden = get_shard_shapes(x_hidden, 0, model_comm_group)

        x_data_latent, x_hidden_latent, x_data_skip = self.encode(
            (x, x_hidden),
            graph,
            shard_shapes_hidden=shard_shapes_hidden,
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        x_hidden_latent = self.merge_latents(x_hidden_latent)

        x_latent_proc = self.processor(
            x_hidden_latent,
            graph[(self.hidden_name, "to", self.hidden_name)].to(x_hidden_latent.device),
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        if self.latent_residual_connection:
            x_latent_proc = x_latent_proc + x_hidden_latent

        x_out = self.decode(
            (x_hidden_latent, x_data_latent, x),
            graph,
            shard_shapes_hidden=shard_shapes_hidden,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            model_comm_group=model_comm_group,
        )

        x_out = self.residual_connection(x_out, x_data_skip)
        x_out = self.bound(x_out)
        return x_out
