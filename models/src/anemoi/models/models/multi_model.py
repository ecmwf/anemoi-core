# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from typing import Optional

import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.projection import NodeEmbedder
from anemoi.models.layers.projection import NodeProjector
from anemoi.training.data.refactor.structure import Dict

# from anemoi.models.preprocessing.normalisers import build_normaliser
from anemoi.utils.config import DotDict

from .base import AnemoiModel

LOGGER = logging.getLogger(__name__)


NODE_COORDS_NDIMS = 4  # cos_lat, sin_lat, cos_lon, sin_lon
EDGE_ATTR_NDIM = 3  # edge_length, edge_dir0, edge_dir1


def extract_sources(config, reversed: bool = False) -> tuple[dict, dict[str, list[str]]]:
    # TODO: remove useless argument?
    assert reversed is False

    mapper_config, sources, num_channels = {}, {}, {}
    for i, component in enumerate(config.copy()):
        name = component.pop("name", f"{i+1}")
        for source in component.pop("sources"):
            if reversed:
                sources[name] = source
            else:
                sources[source] = name
        mapper_config[name] = component["mapper"]
        num_channels[name] = component["num_channels"]
    return mapper_config, sources, num_channels


def merge_nodes(graph: HeteroData, merged_name: str, nodes_names: list[str]) -> tuple[HeteroData, dict]:
    if len(nodes_names) == 1:
        graph = graph.rename(nodes_names[0], merged_name)
        slices = {nodes_names[0]: slice(0, graph[nodes_names[0]].num_nodes)}
        return graph, slices

    num_nodes = {name: graph[name].num_nodes for name in nodes_names}
    graph[merged_name].x = torch.cat([graph[nodes].x for nodes in nodes_names])
    # TODO: Merge edge_index of all subgraphs
    slices, count = {}, 0
    for nodes in nodes_names:
        num_nodes = graph[nodes].num_nodes
        slices[nodes] = slice(count, count + num_nodes)
        count += num_nodes
    return graph, slices


def merge_graph_sources(graph: HeteroData, sources: dict[str, str]) -> HeteroData:
    if sources is None or len(sources) == 0:
        return graph, None

    graph = graph.clone()
    new = {}
    for k, v in sources.items():
        new[v] = (new[v] + [k]) if v in new else [k]

    slices = {}
    for new_node_name, old_node_names in new.items():
        graph, nodes_slices = merge_nodes(graph, new_node_name, old_node_names)
        slices[new_node_name] = nodes_slices

    return graph, slices


class AnemoiMultiModel(AnemoiModel):
    """Message passing graph neural network."""

    name = None

    def __init__(self, *, sample_static_info: "Structure", model_config: DotDict, metadata) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        graph_data : HeteroData
            Graph definition
        """
        super().__init__(model_config=model_config, sample_static_info=sample_static_info, metadata=metadata)
        print(f"✅ model : {self.__class__.__name__}")

        from anemoi.models.preprocessing.normalisers import build_normaliser

        self.normaliser = self.sample_static_info.map_expanded(build_normaliser).as_module_dict()
        print("Normaliser for full batch", self.normaliser)

        self.supporting_arrays = {}
        self.build()

    def build(self):
        input_info = self.sample_static_info["input"]
        target_info = self.sample_static_info["target"]

        print(type(input_info), input_info)
        assert isinstance(input_info, Dict), type(input_info)

        self.num_channels = self.model_config.num_channels

        self.latent_residual_connection = True
        self.merge_latents_method = self.model_config.model.merge_latents

        if self.model_config.get("residual_connections"):
            warnings.warn("Residual connections not supported")
        # def _define_residual_connection_indices(
        #     input,
        #     target,
        #     residual_connections: list[str],
        # ):
        #     residual_connection_indices = {}
        #     for source in residual_connections:
        #         input_vars = input_name_to_index[source]
        #         target_vars = target_name_to_index[source]
        #         common_vars = set(target_vars).intersection(input_vars)
        #         target_idx = [input_vars[v] for v in common_vars]
        #         input_idx = [target_vars[v] for v in common_vars]
        #         residual_connection_indices[source] = target_idx, input_idx
        #     return residual_connection_indices

        # self.residual_connection_indices = _define_residual_connection_indices(
        #     self.sample_static_info["input"],
        #     self.sample_static_info["target"],
        #     residual_connections=self.model_config.get("residual_connections", []),
        # )

        self.num_input_channels = {k: v["number_of_features"] for k, v in self.sample_static_info["input"].items()}
        self.num_target_channels = {k: v["number_of_features"] for k, v in self.sample_static_info["target"].items()}
        input_dims_order = {k: ("batch",) + v["dimensions_order"] for k, v in self.sample_static_info["input"].items()}
        target_dims_order = {
            k: ("batch",) + v["dimensions_order"] for k, v in self.sample_static_info["target"].items()
        }

        self.hidden_name: str = self.model_config.model.hidden_name
        encoders, self.encoder_sources, num_encoded_channels = extract_sources(self.model_config.model.encoders)
        decoders, self.decoder_sources, num_decoded_channels = extract_sources(self.model_config.model.decoders)
        # def build_embeder(number_of_features, **kwargs):
        #     return NodeEmbedder(
        #         num_input_channels=number_of_features,
        #         # TODO
        #     )
        # self.embeders = self.sample_static_info.create_function(build_embeder)
        # Embedding layers

        # def build_embeder(number_of_features, **kwargs):
        #         return dict(
        #             num_input_channels=number_of_features,
        #             # TODO
        #         )
        # self.embeders = self.sample_static_info.create_function(build_embeder)

        num_embedded_channels = {
            key: num_encoded_channels[self.encoder_sources[key]] for key in self.num_input_channels.keys()
        }
        self.node_embedder = NodeEmbedder(
            self.model_config.model.emb_data,
            num_input_channels=self.num_input_channels,
            num_output_channels=num_embedded_channels,
            dimensions_order=input_dims_order,
        )
        num_embedded_channels = {
            key: num_decoded_channels[self.decoder_sources[key]] for key in self.num_target_channels.keys()
        }
        self.unemb_target_nodes = NodeProjector(
            self.model_config.model.emb_data,
            num_input_channels=num_embedded_channels,
            num_output_channels=self.num_target_channels,
            dimensions_order=target_dims_order,
        )

        # Encoders: ??? -> hidden
        self.encoders = nn.ModuleDict({})
        for enc_name, encoder_config in encoders.items():
            self.encoders[enc_name] = instantiate(
                encoder_config,
                _recursive_=False,
                in_channels_src=num_encoded_channels[enc_name],
                in_channels_dst=NODE_COORDS_NDIMS,
                hidden_dim=num_encoded_channels[enc_name],
                edge_dim=EDGE_ATTR_NDIM,
            )

        # Processor hidden -> hidden
        self.processor = instantiate(
            self.model_config.model.processor,
            _recursive_=False,
            num_channels=self.num_channels,
            edge_dim=EDGE_ATTR_NDIM,
        )

        # Decoders: hidden -> ???
        self.decoders = nn.ModuleDict({})
        for dec_name, decoder_config in decoders.items():
            self.decoders[dec_name] = instantiate(
                decoder_config,
                _recursive_=False,
                in_channels_src=self.num_channels,
                in_channels_dst=num_decoded_channels[dec_name],
                hidden_dim=num_decoded_channels[dec_name],
                out_channels_dst=num_decoded_channels[dec_name],
                edge_dim=EDGE_ATTR_NDIM,
            )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        # TODO: Bring bounding back
        self.boundings = nn.ModuleList([])
        # self.boundings = nn.ModuleList(
        #    [
        #        instantiate(
        #            cfg,
        #            name_to_index=sample_static_info.target.name_to_index,
        #            statistics=sample_static_info.input.statistics,
        #            name_to_index_stats=sample_static_info.input.name_to_index,
        #        )
        #        for cfg in getattr(model_config.model, "bounding", [])
        #    ]
        # )

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
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        x_data_latents, x_hidden_raw = x

        # do we need to have the batch size in argument ?
        shard_shapes_hidden = get_shard_shapes(x_hidden_raw, 0, model_comm_group)

        x_hidden_latents = {}
        for source, x_data_latent in x_data_latents.items():
            if source == self.hidden_name:
                continue

            encoder = self.encoders[self.encoder_sources[source]]

            _, x_hidden_latents[source] = self._run_mapper(
                encoder,
                (x_data_latent["data"], x_hidden_raw),
                sub_graph=graph[(source, "to", self.hidden_name)].to(x_data_latent["data"].device),
                batch_size=batch_size,
                shard_shapes=(x_data_latent["shard_shapes"], shard_shapes_hidden),
                model_comm_group=x_data_latent["model_comm_group"],
            )

        x_hidden_latents = self.merge_latents(x_hidden_latents)

        return x_data_latents, x_hidden_latents, shard_shapes_hidden

    def merge_latents(self, latents: dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: implement different strategies: sum, average, concat, learnable, ...
        last = list(latents.keys())[-1]
        return latents[last]

    def decode(
        self,
        x: tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]],
        graph: HeteroData,
        shard_shapes_hidden: tuple[list],
        batch_size: int,
        ensemble_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ):
        x_hidden_latent, x_target_latents = x

        # sources = {v: k for k, v in self.decoder_sources.items()}
        # graph, node_slices = merge_graph_sources(graph, sources)

        x_out = self.sample_static_info["target"].new_empty()
        for dec_source, x_data_latent in x_target_latents.items():
            if dec_source not in self.sample_static_info["target"]:
                continue

            dec_name = self.decoder_sources[dec_source]

            x_out[dec_source] = self._run_mapper(
                self.decoders[dec_name],
                (x_hidden_latent, x_data_latent["data"]),
                sub_graph=graph[(self.hidden_name, "to", dec_source)].to(x_hidden_latent.device),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, x_data_latent["shard_shapes"]),
                model_comm_group=x_data_latent["model_comm_group"],
            )

        x_out = self.unemb_target_nodes(x_out, batch_size=batch_size)

        return x_out

    def residual_connection(
        self, y: dict[str, torch.Tensor], x_skips: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for source, (target_idx, input_idx) in self.residual_connection_indices.items():
            # residual connection (just for the prognostic variables)
            y[source][..., target_idx] += x_skips[source][..., input_idx]

        return y

    def bound(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        y = x.new_empty()
        for tensor_name, pred in x.items():
            for bounding in self.boundings:
                # bounding performed in the order specified in the config file
                pred = bounding(pred)

            y[tensor_name] = pred

        return y

    def _assemble_input(self, x_data_raw: dict[str, torch.Tensor], graph: HeteroData, batch_size: int, model_comm_group):
        #assert batch_size == x_data_raw.first["data"].shape[0], (batch_size, x_data_raw.first["data"].shape)

        x_data_latents = self.node_embedder(x_data_raw, batch_size=batch_size)

        x_data_latents = x_data_latents.wrap("data")
        for x_data_latent in x_data_latents.values():
            x_data_latent["shard_shapes"] = get_shard_shapes(x_data_latent["data"], 0, model_comm_group)
            x_data_latent["model_comm_group"] = model_comm_group

        return x_data_latents

    def get_node_coords(self, graph: HeteroData, name: str) -> torch.Tensor:
        assert name in graph.node_types, f"{name} do not exist. Valid graph nodes: {graph.node_types}."
        #        assert number_dim(graph[name].x) == (n, 2)
        return torch.cat([torch.sin(graph[name].x), torch.cos(graph[name].x)], dim=-1)

    def forward(
        self, x: dict[str, Tensor], graph: HeteroData, *, model_comm_group: Optional[ProcessGroup] = None, **kwargs
    ) -> dict[str, Tensor]:
        # at this point, the input (x) has already been normalised
        # if this is not wanted, don't normalise it in the task
        batch_size = 1
        ensemble_size = 1
        from anemoi.training.data.refactor.structure import Dict

        assert isinstance(x, Dict), type(x)

        x_data_latents = self._assemble_input(x, graph, batch_size=batch_size, model_comm_group=model_comm_group)
        
        x_hidden_raw = self.get_node_coords(graph, self.hidden_name)

        x_data_latents, x_hidden_latent, shard_shapes_hidden = self.encode(
            (x_data_latents, x_hidden_raw),
            graph,
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

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
            (x_latent_proc, x_data_latents),
            graph,
            shard_shapes_hidden=shard_shapes_hidden,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            model_comm_group=model_comm_group,
        )

        #x_out = self.residual_connection(x_out, x_data_skip)
        x_out = self.bound(x_out)
        return x_out.wrap("data")

    def apply_normalisers(self, batch):
        return self.normaliser(batch)
