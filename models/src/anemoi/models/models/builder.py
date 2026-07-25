# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Configuration-driven builders for Anemoi models (object injection).

A :class:`ModelBuilder` reads the model configuration together with the runtime data
(data indices, statistics, graph) and builds every polymorphic sub-object — node
attributes, graph providers, encoder/processor/decoder, residual connections and output
boundings — then passes them as ordinary parameters to the model constructor. The model
classes themselves never read configuration nor call :func:`anemoi.utils.builder.build`;
they only store the injected sub-objects and run the forward pass.

Dispatch is by the model ``_target_``: it is resolved to a class and matched against each
concrete builder's :attr:`ModelBuilder.model_cls`.
"""

from __future__ import annotations

from typing import Any

from torch import nn
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.models.layers.bounding import build_boundings
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.models import BaseGraphModel
from anemoi.models.models.autoencoder import AnemoiModelAutoEncoder
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec
from anemoi.models.models.hierarchical import AnemoiModelEncProcDecHierarchical
from anemoi.models.models.hierarchical_autoencoder import AnemoiModelHierarchicalAutoEncoder
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportTendModelEncProcDec
from anemoi.models.utils.config import broadcast_config_keys
from anemoi.utils.builder import BuilderError
from anemoi.utils.builder import as_dict
from anemoi.utils.builder import build
from anemoi.utils.builder import locate
from anemoi.utils.config import DotDict


def _named_node_attributes_graph(graph_data: HeteroData, node_names: list[str]) -> HeteroData:
    """Build the reduced graph (coordinates + node counts) fed to ``NamedNodesAttributes``."""
    graph = HeteroData()
    for name in node_names:
        graph[name].x = graph_data[name].x
        graph[name].num_nodes = graph_data[name].num_nodes
    return graph


class ModelBuilder:
    """Base class: builds a :class:`BaseGraphModel` from configuration via injection.

    Concrete subclasses set :attr:`model_cls` (the model class they build) and implement
    :meth:`build_networks`, returning the variant-specific sub-modules that are forwarded
    as keyword arguments to the model constructor. The registry is keyed by the model
    class, so any config ``_target_`` (full path or a re-export alias) that resolves to it
    selects the right builder.
    """

    registry: dict[type[BaseGraphModel], type["ModelBuilder"]] = {}
    model_cls: type[BaseGraphModel] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.model_cls is not None:
            ModelBuilder.registry[cls.model_cls] = cls

    def __init__(
        self,
        model_config: Any,
        *,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        n_step_input: int,
        n_step_output: int,
    ) -> None:
        self.model_config = model_config
        self.data_indices = data_indices
        self.statistics = statistics
        self.graph_data = graph_data
        self.n_step_input = n_step_input
        self.n_step_output = n_step_output

        self.dataset_names = list(data_indices.keys())
        self.hidden_nodes_name = model_config.model.model.hidden_nodes_name
        self.hidden_names = BaseGraphModel._as_hidden_node_names(self.hidden_nodes_name)
        self.num_channels = model_config.model.num_channels
        self.latent_skip = model_config.model.model.latent_skip
        self.node_attributes = self.build_node_attributes()

    # -- shared sub-object builders -------------------------------------------
    def build_node_attributes(self) -> NamedNodesAttributes:
        trainable_parameters = broadcast_config_keys(
            self.model_config.model.trainable_parameters,
            data=self.dataset_names,
            hidden=self.hidden_nodes_name,
        )
        graph = _named_node_attributes_graph(self.graph_data, self.dataset_names + self.hidden_names)
        return NamedNodesAttributes(trainable_parameters, graph)

    def build_residual(self) -> nn.ModuleDict:
        residual = nn.ModuleDict()
        fused = uses_fused_dataset_graph(self.graph_data, self.dataset_names)
        num_chunks = self.model_config.model.get("sparse_projector", {}).get("num_chunks", 1)
        for dataset_name in self.dataset_names:
            data_node_name = dataset_name if fused else DEFAULT_DATASET_NAME
            residual[dataset_name] = build(
                self.model_config.model.residual,
                graph=self.graph_data,
                data_node_name=data_node_name,
                statistics=self.statistics[dataset_name],
                data_indices=self.data_indices[dataset_name],
                dataset_name=dataset_name,
                sparse_projector_num_chunks=num_chunks,
            )
        return residual

    def build_boundings(self) -> nn.ModuleDict:
        return build_boundings(self.model_config, self.data_indices, self.statistics)

    def create_graph_provider(self, config: Any, source: str, target: str) -> nn.Module:
        return create_graph_provider(
            graph=self.graph_data[(source, "to", target)],
            edge_attributes=config.get("sub_graph_edge_attributes"),
            src_size=self.node_attributes.num_nodes[source],
            dst_size=self.node_attributes.num_nodes[target],
            trainable_size=config.get("trainable_size", 0),
        )

    # -- dimensions (target_dim is overridable, e.g. by the auto-encoder) ------
    def num_input_channels(self, dataset_name: str) -> int:
        return len(self.data_indices[dataset_name].model.input)

    def num_output_channels(self, dataset_name: str) -> int:
        return len(self.data_indices[dataset_name].model.output)

    def input_dim(self, dataset_name: str) -> int:
        return self.n_step_input * self.num_input_channels(dataset_name) + self.node_attributes.attr_ndims[dataset_name]

    def input_dim_latent(self) -> int:
        return self.node_attributes.attr_ndims[self.hidden_names[0]]

    def target_dim(self, dataset_name: str) -> int:
        return self.input_dim(dataset_name)

    def output_dim(self, dataset_name: str) -> int:
        return self.n_step_output * self.num_output_channels(dataset_name)

    # -- assembly --------------------------------------------------------------
    def base_kwargs(self) -> dict:
        return {
            "node_attributes": self.node_attributes,
            "residual": self.build_residual(),
            "boundings": self.build_boundings(),
            "data_indices": self.data_indices,
            "statistics": self.statistics,
            "n_step_input": self.n_step_input,
            "n_step_output": self.n_step_output,
            "graph_data": self.graph_data,
            "hidden_nodes_name": self.hidden_nodes_name,
            "num_channels": self.num_channels,
            "latent_skip": self.latent_skip,
        }

    def build_networks(self) -> dict:
        """Return the variant-specific built sub-modules as constructor kwargs."""
        raise NotImplementedError

    def build(self) -> BaseGraphModel:
        return self.model_cls(**self.build_networks(), **self.base_kwargs())


class AnemoiModelEncProcDecBuilder(ModelBuilder):
    """Builder for the standard encoder/processor/decoder model."""

    model_cls = AnemoiModelEncProcDec

    def build_networks(self) -> dict:
        hidden = self.hidden_nodes_name
        encoder_cfg = self.model_config.model.encoder
        processor_cfg = self.model_config.model.processor
        decoder_cfg = self.model_config.model.decoder

        encoder_graph_provider = nn.ModuleDict()
        encoder = nn.ModuleDict()
        for dataset_name in self.dataset_names:
            provider = self.create_graph_provider(encoder_cfg, dataset_name, hidden)
            encoder_graph_provider[dataset_name] = provider
            encoder[dataset_name] = build(
                encoder_cfg,
                _recursive_=False,  # the encoder builds its own layer_kernels
                in_channels_src=self.input_dim(dataset_name),
                in_channels_dst=self.input_dim_latent(),
                hidden_dim=self.num_channels,
                edge_dim=provider.edge_dim,
            )

        processor_graph_provider = self.create_graph_provider(processor_cfg, hidden, hidden)
        processor = build(
            processor_cfg,
            _recursive_=False,
            num_channels=self.num_channels,
            edge_dim=processor_graph_provider.edge_dim,
        )

        decoder_graph_provider = nn.ModuleDict()
        decoder = nn.ModuleDict()
        for dataset_name in self.dataset_names:
            provider = self.create_graph_provider(decoder_cfg, hidden, dataset_name)
            decoder_graph_provider[dataset_name] = provider
            decoder[dataset_name] = build(
                decoder_cfg,
                _recursive_=False,
                in_channels_src=self.num_channels,
                in_channels_dst=self.target_dim(dataset_name),
                hidden_dim=self.num_channels,
                out_channels_dst=self.output_dim(dataset_name),
                edge_dim=provider.edge_dim,
            )

        return {
            "encoder": encoder,
            "processor": processor,
            "decoder": decoder,
            "encoder_graph_provider": encoder_graph_provider,
            "processor_graph_provider": processor_graph_provider,
            "decoder_graph_provider": decoder_graph_provider,
        }


class AnemoiModelAutoEncoderBuilder(AnemoiModelEncProcDecBuilder):
    """Builder for the auto-encoder: identical wiring but a forcing-based decoder input dim."""

    model_cls = AnemoiModelAutoEncoder

    def _num_decoding_forcings(self, dataset_name: str) -> int:
        data_indices = self.data_indices[dataset_name]
        return len([data_indices.name_to_index[name] for name in data_indices.model._forcing])

    def target_dim(self, dataset_name: str) -> int:
        return (
            self.n_step_output * self._num_decoding_forcings(dataset_name)
            + self.node_attributes.attr_ndims[dataset_name]
        )


class AnemoiEnsModelEncProcDecBuilder(AnemoiModelEncProcDecBuilder):
    """Builder for the ensemble model: adds a noise injector and a wider encoder input."""

    model_cls = AnemoiEnsModelEncProcDec

    @property
    def condition_on_residual(self) -> bool:
        return self.model_config.model.condition_on_residual

    def _num_input_channels_prognostic(self, dataset_name: str) -> int:
        return len(self.data_indices[dataset_name].model.input.prognostic)

    def input_dim(self, dataset_name: str) -> int:
        dim = super().input_dim(dataset_name) + 1  # for forecast step (fcstep)
        if self.condition_on_residual:
            dim += self._num_input_channels_prognostic(dataset_name)
        return dim

    def build_networks(self) -> dict:
        networks = super().build_networks()
        networks["noise_injector"] = build(
            self.model_config.model.noise_injector,
            _recursive_=False,
            num_channels=self.num_channels,
            graph_data=self.graph_data,
            sparse_projector_num_chunks=self.model_config.model.get("sparse_projector", {}).get("num_chunks", 1),
        )
        networks["condition_on_residual"] = self.condition_on_residual
        return networks


class AnemoiModelEncProcDecHierarchicalBuilder(AnemoiModelEncProcDecBuilder):
    """Builder for the hierarchical model (multiple hidden levels)."""

    model_cls = AnemoiModelEncProcDecHierarchical

    @property
    def hidden_dims(self) -> dict:
        return {hidden: self.num_channels * (2**i) for i, hidden in enumerate(self.hidden_names)}

    def build_networks(self) -> dict:
        hidden = self.hidden_names
        num_hidden = len(hidden)
        hidden_dims = self.hidden_dims
        encoder_cfg = self.model_config.model.encoder
        processor_cfg = self.model_config.model.processor
        decoder_cfg = self.model_config.model.decoder

        # Encoder: data -> hidden[0]
        encoder_graph_provider = nn.ModuleDict()
        encoder = nn.ModuleDict()
        for dataset_name in self.dataset_names:
            provider = self.create_graph_provider(encoder_cfg, dataset_name, hidden[0])
            encoder_graph_provider[dataset_name] = provider
            encoder[dataset_name] = build(
                encoder_cfg,
                _recursive_=False,
                in_channels_src=self.input_dim(dataset_name),
                in_channels_dst=self.input_dim_latent(),
                hidden_dim=hidden_dims[hidden[0]],
                edge_dim=provider.edge_dim,
            )

        level_process = self.model_config.model.enable_hierarchical_level_processing
        networks: dict = {
            "encoder": encoder,
            "encoder_graph_provider": encoder_graph_provider,
            "level_process": level_process,
        }

        if level_process:
            num_layers = self.model_config.model.level_process_num_layers
            down_processor = nn.ModuleDict()
            down_provider = nn.ModuleDict()
            up_processor = nn.ModuleDict()
            up_provider = nn.ModuleDict()
            for i in range(num_hidden - 1):
                name = hidden[i]
                dp = self.create_graph_provider(processor_cfg, name, name)
                down_provider[name] = dp
                down_processor[name] = build(
                    processor_cfg,
                    _recursive_=False,
                    num_channels=hidden_dims[name],
                    edge_dim=dp.edge_dim,
                    num_layers=num_layers,
                )
                up = self.create_graph_provider(processor_cfg, name, name)
                up_provider[name] = up
                up_processor[name] = build(
                    processor_cfg,
                    _recursive_=False,
                    num_channels=hidden_dims[name],
                    edge_dim=up.edge_dim,
                    num_layers=num_layers,
                )
            networks.update(
                down_level_processor=down_processor,
                down_level_processor_graph_providers=down_provider,
                up_level_processor=up_processor,
                up_level_processor_graph_providers=up_provider,
            )

        # Main processor at the deepest level
        deepest = hidden[num_hidden - 1]
        processor_graph_provider = self.create_graph_provider(processor_cfg, deepest, deepest)
        networks["processor_graph_provider"] = processor_graph_provider
        networks["processor"] = build(
            processor_cfg,
            _recursive_=False,
            num_channels=hidden_dims[deepest],
            edge_dim=processor_graph_provider.edge_dim,
        )

        # Downscale mappers (encoder re-used between adjacent levels going down)
        downscale = nn.ModuleDict()
        downscale_provider = nn.ModuleDict()
        for i in range(num_hidden - 1):
            src, dst = hidden[i], hidden[i + 1]
            provider = self.create_graph_provider(encoder_cfg, src, dst)
            downscale_provider[src] = provider
            downscale[src] = build(
                encoder_cfg,
                _recursive_=False,
                in_channels_src=hidden_dims[src],
                in_channels_dst=self.node_attributes.attr_ndims[dst],
                hidden_dim=hidden_dims[dst],
                edge_dim=provider.edge_dim,
            )
        networks.update(downscale=downscale, downscale_graph_providers=downscale_provider)

        # Upscale mappers (decoder re-used between adjacent levels going up)
        upscale = nn.ModuleDict()
        upscale_provider = nn.ModuleDict()
        for i in range(1, num_hidden):
            src, dst = hidden[i], hidden[i - 1]
            provider = self.create_graph_provider(decoder_cfg, src, dst)
            upscale_provider[src] = provider
            upscale[src] = build(
                decoder_cfg,
                _recursive_=False,
                in_channels_src=hidden_dims[src],
                in_channels_dst=hidden_dims[dst],
                hidden_dim=hidden_dims[src],
                out_channels_dst=hidden_dims[dst],
                edge_dim=provider.edge_dim,
            )
        networks.update(upscale=upscale, upscale_graph_providers=upscale_provider)

        # Decoder: hidden[0] -> data
        decoder_graph_provider = nn.ModuleDict()
        decoder = nn.ModuleDict()
        for dataset_name in self.dataset_names:
            provider = self.create_graph_provider(decoder_cfg, hidden[0], dataset_name)
            decoder_graph_provider[dataset_name] = provider
            decoder[dataset_name] = build(
                decoder_cfg,
                _recursive_=False,
                in_channels_src=hidden_dims[hidden[0]],
                in_channels_dst=self.input_dim(dataset_name),
                hidden_dim=hidden_dims[hidden[0]],
                out_channels_dst=self.output_dim(dataset_name),
                edge_dim=provider.edge_dim,
            )
        networks.update(decoder=decoder, decoder_graph_provider=decoder_graph_provider)

        return networks


class AnemoiTransportModelEncProcDecBuilder(AnemoiModelEncProcDecBuilder):
    """Builder for the transport (diffusion/bridge) model."""

    model_cls = AnemoiTransportModelEncProcDec

    @property
    def transport_params(self) -> Any:
        return self.model_config.model.model.transport

    def input_dim(self, dataset_name: str) -> int:
        # input history plus corrupted target
        return super().input_dim(dataset_name) + self.output_dim(dataset_name)

    def build_networks(self) -> dict:
        networks = super().build_networks()
        networks["noise_embedder"] = build(self.transport_params.noise_embedder)
        networks["transport_params"] = self.transport_params
        return networks


class AnemoiTransportTendModelEncProcDecBuilder(AnemoiTransportModelEncProcDecBuilder):
    """Builder for the tendency-predicting transport model."""

    model_cls = AnemoiTransportTendModelEncProcDec

    @property
    def condition_on_residual(self) -> bool:
        return self.model_config.model.condition_on_residual

    def input_dim(self, dataset_name: str) -> int:
        dim = super().input_dim(dataset_name)
        if self.condition_on_residual:
            dim += len(self.data_indices[dataset_name].model.input.prognostic) * self.n_step_output
        return dim

    def build_networks(self) -> dict:
        networks = super().build_networks()
        networks["condition_on_residual"] = self.condition_on_residual
        return networks


class AnemoiModelHierarchicalAutoEncoderBuilder(AnemoiModelEncProcDecBuilder):
    """Builder for the hierarchical auto-encoder (hierarchical, no main processor)."""

    model_cls = AnemoiModelHierarchicalAutoEncoder

    @property
    def hidden_dims(self) -> dict:
        return {hidden: self.num_channels * (2**i) for i, hidden in enumerate(self.hidden_names)}

    def build_node_attributes(self) -> NamedNodesAttributes:
        # Matches the original hierarchical auto-encoder: node attributes from the full graph,
        # without ``broadcast_config_keys``.
        return NamedNodesAttributes(self.model_config.model.trainable_parameters, self.graph_data)

    def _num_decoding_forcings(self, dataset_name: str) -> int:
        data_indices = self.data_indices[dataset_name]
        return len([data_indices.name_to_index[name] for name in data_indices.model._forcing])

    def target_dim(self, dataset_name: str) -> int:
        return (
            self.n_step_output * self._num_decoding_forcings(dataset_name)
            + self.node_attributes.attr_ndims[dataset_name]
        )

    def build_networks(self) -> dict:
        hidden = self.hidden_names
        num_hidden = len(hidden)
        hidden_dims = self.hidden_dims
        encoder_cfg = self.model_config.model.encoder
        processor_cfg = self.model_config.model.processor
        decoder_cfg = self.model_config.model.decoder

        # Encoder: data -> hidden[0]
        encoder_graph_provider = nn.ModuleDict()
        encoder = nn.ModuleDict()
        for dataset_name in self.dataset_names:
            provider = self.create_graph_provider(encoder_cfg, dataset_name, hidden[0])
            encoder_graph_provider[dataset_name] = provider
            encoder[dataset_name] = build(
                encoder_cfg,
                _recursive_=False,
                in_channels_src=self.input_dim(dataset_name),
                in_channels_dst=self.input_dim_latent(),
                hidden_dim=hidden_dims[hidden[0]],
                edge_dim=provider.edge_dim,
            )

        level_process = self.model_config.model.enable_hierarchical_level_processing
        networks: dict = {
            "encoder": encoder,
            "encoder_graph_provider": encoder_graph_provider,
            "level_process": level_process,
        }
        if level_process:
            num_layers = self.model_config.model.level_process_num_layers
            down_processor = nn.ModuleDict()
            down_provider = nn.ModuleDict()
            up_processor = nn.ModuleDict()
            up_provider = nn.ModuleDict()
            for i in range(num_hidden - 1):
                name = hidden[i]
                dp = self.create_graph_provider(processor_cfg, name, name)
                down_provider[name] = dp
                down_processor[name] = build(
                    processor_cfg,
                    _recursive_=False,
                    num_channels=hidden_dims[name],
                    edge_dim=dp.edge_dim,
                    num_layers=num_layers,
                )
                up = self.create_graph_provider(processor_cfg, name, name)
                up_provider[name] = up
                up_processor[name] = build(
                    processor_cfg,
                    _recursive_=False,
                    num_channels=hidden_dims[name],
                    edge_dim=up.edge_dim,
                    num_layers=num_layers,
                )
            networks.update(
                down_level_processor=down_processor,
                down_level_processor_graph_providers=down_provider,
                up_level_processor=up_processor,
                up_level_processor_graph_providers=up_provider,
            )

        # Downscale (encoder re-used going down)
        downscale = nn.ModuleDict()
        downscale_provider = nn.ModuleDict()
        for i in range(num_hidden - 1):
            src, dst = hidden[i], hidden[i + 1]
            provider = self.create_graph_provider(encoder_cfg, src, dst)
            downscale_provider[src] = provider
            downscale[src] = build(
                encoder_cfg,
                _recursive_=False,
                in_channels_src=hidden_dims[src],
                in_channels_dst=self.node_attributes.attr_ndims[dst],
                hidden_dim=hidden_dims[dst],
                edge_dim=provider.edge_dim,
            )
        networks.update(downscale=downscale, downscale_graph_providers=downscale_provider)

        # Upscale (decoder re-used going up)
        upscale = nn.ModuleDict()
        upscale_provider = nn.ModuleDict()
        for i in range(1, num_hidden):
            src, dst = hidden[i], hidden[i - 1]
            provider = self.create_graph_provider(decoder_cfg, src, dst)
            upscale_provider[src] = provider
            upscale[src] = build(
                decoder_cfg,
                _recursive_=False,
                in_channels_src=hidden_dims[src],
                in_channels_dst=hidden_dims[dst],
                hidden_dim=hidden_dims[src],
                out_channels_dst=hidden_dims[dst],
                edge_dim=provider.edge_dim,
            )
        networks.update(upscale=upscale, upscale_graph_providers=upscale_provider)

        # Decoder: hidden[0] -> data (auto-encoder: forcing-based target dim)
        decoder_graph_provider = nn.ModuleDict()
        decoder = nn.ModuleDict()
        for dataset_name in self.dataset_names:
            provider = self.create_graph_provider(decoder_cfg, hidden[0], dataset_name)
            decoder_graph_provider[dataset_name] = provider
            decoder[dataset_name] = build(
                decoder_cfg,
                _recursive_=False,
                in_channels_src=hidden_dims[hidden[0]],
                in_channels_dst=self.target_dim(dataset_name),
                hidden_dim=hidden_dims[hidden[0]],
                out_channels_dst=self.output_dim(dataset_name),
                edge_dim=provider.edge_dim,
            )
        networks.update(decoder=decoder, decoder_graph_provider=decoder_graph_provider)

        return networks


def build_model(
    model_config: Any,
    *,
    data_indices: dict,
    statistics: dict,
    graph_data: HeteroData,
    n_step_input: int,
    n_step_output: int,
) -> BaseGraphModel:
    """Build a model from ``model_config`` by dispatching on its ``_target_``.

    ``model_config`` may be an OmegaConf ``DictConfig``, a ``DotDict``, or a plain (e.g.
    JSON-loaded) ``dict`` — it is normalised so a serialised recipe round-trips.
    """
    recipe = as_dict(model_config)  # plain, JSON-able containers (interpolations resolved)
    config = DotDict(recipe)  # attribute access for the builders
    target = config.model.model["_target_"]
    model_cls = locate(target)
    builder_cls = ModelBuilder.registry.get(model_cls)
    if builder_cls is None:
        raise BuilderError(
            f"No ModelBuilder registered for model {target!r} ({model_cls}). "
            f"Available: {sorted(c.__name__ for c in ModelBuilder.registry)}"
        )
    model = builder_cls(
        config,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
        n_step_input=n_step_input,
        n_step_output=n_step_output,
    ).build()
    # Record the architecture recipe so the model can be serialised with
    # ``anemoi.utils.builder.to_dict`` and rebuilt with ``build_model`` (given the runtime
    # context: graph, data indices, statistics).
    model.__anemoi_spec__ = recipe
    return model
