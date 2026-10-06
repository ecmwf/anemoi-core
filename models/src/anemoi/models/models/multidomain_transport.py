# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

"""Paper-structured latent transport components for the multidomain campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
from anemoi.utils.config import DotDict

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.processor import GraphTransformerProcessor


class _PaperSinusoidalEmbeddings(torch.nn.Module):
    """Exact 32-channel transport-time embedding from the selected model."""

    def __init__(self, channels: int = 32, max_period: int = 1000) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("Sinusoidal embedding width must be even")
        half = channels // 2
        self.register_buffer(
            "frequencies",
            torch.exp(-torch.log(torch.tensor(float(max_period))) * torch.arange(half) / half),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        angles = value.float() * self.frequencies.to(value.device)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


def conditional_layer_kernels(width: int, condition_width: int) -> DotDict:
    return DotDict(
        {
            "LayerNorm": {
                "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
                "normalized_shape": width,
                "condition_shape": condition_width,
                "zero_init": True,
                "autocast": False,
            },
            "Linear": {"_target_": "torch.nn.Linear"},
            "Activation": {"_target_": "torch.nn.GELU"},
            "QueryNorm": {
                "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
                "bias": False,
            },
            "KeyNorm": {
                "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
                "bias": False,
            },
        }
    )


def _safe_key(value: str) -> str:
    return value.lower().replace("-", "_")


def _position_features(latlon_radians: torch.Tensor) -> torch.Tensor:
    latitude = latlon_radians[:, 0].float()
    longitude = latlon_radians[:, 1].float()
    return torch.stack(
        (
            torch.cos(latitude),
            torch.sin(longitude),
            torch.sin(latitude),
            torch.cos(longitude),
        ),
        dim=-1,
    )


class PaperStructuredMultidomainTransport(torch.nn.Module):
    """Velocity model retaining the selected paper transport mechanisms.

    Retrieval descriptors intentionally do not enter this network.  They choose
    an empirical source field; the velocity is conditioned on the current
    nodewise weather context and transport time.
    """

    def __init__(
        self,
        *,
        factors: int,
        context_factors: int,
        provenances: Sequence[str],
        graph_data: Mapping[str, object],
        width: int = 512,
        layers: int = 8,
        heads: int = 8,
        chunks: int = 2,
        condition_width: int = 16,
        learned_node_channels: int = 8,
        learned_edge_channels: int = 8,
        processor_context_scale: float = 32.0,
    ) -> None:
        super().__init__()
        if condition_width != 16:
            raise ValueError("The recovered paper transport uses condition width 16")
        self.factors = int(factors)
        self.context_factors = int(context_factors)
        self.condition_width = int(condition_width)
        self.processor_context_scale = float(processor_context_scale)
        self.provenances = tuple(provenances)
        self.provenance_to_index = {name: index for index, name in enumerate(self.provenances)}
        if set(graph_data) != set(self.provenances):
            raise ValueError("Transport graphs and provenances do not align")
        nodes = {int(graph_data[name]["hidden"].num_nodes) for name in self.provenances}
        if nodes != {100_000}:
            raise ValueError(f"Expected a common 100,000-node latent, got {nodes}")

        positions = torch.stack([_position_features(graph_data[name]["hidden"].x) for name in self.provenances])
        self.register_buffer("position_features", positions, persistent=True)
        self.learned_node_attributes = torch.nn.Parameter(
            torch.zeros(len(self.provenances), 100_000, learned_node_channels)
        )
        torch.nn.init.normal_(self.learned_node_attributes, std=0.02)

        self.initial_hidden_projection = torch.nn.Linear(
            self.factors + positions.shape[-1] + learned_node_channels,
            width,
        )
        self.mesh_context_projection = torch.nn.Linear(self.context_factors, width)
        self.context_fusion = torch.nn.Sequential(
            torch.nn.Linear(2 * width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.processor_context_condition_projection = torch.nn.Linear(width, self.condition_width)
        torch.nn.init.zeros_(self.processor_context_condition_projection.weight)
        torch.nn.init.zeros_(self.processor_context_condition_projection.bias)
        self.noise_embedder = _PaperSinusoidalEmbeddings(channels=32, max_period=1000)
        self.noise_cond_mlp = torch.nn.Sequential()
        self.noise_cond_mlp.add_module("linear1_no_gradscaling", torch.nn.Linear(32, 32))
        self.noise_cond_mlp.add_module("activation", torch.nn.SiLU())
        self.noise_cond_mlp.add_module("linear2_no_gradscaling", torch.nn.Linear(32, self.condition_width))

        providers = {}
        for provenance in self.provenances:
            graph = graph_data[provenance][("hidden", "to", "hidden")]
            providers[_safe_key(provenance)] = create_graph_provider(
                graph=graph,
                edge_attributes=("edge_length", "edge_dirs"),
                src_size=100_000,
                dst_size=100_000,
                trainable_size=learned_edge_channels,
            )
        self.graph_providers = torch.nn.ModuleDict(providers)
        edge_dimensions = {provider.edge_dim for provider in self.graph_providers.values()}
        if edge_dimensions != {3 + learned_edge_channels}:
            raise ValueError(f"Unexpected edge dimensions {edge_dimensions}")
        self.processor = GraphTransformerProcessor(
            num_layers=layers,
            num_channels=width,
            num_chunks=chunks,
            num_heads=heads,
            mlp_hidden_ratio=4,
            edge_dim=next(iter(edge_dimensions)),
            qk_norm=True,
            cpu_offload=False,
            gradient_checkpointing=True,
            layer_kernels=conditional_layer_kernels(width, self.condition_width),
            shard_strategy="edges",
            graph_attention_backend="triton",
        )
        self.factor_head = torch.nn.Linear(width, self.factors)
        self.residual_bias = torch.nn.Parameter(torch.zeros(len(self.provenances), 100_000, self.factors))

    @staticmethod
    def _local_slice(
        shard_sizes: Sequence[int] | None,
        model_comm_group,
    ) -> slice:
        if shard_sizes is None:
            return slice(None)
        group_rank = 0 if model_comm_group is None else dist.get_rank(model_comm_group)
        start = sum(int(value) for value in shard_sizes[:group_rank])
        return slice(start, start + int(shard_sizes[group_rank]))

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        flow_time: torch.Tensor,
        provenance: str,
        *,
        model_comm_group=None,
        node_shard_sizes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.factors:
            raise ValueError(f"Invalid state shape {tuple(state.shape)}")
        if context.shape != (state.shape[0], self.context_factors):
            raise ValueError("Current-weather context does not align with the latent state")
        domain = self.provenance_to_index[provenance]
        local = self._local_slice(node_shard_sizes, model_comm_group)
        attributes = torch.cat(
            (
                self.position_features[domain, local].to(state),
                self.learned_node_attributes[domain, local].to(state.dtype),
            ),
            dim=-1,
        )
        hidden = self.initial_hidden_projection(torch.cat((state, attributes), dim=-1))
        context_latent = self.mesh_context_projection(context)
        hidden = hidden + self.context_fusion(torch.cat((hidden, context_latent), dim=-1))
        condition = self.noise_cond_mlp(self.noise_embedder(flow_time)).expand(state.shape[0], -1)
        condition = condition + self.processor_context_scale * (
            self.processor_context_condition_projection(context_latent)
        )
        provider = self.graph_providers[_safe_key(provenance)]
        edge_attr, edge_index, edge_shards = provider.get_edges(
            batch_size=1,
            model_comm_group=model_comm_group,
        )
        processed = self.processor(
            x=hidden,
            batch_size=1,
            shard_info=GraphShardInfo(nodes=node_shard_sizes, edges=edge_shards),
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
            cond=condition,
        )
        processed = processed + hidden
        return self.factor_head(processed) + self.residual_bias[domain, local].to(processed.dtype)

    def transfer_paper_weights(self, paper_state: Mapping[str, torch.Tensor]) -> dict[str, object]:
        """Copy basis-independent weights from the selected paper transport."""

        current = self.state_dict()
        prefixes = (
            "processor.",
            "context_fusion.",
            "processor_context_condition_projection.",
            "noise_cond_mlp.",
            # These tensors are basis-dependent for the atmospheric model and
            # therefore fail the shape gate there.  The one-coordinate paper
            # precipitation model has the same input/output dimensions, so its
            # exact input projection and velocity head can be retained.
            "initial_hidden_projection.",
            "factor_head.",
        )
        copied = []
        for key, value in paper_state.items():
            if key in current and key.startswith(prefixes) and current[key].shape == value.shape:
                current[key] = value.detach().clone()
                copied.append(key)
        for key in (
            "initial_hidden_projection.bias",
            "mesh_context_projection.bias",
            "factor_head.bias",
            "noise_embedder.frequencies",
        ):
            value = paper_state.get(key)
            if value is not None and key in current and current[key].shape == value.shape:
                current[key] = value.detach().clone()
                copied.append(key)
        self.load_state_dict(current, strict=True)
        return {
            "copied_parameter_tensors": len(copied),
            "copied": sorted(copied),
            "reinitialised": [
                "basis-dependent input/context/output weights",
                "domain node attributes",
                "domain edge attributes",
                "domain residual bias",
            ],
        }
