# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from anemoi.graphs.bundle import GraphBundle


class GraphProviderRegistry(nn.Module):
    """Registry for graph providers built from graph bundles + specs."""

    def __init__(self, graph_bundles: dict[str, GraphBundle], provider_specs: dict[str, Any] | None) -> None:
        """Initialize the registry with graph bundles and provider specs."""
        super().__init__()
        if provider_specs is None:
            raise ValueError("graph.providers must be defined in the config.")
        if OmegaConf.is_config(provider_specs) and len(provider_specs) == 0:
            raise ValueError("graph.providers must be defined in the config.")
        if isinstance(provider_specs, dict) and len(provider_specs) == 0:
            raise ValueError("graph.providers must be defined in the config.")

        self._graph_bundles = graph_bundles
        self._provider_specs = provider_specs
        self._providers = nn.ModuleDict()

    def get(self, name: str, dataset_name: str, subkey: str | None = None) -> nn.Module:
        """Return a provider instance by name for a given dataset."""
        spec = self._get_spec(name, subkey)
        provider_key = name if subkey is None else f"{name}__{subkey}"

        if dataset_name not in self._providers:
            self._providers[dataset_name] = nn.ModuleDict()

        dataset_providers = self._providers[dataset_name]
        if provider_key not in dataset_providers:
            dataset_providers[provider_key] = self._build_provider(spec, dataset_name)

        return dataset_providers[provider_key]

    def has(self, name: str) -> bool:
        """Return True if a provider with this name exists in the specs."""
        return name in self._provider_specs

    def _get_spec(self, name: str, subkey: str | None) -> dict[str, Any]:
        """Resolve a provider spec, handling OmegaConf and grouped specs."""
        spec = self._provider_specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown graph provider: {name}")

        if OmegaConf.is_config(spec):
            spec = OmegaConf.to_container(spec, resolve=True)

        if isinstance(spec, dict) and "_target_" in spec:
            return dict(spec)

        if subkey is None:
            raise KeyError(f"Graph provider group '{name}' requires subkey.")

        nested = spec.get(subkey)
        if nested is None:
            raise KeyError(f"Unknown graph provider: {name}.{subkey}")

        if OmegaConf.is_config(nested):
            nested = OmegaConf.to_container(nested, resolve=True)

        if not isinstance(nested, dict):
            raise TypeError(f"Graph provider spec for {name}.{subkey} must be a mapping.")

        return dict(nested)

    def _build_provider(self, spec: dict[str, Any], dataset_name: str) -> nn.Module:
        """Instantiate a provider, injecting graph/edge information as needed."""
        spec = dict(spec)
        if "_target_" not in spec:
            raise ValueError("Graph provider spec must include _target_.")

        graph_ref = spec.pop("graph_ref", "main")
        edges = spec.pop("edges", None)
        edges_name = spec.pop("edges_name", None)

        if edges is not None and edges_name is not None:
            raise ValueError("Graph provider spec must not define both edges and edges_name.")

        if edges is not None or edges_name is not None:
            graph = self._graph_bundles[dataset_name].get(graph_ref)
            if edges is not None:
                # `edges` is for providers that take an edge store (graph[edge_type]) directly
                # (e.g., StaticGraphProvider).
                edge_key = tuple(edges)
                if edge_key not in graph.edge_types:
                    raise KeyError(f"Unknown edge type in graph '{graph_ref}': {edge_key}")
                spec.setdefault("graph", graph[edge_key])
                spec.setdefault("src_size", graph[edge_key[0]].num_nodes)
                spec.setdefault("dst_size", graph[edge_key[2]].num_nodes)
            else:
                # `edges_name` is for providers that need the full graph + edge type name
                # (e.g., ProjectionGraphProvider).
                edges_key = tuple(edges_name)
                if edges_key not in graph.edge_types:
                    raise KeyError(f"Unknown edge type in graph '{graph_ref}': {edges_key}")
                spec.setdefault("graph", graph)
                spec.setdefault("edges_name", edges_key)

        return instantiate(spec, _convert_="all")
