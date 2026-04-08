# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Projection metadata resolved from a built graph and its projection config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

from omegaconf import OmegaConf

from anemoi.graphs.projection_helpers import DEFAULT_EDGE_WEIGHT_ATTRIBUTE
from anemoi.graphs.projection_helpers import multiscale_loss_matrices_graph
from anemoi.graphs.projection_helpers import residual_projection_edge_names

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


@dataclass
class ProjectionData:
    """Resolved projection metadata produced by ProjectionCreator.

    Consumed by model layers (TruncatedConnection) and loss wrappers
    (MultiscaleLossWrapper) so they receive pre-resolved names rather than
    performing their own runtime graph lookups.

    Attributes
    ----------
    truncation_down_edges_name:
        Concrete ``(src, relation, dst)`` edge tuple for the data→truncation
        down-projection. ``None`` when truncation is not configured.
    truncation_up_edges_name:
        Concrete ``(src, relation, dst)`` edge tuple for the truncation→data
        up-projection. ``None`` when truncation is not configured.
    truncation_edge_weight_attribute:
        Name of the edge attribute used as projection weights.
    multiscale_loss_matrices_graph:
        List of graph-edge references for multiscale smoothing matrices,
        one entry per scale (``None`` entries signal no smoothing).
        ``None`` when multiscale is not configured.
    """

    truncation_down_edges_name: tuple[str, str, str] | None = None
    truncation_up_edges_name: tuple[str, str, str] | None = None
    truncation_edge_weight_attribute: str = DEFAULT_EDGE_WEIGHT_ATTRIBUTE
    multiscale_loss_matrices_graph: list[dict | None] | None = field(default=None)


class ProjectionCreator:
    """Resolve concrete projection metadata from config and a built graph.

    Mirrors ``GraphCreator``: takes a projections config, produces a typed
    ``ProjectionData`` object. Called once after graph creation so that model
    layers and loss wrappers receive pre-resolved names rather than performing
    their own runtime graph lookups.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def create(self, graph_data: HeteroData, dataset_names: list[str]) -> ProjectionData:
        """Resolve projection metadata from the built graph.

        Parameters
        ----------
        graph_data:
            The fully built graph returned by ``GraphCreator``.
        dataset_names:
            Dataset node names expected in the graph.
        """
        dataset_name = dataset_names[0]

        return ProjectionData(
            **self._resolve_truncation(graph_data, dataset_name, dataset_names),
            multiscale_loss_matrices_graph=self._resolve_multiscale(graph_data, dataset_name, dataset_names),
        )

    def _resolve_truncation(
        self,
        graph_data: HeteroData,
        dataset_name: str,
        dataset_names: list[str],
    ) -> dict[str, Any]:
        truncation_cfg = self._get("truncation")
        if truncation_cfg is None:
            return {}

        if OmegaConf.is_config(truncation_cfg):
            truncation_cfg = OmegaConf.to_container(truncation_cfg, resolve=True)

        down_edges, up_edges = residual_projection_edge_names(
            dataset_name=dataset_name,
            graph_or_config=graph_data,
            dataset_names=dataset_names,
            truncation_projection_config=truncation_cfg,
        )
        edge_weight = truncation_cfg.get("edge_weight_attribute", DEFAULT_EDGE_WEIGHT_ATTRIBUTE)

        return {
            "truncation_down_edges_name": down_edges,
            "truncation_up_edges_name": up_edges,
            "truncation_edge_weight_attribute": edge_weight,
        }

    def _resolve_multiscale(
        self,
        graph_data: HeteroData,
        dataset_name: str,
        dataset_names: list[str],
    ) -> list[dict | None] | None:
        multiscale_cfg = self._get("multiscale")
        if multiscale_cfg is None:
            return None

        return multiscale_loss_matrices_graph(
            multiscale_cfg,
            dataset_name=dataset_name,
            graph_or_config=graph_data,
            dataset_names=dataset_names,
        )

    def _get(self, key: str) -> Mapping | None:
        """Return a sub-config by key, or None if absent/null."""
        value = (
            self.config.get(key)
            if isinstance(self.config, Mapping) or OmegaConf.is_config(self.config)
            else getattr(self.config, key, None)
        )
        return value if value is not None else None
