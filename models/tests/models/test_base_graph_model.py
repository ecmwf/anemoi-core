# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.models.base import BaseGraphModel
from anemoi.models.models.builder import _named_node_attributes_graph
from anemoi.models.utils.config import broadcast_config_keys
from anemoi.utils.builder import build


class DummyGraphModel(BaseGraphModel):
    """Concrete model with no networks of its own — exercises the injected base only."""

    def _assemble_input(self, x, batch_size, grid_shard_sizes=None, model_comm_group=None):
        return x

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        return x_out

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _IndexGroup(SimpleNamespace):
    def __len__(self):
        return len(self.prognostic)


def _make_data_indices() -> dict:
    dataset_indices = SimpleNamespace(
        model=SimpleNamespace(
            input=_IndexGroup(prognostic=[0]),
            output=_IndexGroup(prognostic=[0], full=[0], diagnostic=[], name_to_index={"var": 0}),
            _forcing=[],
        ),
        data=SimpleNamespace(
            input=SimpleNamespace(
                name_to_index={"var": 0},
            ),
        ),
        name_to_index={"var": 0},
    )
    return {"data": dataset_indices}


def _make_graph() -> HeteroData:
    graph = HeteroData()
    graph["data"].x = torch.zeros(2, 2)
    graph["data"].num_nodes = 2
    graph["hidden"].x = torch.zeros(1, 2)
    graph["hidden"].num_nodes = 1
    return graph


def _make_hierarchical_graph() -> HeteroData:
    graph = HeteroData()
    graph["data"].x = torch.zeros(2, 2)
    graph["data"].num_nodes = 2
    for hidden_name in ["hidden_1", "hidden_2", "hidden_3"]:
        graph[hidden_name].x = torch.zeros(1, 2)
        graph[hidden_name].num_nodes = 1
    return graph


def _injected_kwargs(model_config, data_indices, statistics, graph, hidden_nodes_name) -> dict:
    """Build the injected sub-objects the way a ModelBuilder would, for the base container."""
    dataset_names = list(data_indices)
    hidden_names = BaseGraphModel._as_hidden_node_names(hidden_nodes_name)
    trainable = broadcast_config_keys(
        model_config.model.trainable_parameters, data=dataset_names, hidden=hidden_nodes_name
    )
    node_attributes = NamedNodesAttributes(trainable, _named_node_attributes_graph(graph, dataset_names + hidden_names))
    residual = nn.ModuleDict(
        {
            ds: build(
                model_config.model.residual,
                graph=graph,
                data_node_name=ds,
                statistics=statistics[ds],
                data_indices=data_indices[ds],
                dataset_name=ds,
                sparse_projector_num_chunks=1,
            )
            for ds in dataset_names
        }
    )
    boundings = nn.ModuleDict({ds: nn.ModuleList() for ds in dataset_names})
    return {
        "node_attributes": node_attributes,
        "residual": residual,
        "boundings": boundings,
        "data_indices": data_indices,
        "statistics": statistics,
        "n_step_input": 1,
        "n_step_output": 1,
        "graph_data": graph,
        "hidden_nodes_name": hidden_nodes_name,
        "num_channels": model_config.model.num_channels,
        "latent_skip": model_config.model.model.latent_skip,
    }


def test_base_graph_model_stores_injected_objects() -> None:
    model_config = OmegaConf.create(
        {
            "model": {
                "num_channels": 8,
                "trainable_parameters": {"data": 0, "hidden": 0},
                "model": {"hidden_nodes_name": "hidden", "latent_skip": False},
                "residual": {"_target_": "anemoi.models.layers.residual.SkipConnection"},
                "bounding": [],
            },
        },
    )
    data_indices = _make_data_indices()
    statistics = {"data": None}
    graph = _make_graph()

    model = DummyGraphModel(**_injected_kwargs(model_config, data_indices, statistics, graph, "hidden"))

    assert model._graph_name_hidden == "hidden"
    assert "data" in model.residual
    assert model.node_attributes.num_nodes["hidden"] == 1


def test_base_graph_model_accepts_hidden_node_lists() -> None:
    model_config = OmegaConf.create(
        {
            "model": {
                "num_channels": 8,
                "trainable_parameters": {"data": 0, "hidden": 0},
                "model": {"hidden_nodes_name": ["hidden_1", "hidden_2", "hidden_3"], "latent_skip": False},
                "residual": {"_target_": "anemoi.models.layers.residual.SkipConnection"},
                "bounding": [],
            },
        },
    )
    data_indices = _make_data_indices()
    statistics = {"data": None}
    graph = _make_hierarchical_graph()

    model = DummyGraphModel(
        **_injected_kwargs(model_config, data_indices, statistics, graph, ["hidden_1", "hidden_2", "hidden_3"])
    )

    assert list(model._graph_name_hidden) == ["hidden_1", "hidden_2", "hidden_3"]
    assert model.node_attributes.num_nodes["hidden_3"] == 1
