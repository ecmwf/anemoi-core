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
from torch_geometric.data import HeteroData

from anemoi.models.models.base import BaseGraphModel
from anemoi.utils.config import DotDict


class DummyGraphModel(BaseGraphModel):
    def _build_networks(self, model_config) -> None:
        self.seen_hidden_name = model_config.model.model.hidden_nodes_name

    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):
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


def test_base_graph_model_builds_with_omegaconf_config() -> None:
    model_config = OmegaConf.create(
        {
            "graph": {
                "projections": {},
            },
            "training": {
                "multistep_input": 1,
                "multistep_output": 1,
            },
            "model": {
                "num_channels": 8,
                "trainable_parameters": {
                    "data": 0,
                    "hidden": 0,
                },
                "model": {
                    "hidden_nodes_name": "hidden",
                    "latent_skip": False,
                },
                "residual": {
                    "_target_": "anemoi.models.layers.residual.SkipConnection",
                },
                "bounding": [],
            },
        },
    )

    model = DummyGraphModel(
        model_config=model_config,
        data_indices=_make_data_indices(),
        statistics={"data": None},
        graph_data=_make_graph(),
    )

    assert model.seen_hidden_name == "hidden"
    assert "data" in model.residual


def test_base_graph_model_accepts_omegaconf_hidden_node_lists() -> None:
    model_config = OmegaConf.create(
        {
            "graph": {
                "projections": {},
            },
            "training": {
                "multistep_input": 1,
                "multistep_output": 1,
            },
            "model": {
                "num_channels": 8,
                "trainable_parameters": {
                    "data": 0,
                    "hidden": 0,
                },
                "model": {
                    "hidden_nodes_name": ["hidden_1", "hidden_2", "hidden_3"],
                    "latent_skip": False,
                },
                "residual": {
                    "_target_": "anemoi.models.layers.residual.SkipConnection",
                },
                "bounding": [],
            },
        },
    )

    model = DummyGraphModel(
        model_config=model_config,
        data_indices=_make_data_indices(),
        statistics={"data": None},
        graph_data=_make_hierarchical_graph(),
    )

    assert list(model.seen_hidden_name) == ["hidden_1", "hidden_2", "hidden_3"]
    assert model.node_attributes.num_nodes["hidden_3"] == 1


def test_base_graph_model_accepts_dotdict_model_config() -> None:
    model_config = DotDict(
        {
            "graph": {
                "projections": {},
            },
            "training": {
                "multistep_input": 1,
                "multistep_output": 1,
            },
            "model": {
                "num_channels": 8,
                "trainable_parameters": {
                    "data": 0,
                    "hidden": 0,
                },
                "model": {
                    "hidden_nodes_name": "hidden",
                    "latent_skip": False,
                },
                "residual": {
                    "_target_": "anemoi.models.layers.residual.SkipConnection",
                },
                "bounding": [],
            },
        },
    )

    model = DummyGraphModel(
        model_config=model_config,
        data_indices=_make_data_indices(),
        statistics={"data": None},
        graph_data=_make_graph(),
    )

    assert model.seen_hidden_name == "hidden"
    assert model.node_attributes.num_nodes["hidden"] == 1
