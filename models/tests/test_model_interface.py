# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

import anemoi.models.interface as interface_module
from anemoi.models.interface import AnemoiModelInterface
from anemoi.models.preprocessing.cross_grid_projector import CrossGridProjector


class _DummyModel(torch.nn.Module):
    def forward(self, x, **_kwargs):
        return x


def test_interface_passes_complete_graph_to_spatial_preprocessor(monkeypatch) -> None:
    graph = HeteroData()
    graph["source"].num_nodes = 2
    graph["projected"].num_nodes = 1
    graph["source", "to", "projected"].edge_index = torch.tensor([[0, 1], [0, 0]])

    config = OmegaConf.create(
        {
            "data": {
                "datasets": {},
                "spatial_processors": {
                    "projected": {
                        "_target_": "anemoi.models.preprocessing.cross_grid_projector.CrossGridProjector",
                        "edges_name": ["source", "to", "projected"],
                    }
                },
            },
            "model": {"model": {"_target_": "unused.DummyModel"}},
        }
    )
    spatial_config = config.data.spatial_processors.projected

    def instantiate(config_to_instantiate, **kwargs):
        if config_to_instantiate is spatial_config:
            return CrossGridProjector(
                graph=kwargs["graph"],
                edges_name=tuple(config_to_instantiate.edges_name),
            )
        return _DummyModel()

    monkeypatch.setattr(interface_module, "instantiate", instantiate)

    model_interface = AnemoiModelInterface.__new__(AnemoiModelInterface)
    torch.nn.Module.__init__(model_interface)
    model_interface.config = config
    model_interface.graph_data = graph
    model_interface.statistics = {}
    model_interface.statistics_tendencies = None
    model_interface.data_indices = {}
    model_interface.n_step_input = 1
    model_interface.n_step_output = 1

    model_interface._build_model()

    projector = model_interface.spatial_pre_processors["projected"]
    projected, grid_shard_sizes = projector(torch.tensor([[[[[1.0], [3.0]]]]]))

    assert grid_shard_sizes is None
    torch.testing.assert_close(projected, torch.tensor([[[[[2.0]]]]]))
