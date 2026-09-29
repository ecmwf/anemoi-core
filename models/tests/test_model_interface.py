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

from anemoi.models.interface import AnemoiModelInterface


def _projection_graph() -> HeteroData:
    graph = HeteroData()
    graph["source"].num_nodes = 2
    graph["projected"].num_nodes = 1
    graph["source", "to", "projected"].edge_index = torch.tensor([[0, 1], [0, 0]])
    return graph


class _GraphBuildingModel(torch.nn.Module):
    """Stand-in model: like the real ones, it builds the graph from the graph config."""

    def __init__(self, *, model_graph_config, **kwargs) -> None:
        super().__init__()
        self._graph_data = _projection_graph()


def test_interface_passes_complete_graph_to_spatial_preprocessor() -> None:
    config = OmegaConf.create(
        {
            "data": {
                "datasets": {
                    "projected": {
                        "processors": {},
                        "spatial_processor": {
                            "_target_": "anemoi.models.preprocessing.cross_grid_projector.CrossGridProjector",
                            "edges_name": ["source", "to", "projected"],
                        },
                    }
                },
            },
            "graph": {"nodes": {}, "edges": []},
            "model": {"model": {"_target_": f"{__name__}._GraphBuildingModel"}},
        }
    )
    model_interface = AnemoiModelInterface.__new__(AnemoiModelInterface)
    torch.nn.Module.__init__(model_interface)
    model_interface.config = config
    model_interface.statistics = {}
    model_interface.statistics_tendencies = None
    model_interface.data_indices = {}
    model_interface.is_dataset_static = {}
    model_interface.n_step_input = {}
    model_interface.n_step_output = {}

    model_interface._build_model()

    # The interface takes the graph the model built.
    assert model_interface.graph_data is model_interface.model._graph_data
    projector = model_interface.spatial_pre_processors["projected"]
    projected, grid_shard_sizes = projector(torch.tensor([[[[[1.0], [3.0]]]]]))

    assert grid_shard_sizes is None
    torch.testing.assert_close(projected, torch.tensor([[[[[2.0]]]]]))
