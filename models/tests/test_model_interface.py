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


class _GraphModel(torch.nn.Module):
    """Stand-in model: like the real ones, it receives the static graph from the interface."""

    def __init__(self, *, static_graph_data, dynamic_graph_config, **kwargs) -> None:
        super().__init__()
        self.received_graph = static_graph_data


def test_interface_passes_complete_graph_to_spatial_preprocessor(tmp_path) -> None:
    graph_path = tmp_path / "graph.pt"
    torch.save(_projection_graph(), graph_path)
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
            "graph": {"overwrite": False},
            "system": {"input": {"graph": str(graph_path)}},
            "model": {"model": {"_target_": f"{__name__}._GraphModel", "hidden_nodes_name": "projected"}},
        }
    )
    model_interface = AnemoiModelInterface.__new__(AnemoiModelInterface)
    torch.nn.Module.__init__(model_interface)
    model_interface.config = config
    model_interface.statistics = {}
    model_interface.statistics_tendencies = None
    model_interface.data_indices = {}
    model_interface.data_types = {"source": "gridded"}
    model_interface.n_step_input = {}
    model_interface.n_step_output = {}

    model_interface._build_model()

    # The interface loads the graph and hands the same object to the model.
    assert model_interface.graph_data is model_interface.model.received_graph
    projector = model_interface.spatial_pre_processors["projected"]
    projected, grid_shard_sizes = projector(torch.tensor([[[[[1.0], [3.0]]]]]))

    assert grid_shard_sizes is None
    torch.testing.assert_close(projected, torch.tensor([[[[[2.0]]]]]))
