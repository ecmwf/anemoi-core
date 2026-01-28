# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from torch_geometric.data import HeteroData

from anemoi.models.api import build_model
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.utils.config import DotDict


def _build_graph(num_data: int, num_hidden: int) -> HeteroData:
    graph = HeteroData()
    graph["data"].x = torch.rand(num_data, 2)
    graph["hidden"].x = torch.rand(num_hidden, 2)

    data_ids = torch.arange(num_data)
    hidden_ids = torch.arange(num_hidden)

    data_to_hidden = torch.cartesian_prod(data_ids, hidden_ids).t().contiguous()
    hidden_to_data = torch.cartesian_prod(hidden_ids, data_ids).t().contiguous()
    hidden_to_hidden = torch.stack([hidden_ids, hidden_ids], dim=0)

    graph[("data", "to", "hidden")].edge_index = data_to_hidden
    graph[("data", "to", "hidden")].edge_length = torch.ones(data_to_hidden.shape[1], 1)

    graph[("hidden", "to", "data")].edge_index = hidden_to_data
    graph[("hidden", "to", "data")].edge_length = torch.ones(hidden_to_data.shape[1], 1)

    graph[("hidden", "to", "hidden")].edge_index = hidden_to_hidden
    graph[("hidden", "to", "hidden")].edge_length = torch.ones(hidden_to_hidden.shape[1], 1)

    return graph


def _build_config(layer_kernels: dict) -> dict:
    return {
        "graph": {"data": "data", "hidden": "hidden"},
        "training": {"multistep_input": 1},
        "model": {
            "model": {"_target_": "anemoi.models.models.AnemoiModelEncProcDec"},
            "num_channels": 4,
            "trainable_parameters": {
                "data": 0,
                "hidden": 0,
                "data2hidden": 0,
                "hidden2data": 0,
                "hidden2hidden": 0,
            },
            "layer_kernels": layer_kernels,
            "processor": {
                "_target_": "anemoi.models.layers.processor.GNNProcessor",
                "trainable_size": 0,
                "sub_graph_edge_attributes": ["edge_length"],
                "num_layers": 1,
                "num_chunks": 1,
                "mlp_extra_layers": 0,
                "cpu_offload": False,
                "layer_kernels": layer_kernels,
            },
            "encoder": {
                "_target_": "anemoi.models.layers.mapper.GNNForwardMapper",
                "trainable_size": 0,
                "sub_graph_edge_attributes": ["edge_length"],
                "num_chunks": 1,
                "mlp_extra_layers": 0,
                "cpu_offload": False,
                "layer_kernels": layer_kernels,
            },
            "decoder": {
                "_target_": "anemoi.models.layers.mapper.GNNBackwardMapper",
                "trainable_size": 0,
                "sub_graph_edge_attributes": ["edge_length"],
                "num_chunks": 1,
                "mlp_extra_layers": 0,
                "cpu_offload": False,
                "layer_kernels": layer_kernels,
            },
            "residual": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1},
            "bounding": [],
        },
    }


def test_models_api_smoke():
    graph = _build_graph(num_data=4, num_hidden=3)

    config = DotDict({"data": {"forcing": [], "diagnostic": [], "target": []}})
    name_to_index = {"var1": 0, "var2": 1}
    data_indices = {"data": IndexCollection(config.data, name_to_index=name_to_index)}

    statistics = {"data": {}}

    layer_kernels = {
        "LayerNorm": {"_target_": "torch.nn.LayerNorm"},
        "Linear": {"_target_": "torch.nn.Linear"},
        "Activation": {"_target_": "torch.nn.GELU"},
    }

    model = build_model(
        _build_config(layer_kernels),
        graph_data={"data": graph},
        data_indices=data_indices,
        statistics=statistics,
    )

    assert isinstance(model, torch.nn.Module)

    x = {"data": torch.randn(1, 1, 1, graph["data"].num_nodes, len(name_to_index))}
    out = model(x)
    assert "data" in out
    assert out["data"].shape[-1] == len(data_indices["data"].model.output.full)
