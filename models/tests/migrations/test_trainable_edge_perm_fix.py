# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib

import torch
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import StaticGraphProvider

migrate = importlib.import_module("anemoi.models.migrations.scripts.1779202136_trainable_edge_perm_fix").migrate


def _make_static_graph_provider() -> StaticGraphProvider:
    graph = HeteroData()
    graph.edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 1, 0]], dtype=torch.long)
    graph.edge_attr = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)

    return StaticGraphProvider(
        graph=graph,
        edge_attributes=["edge_attr"],
        src_size=3,
        dst_size=2,
        trainable_size=2,
    )


class _InnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_graph_provider = nn.ModuleDict({"data": _make_static_graph_provider()})
        self.processor_graph_provider = _make_static_graph_provider()


class _InterfaceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _InnerModel()


class _RootModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _InterfaceModel()


def test_migration_permutes_static_graph_provider_trainables() -> None:
    model = _RootModel()

    encoder_provider = model.get_submodule("model.model.encoder_graph_provider.data")
    processor_provider = model.get_submodule("model.model.processor_graph_provider")

    encoder_key = "model.model.encoder_graph_provider.data.trainable.trainable"
    processor_key = "model.model.processor_graph_provider.trainable.trainable"

    encoder_legacy = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]],
        dtype=torch.float32,
    )
    processor_legacy = torch.tensor(
        [[50.0, 51.0], [60.0, 61.0], [70.0, 71.0], [80.0, 81.0]],
        dtype=torch.float32,
    )

    checkpoint = {
        "state_dict": {
            encoder_key: encoder_legacy.clone(),
            processor_key: processor_legacy.clone(),
        }
    }

    migrate(checkpoint, model)

    assert torch.equal(
        checkpoint["state_dict"][encoder_key],
        encoder_legacy.index_select(0, encoder_provider.perm),
    )
    assert torch.equal(
        checkpoint["state_dict"][processor_key],
        processor_legacy.index_select(0, processor_provider.perm),
    )
    assert checkpoint["state_dict"]["model.model.encoder_graph_provider.data.trainable_layout_version"].item() == 1
    assert checkpoint["state_dict"]["model.model.processor_graph_provider.trainable_layout_version"].item() == 1


def test_migration_does_not_repermute_current_layout() -> None:
    model = _RootModel()

    processor_key = "model.model.processor_graph_provider.trainable.trainable"
    layout_key = "model.model.processor_graph_provider.trainable_layout_version"
    current_trainable = torch.tensor(
        [[100.0, 101.0], [200.0, 201.0], [300.0, 301.0], [400.0, 401.0]],
        dtype=torch.float32,
    )

    checkpoint = {
        "state_dict": {
            processor_key: current_trainable.clone(),
            layout_key: torch.tensor(1, dtype=torch.int64),
        }
    }

    migrate(checkpoint, model)

    assert torch.equal(checkpoint["state_dict"][processor_key], current_trainable)
