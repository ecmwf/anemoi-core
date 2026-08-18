# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""End-to-end construction test for the ModelBuilder (object injection)."""

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.models.builder import build_model
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec


class _IndexGroup(SimpleNamespace):
    def __len__(self):
        return len(self.prognostic)


def _data_indices() -> dict:
    di = SimpleNamespace(
        model=SimpleNamespace(
            input=_IndexGroup(prognostic=[0, 1]),
            output=_IndexGroup(prognostic=[0, 1], full=[0, 1], diagnostic=[], name_to_index={"a": 0, "b": 1}),
            _forcing=[],
        ),
        data=SimpleNamespace(input=SimpleNamespace(name_to_index={"a": 0, "b": 1})),
        name_to_index={"a": 0, "b": 1},
    )
    return {"data": di}


def _graph() -> HeteroData:
    graph = HeteroData()
    graph["data"].x = torch.rand(4, 2)
    graph["data"].num_nodes = 4
    graph["hidden"].x = torch.rand(3, 2)
    graph["hidden"].num_nodes = 3
    for src, dst, n_src, n_dst in [("data", "hidden", 4, 3), ("hidden", "hidden", 3, 3), ("hidden", "data", 3, 4)]:
        edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int64)
        graph[(src, "to", dst)].edge_index = edge_index
        graph[(src, "to", dst)].edge_length = torch.rand(3, 1)
        graph[(src, "to", dst)].edge_dirs = torch.rand(3, 2)
    return graph


def _model_config() -> OmegaConf:
    gnn = OmegaConf.load("training/src/anemoi/training/config/model/gnn.yaml")
    gnn.num_channels = 16
    gnn.processor.num_layers = 2
    gnn.bounding = []
    # keep it lightweight / deterministic on CPU
    gnn.processor.gradient_checkpointing = False
    gnn.encoder.gradient_checkpointing = False
    gnn.decoder.gradient_checkpointing = False
    return OmegaConf.create({"model": gnn})


def test_build_model_enc_proc_dec_injects_networks() -> None:
    model = build_model(
        _model_config(),
        data_indices=_data_indices(),
        statistics={"data": None},
        graph_data=_graph(),
        n_step_input=1,
        n_step_output=1,
    )

    assert isinstance(model, AnemoiModelEncProcDec)
    # Networks were injected (not built inside the model).
    assert isinstance(model.encoder, nn.ModuleDict) and "data" in model.encoder
    assert isinstance(model.decoder, nn.ModuleDict) and "data" in model.decoder
    assert model.processor is not None
    assert "data" in model.encoder_graph_provider
    assert "data" in model.residual
    assert "data" in model.node_attributes.num_nodes
