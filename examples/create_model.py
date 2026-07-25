# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Example: create an Anemoi model via dependency injection.

Two ways are shown:

1. ``build_from_config`` — the config-driven path. A :class:`ModelBuilder` (invoked through
   :func:`anemoi.models.models.builder.build_model`) is the ONLY code that reads the
   configuration. It builds every sub-object (node attributes, graph providers,
   encoder/processor/decoder, residual, boundings) and injects them into the model
   constructor. This is what runs today.

2. ``build_from_objects`` — the target end-state (dependency injection with NO settings).
   The model is constructed directly from already-built objects and plain primitives; no
   ``DictConfig``/``DotDict``/settings object reaches any constructor. This is the shape the
   refactor is driving toward (see REFACTOR_SPEC.md, principle P2).

Run:  python examples/create_model.py
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.models.builder import build_model

REPO = Path(__file__).resolve().parents[1]
GNN_CONFIG = REPO / "training/src/anemoi/training/config/model/gnn.yaml"


# --------------------------------------------------------------------------------------
# Minimal runtime inputs (normally produced by the datamodule + graph creation).
# --------------------------------------------------------------------------------------
class _IndexGroup(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.prognostic)


def make_data_indices() -> dict:
    """A tiny single-dataset ``data_indices`` with two prognostic variables."""
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


def make_graph() -> HeteroData:
    """A tiny graph with data & hidden nodes and the three required edge sets."""
    graph = HeteroData()
    graph["data"].x = torch.rand(4, 2)
    graph["data"].num_nodes = 4
    graph["hidden"].x = torch.rand(3, 2)
    graph["hidden"].num_nodes = 3
    for src, dst in [("data", "hidden"), ("hidden", "hidden"), ("hidden", "data")]:
        graph[(src, "to", dst)].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int64)
        graph[(src, "to", dst)].edge_length = torch.rand(3, 1)  # 1-d edge attribute
        graph[(src, "to", dst)].edge_dirs = torch.rand(3, 2)  # 2-d edge attribute
    return graph


def make_model_config() -> OmegaConf:
    """Load the shipped GNN model config and shrink it for a CPU example."""
    gnn = OmegaConf.load(GNN_CONFIG)
    gnn.num_channels = 16
    gnn.processor.num_layers = 2
    gnn.bounding = []
    gnn.processor.gradient_checkpointing = False
    gnn.encoder.gradient_checkpointing = False
    gnn.decoder.gradient_checkpointing = False
    return OmegaConf.create({"model": gnn})


# --------------------------------------------------------------------------------------
# 1. Config-driven construction (what works today).
# --------------------------------------------------------------------------------------
def build_from_config():
    """The ModelBuilder reads the config and injects every built sub-object."""
    model = build_model(
        make_model_config(),
        data_indices=make_data_indices(),
        statistics={"data": None},
        graph_data=make_graph(),
        n_step_input=1,
        n_step_output=1,
    )
    # The networks were built by the builder and injected — the model just stores them.
    assert isinstance(model.encoder, nn.ModuleDict) and "data" in model.encoder
    assert isinstance(model.decoder, nn.ModuleDict) and "data" in model.decoder
    assert model.processor is not None
    assert "data" in model.residual
    return model


# --------------------------------------------------------------------------------------
# 2. Target end-state: pure dependency injection, NO settings object anywhere.
# --------------------------------------------------------------------------------------
def build_from_objects():
    """Illustrative: construct the model from built objects + primitives only.

    This is the shape principle P2 targets — the constructor sees ints/floats/strings and
    already-built modules, never a config/settings object. (Building the encoder/processor/
    decoder mappers here becomes fully turn-key once the layer-kernel registry is injected
    as a built object rather than read from config inside the mapper; see REFACTOR_FINDINGS
    §3. The pattern below shows how the pieces are wired.)
    """
    from anemoi.models.layers.graph import NamedNodesAttributes
    from anemoi.models.layers.graph_provider import create_graph_provider
    from anemoi.models.layers.mapper import GNNBackwardMapper
    from anemoi.models.layers.mapper import GNNForwardMapper
    from anemoi.models.layers.processor import GNNProcessor
    from anemoi.models.layers.residual import SkipConnection
    from anemoi.models.layers.utils import load_layer_kernels
    from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec

    graph = make_graph()
    data_indices = make_data_indices()
    num_channels = 16

    # Built objects (no config): node attributes, graph providers, layer-kernel registry.
    node_attributes = NamedNodesAttributes({"data": 8, "hidden": 8}, graph)
    kernels = load_layer_kernels(None)  # a built registry of layer factories (an object)

    enc_gp = nn.ModuleDict(
        {
            "data": create_graph_provider(
                graph=graph[("data", "to", "hidden")],
                edge_attributes=["edge_length", "edge_dirs"],
                src_size=4,
                dst_size=3,
            )
        }
    )
    proc_gp = create_graph_provider(
        graph=graph[("hidden", "to", "hidden")], edge_attributes=["edge_length", "edge_dirs"], src_size=3, dst_size=3
    )
    dec_gp = nn.ModuleDict(
        {
            "data": create_graph_provider(
                graph=graph[("hidden", "to", "data")],
                edge_attributes=["edge_length", "edge_dirs"],
                src_size=3,
                dst_size=4,
            )
        }
    )

    input_dim = 1 * 2 + node_attributes.attr_ndims["data"]  # n_step_input * n_vars + coord dims
    latent_dim = node_attributes.attr_ndims["hidden"]
    output_dim = 1 * 2

    # Encoder/processor/decoder are built with PRIMITIVES + the built kernel registry.
    encoder = nn.ModuleDict(
        {
            "data": GNNForwardMapper(
                in_channels_src=input_dim,
                in_channels_dst=latent_dim,
                hidden_dim=num_channels,
                edge_dim=enc_gp["data"].edge_dim,
                num_chunks=1,
                mlp_extra_layers=0,
                layer_kernels=kernels,
            )
        }
    )
    processor = GNNProcessor(
        num_channels=num_channels,
        edge_dim=proc_gp.edge_dim,
        num_layers=2,
        num_chunks=1,
        mlp_extra_layers=0,
        layer_kernels=kernels,
    )
    decoder = nn.ModuleDict(
        {
            "data": GNNBackwardMapper(
                in_channels_src=num_channels,
                in_channels_dst=input_dim,
                hidden_dim=num_channels,
                out_channels_dst=output_dim,
                edge_dim=dec_gp["data"].edge_dim,
                num_chunks=1,
                mlp_extra_layers=0,
                layer_kernels=kernels,
            )
        }
    )
    residual = nn.ModuleDict({"data": SkipConnection(step=-1)})
    boundings = nn.ModuleDict({"data": nn.ModuleList()})

    # Everything above is a built object or a primitive — no config reaches the constructor.
    model = AnemoiModelEncProcDec(
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        encoder_graph_provider=enc_gp,
        processor_graph_provider=proc_gp,
        decoder_graph_provider=dec_gp,
        node_attributes=node_attributes,
        residual=residual,
        boundings=boundings,
        data_indices=data_indices,
        statistics={"data": None},
        n_step_input=1,
        n_step_output=1,
        graph_data=graph,
        hidden_nodes_name="hidden",
        num_channels=num_channels,
        latent_skip=True,
    )
    return model


if __name__ == "__main__":
    model = build_from_config()
    print("build_from_config: built", type(model).__name__)
    print("  encoder datasets:", list(model.encoder.keys()))
    print("  processor:", type(model.processor).__name__)
    print("  decoder datasets:", list(model.decoder.keys()))
    print("  #parameters:", sum(p.numel() for p in model.parameters()))

    model2 = build_from_objects()
    print("build_from_objects (no settings): built", type(model2).__name__)
    print("  #parameters:", sum(p.numel() for p in model2.parameters()))
