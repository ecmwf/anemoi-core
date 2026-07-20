#!/usr/bin/env python3
# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Build an Anemoi model four ways, showing sub-module injection.

A model is constructed directly in Python from a :class:`Parametrisation` (Hydra-free).
Any sub-module can be supplied three ways -- and mixed:

1. **default**  -- omit it (``None``): the model builds the class named in the
   parametrisation, injecting the runtime-computed dimensions;
2. **instance** -- pass an already-built ``nn.Module``: used as-is;
3. **string**   -- pass a dotted import path: resolved via ``params.create_module`` (this is
   where Hydra reattaches later). The model injects its runtime kwargs, so the target must be
   constructible from those (``SkipConnection`` swallows them via ``**_``).

Here the *processor* is injected as an instance and the *residual* as a string; everything
else falls back to the parametrisation defaults. Everything runs on CPU (the GraphTransformer
layers fall back to the ``pyg`` attention backend when triton is unavailable).

    python build_model_four_ways.py
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.processor import GraphTransformerProcessor
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.utils.parametrisation import DictParametrisation

NUM_CHANNELS = 16
SKIP_CONNECTION = "anemoi.models.layers.residual.SkipConnection"


def make_graph() -> HeteroData:
    """A tiny synthetic graph: 12 data nodes, 5 hidden nodes, all three edge types."""
    gen = torch.Generator().manual_seed(0)
    graph = HeteroData()
    graph["data"].x = torch.rand(12, 2, generator=gen)  # lat/lon (radians)
    graph["data"].num_nodes = 12
    graph["hidden"].x = torch.rand(5, 2, generator=gen)
    graph["hidden"].num_nodes = 5
    for src, dst in [("data", "hidden"), ("hidden", "hidden"), ("hidden", "data")]:
        edges = graph[(src, "to", dst)]
        edges.edge_index = torch.randint(0, 5, (2, 20), generator=gen)
        edges["edge_length"] = torch.rand(20, 1, generator=gen)
    return graph


def make_parametrisation() -> DictParametrisation:
    """A small model parametrisation (what training would build from the dataset)."""
    layer = dict(
        num_chunks=1,
        num_heads=1,
        mlp_hidden_ratio=2,
        sub_graph_edge_attributes=["edge_length"],
        trainable_size=0,
        layer_kernels={},  # empty -> torch.nn defaults (Linear/LayerNorm/GELU)
    )
    return DictParametrisation(
        {
            "model": {
                "num_channels": NUM_CHANNELS,
                "trainable_parameters": {"data": 0, "hidden": 0},
                "model": {"hidden_nodes_name": "hidden", "latent_skip": False},
                "residual": {"_target_": SKIP_CONNECTION},
                "bounding": [],
                "encoder": {"_target_": "anemoi.models.layers.mapper.GraphTransformerForwardMapper", **layer},
                "processor": {
                    "_target_": "anemoi.models.layers.processor.GraphTransformerProcessor",
                    "num_layers": 1,
                    **layer,
                },
                "decoder": {"_target_": "anemoi.models.layers.mapper.GraphTransformerBackwardMapper", **layer},
            },
        },
    )


def make_data_kwargs(graph: HeteroData) -> dict:
    """The non-parametrisation inputs (data, not config) every model needs."""
    data_config = DictConfig({"data": {"normalizer": {"default": "none"}, "forcing": [], "diagnostic": []}})
    data_indices = IndexCollection(data_config=data_config.data, name_to_index={"a": 0, "b": 1})
    return dict(
        data_indices={"data": data_indices},
        statistics={"data": {"mean": torch.zeros(2), "stdev": torch.ones(2)}},
        n_step_input=1,
        n_step_output=1,
        graph_data=graph,
    )


def main() -> None:
    graph = make_graph()
    params = make_parametrisation()
    data = make_data_kwargs(graph)

    # 1. All defaults: encoder/processor/decoder/residual come from the parametrisation.
    model_defaults = AnemoiModelEncProcDec(params, **data)

    # 2. Sub-module as an instance: a pre-built processor is used as-is. It needs the runtime
    #    edge_dim, which we read from a default-built model's graph provider.
    edge_dim = model_defaults.processor_graph_provider.edge_dim
    processor = GraphTransformerProcessor(
        num_layers=1,
        num_channels=params.get("model.num_channels"),
        num_chunks=1,
        num_heads=1,
        mlp_hidden_ratio=2,
        edge_dim=edge_dim,
        layer_kernels={},
    )
    model_instance = AnemoiModelEncProcDec(params, processor=processor, **data)

    # 3. Sub-module as a string: the residual is resolved from its dotted path.
    model_string = AnemoiModelEncProcDec(params, residual=SKIP_CONNECTION, **data)

    # 4. Mixing both: instance processor + string residual (encoder/decoder still defaulted).
    model_mixed = AnemoiModelEncProcDec(params, processor=processor, residual=SKIP_CONNECTION, **data)

    for label, model in [
        ("1. defaults", model_defaults),
        ("2. instance", model_instance),
        ("3. string  ", model_string),
        ("4. mixed   ", model_mixed),
    ]:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"{label}: processor={type(model.processor).__name__} "
            f"residual={type(model.residual['data']).__name__} params={n_params}"
        )

    assert model_instance.processor is processor  # instance used verbatim
    assert model_mixed.processor is processor
    print("OK: built AnemoiModelEncProcDec four ways.")


if __name__ == "__main__":
    main()
