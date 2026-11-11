# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models import AnemoiModelDisentangledEncProcDec
from anemoi.models.models import AnemoiModelEncProcDec

VARIABLES = ["a", "b", "c", "f"]
N_DATA, N_HIDDEN = 40, 20


def _coords(n: int) -> torch.Tensor:
    lat = torch.linspace(-1.2, 1.2, n)
    lon = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack([lat, lon], dim=-1)


def _add_edges(graph: HeteroData, src: str, dst: str, k: int = 3) -> None:
    n_src, n_dst = graph[src].num_nodes, graph[dst].num_nodes
    edges = graph[(src, "to", dst)]
    edges.edge_index = torch.stack([torch.randint(0, n_src, (n_dst * k,)), torch.arange(n_dst).repeat_interleave(k)])
    edges.edge_length = torch.rand(n_dst * k, 1)
    edges.edge_dirs = torch.rand(n_dst * k, 2)


def _graph() -> HeteroData:
    graph = HeteroData()
    for name, n in (("data", N_DATA), ("hidden", N_HIDDEN)):
        graph[name].x = _coords(n)
        graph[name].num_nodes = n
    _add_edges(graph, "data", "hidden")
    _add_edges(graph, "hidden", "hidden")
    _add_edges(graph, "hidden", "data")
    return graph


def _mapper(target: str) -> dict:
    return {
        "_target_": target,
        "num_channels": 16,
        "trainable_size": 2,
        "sub_graph_edge_attributes": ["edge_length", "edge_dirs"],
        "num_chunks": 1,
        "mlp_hidden_ratio": 2,
        "num_heads": 2,
        "qk_norm": False,
        "cpu_offload": False,
        "gradient_checkpointing": False,
        "layer_kernels": {},
        "shard_strategy": "edges",
        "graph_attention_backend": "pyg",
        "edge_pre_mlp": False,
    }


def _model_config(target: str):
    processor = _mapper("anemoi.models.layers.processor.GraphTransformerProcessor")
    processor["num_layers"] = 2
    return OmegaConf.create(
        {
            "num_channels": 16,
            "model": {
                "_target_": target,
                "hidden_nodes_name": "hidden",
                "latent_skip": True,
            },
            "processor": processor,
            "encoders": {
                "0": {
                    "source_datasets": ["data"],
                    "dataset_fusing_strategy": "not_supported",
                    "mapper": _mapper("anemoi.models.layers.mapper.GraphTransformerForwardMapper"),
                },
            },
            "latent_aggregator": {"_target_": "anemoi.models.layers.aggregator.SumAggregator"},
            "decoders": {
                "0": {
                    "target_datasets": ["data"],
                    "target_node_features": ["encoded_data"],
                    "mapper": {
                        **_mapper("anemoi.models.layers.mapper.GraphTransformerBackwardMapper"),
                        "initialise_data_extractor_zero": False,
                    },
                },
            },
            "residual": {
                "datasets": {"data": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1}},
            },
            "node_trainable_parameters": {"data": 4, "hidden": 4},
            "bounding": {"datasets": {"data": []}},
        },
    )


def _build(model_class, n_step_input: int, graph: HeteroData | None = None):
    name_to_index = {v: i for i, v in enumerate(VARIABLES)}
    data_indices = {"data": IndexCollection(OmegaConf.create({"forcing": ["f"]}), name_to_index)}
    statistics = {"data": {k: np.ones(len(VARIABLES)) for k in ("mean", "stdev", "minimum", "maximum")}}
    target = f"{model_class.__module__}.{model_class.__name__}"
    return model_class(
        model_config=_model_config(target),
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph if graph is not None else _graph(),
        n_step_input=n_step_input,
        n_step_output=1,
    )


@pytest.fixture
def model() -> AnemoiModelDisentangledEncProcDec:
    torch.manual_seed(0)
    return _build(AnemoiModelDisentangledEncProcDec, n_step_input=2)


def _inputs() -> dict[str, torch.Tensor]:
    return {"data": torch.randn(1, 2, 1, N_DATA, len(VARIABLES))}


def test_output_shape(model: AnemoiModelDisentangledEncProcDec) -> None:
    y = model(_inputs())["data"]
    assert y.shape == (1, 1, 1, N_DATA, 3)


def test_encoder_sees_one_timestep(model: AnemoiModelDisentangledEncProcDec) -> None:
    # vars + node attributes (2 coordinates -> 4 features + 4 trainable), no n_step_input multiplier
    assert model.input_dim["data"] == len(VARIABLES) + model.node_attributes.attr_ndims["data"]


def test_every_input_timestep_contributes(model: AnemoiModelDisentangledEncProcDec) -> None:
    x = _inputs()
    y = model(x)["data"]
    x_changed = {"data": x["data"].clone()}
    x_changed["data"][:, 0] += 1.0  # oldest input step only
    assert not torch.allclose(y, model(x_changed)["data"])


def test_gradients_reach_latent_blender(model: AnemoiModelDisentangledEncProcDec) -> None:
    model(_inputs())["data"].pow(2).mean().backward()
    assert all(p.grad is not None for p in model.latent_blender.parameters() if p.requires_grad)


def test_autoencoder_weights_load_by_name(model: AnemoiModelDisentangledEncProcDec) -> None:
    autoencoder = _build(AnemoiModelEncProcDec, n_step_input=1, graph=model._graph_data)
    missing, unexpected = model.load_state_dict(autoencoder.state_dict(), strict=False)
    assert not unexpected
    assert {key.split(".")[0] for key in missing} == {"latent_blender"}


def test_supports_latent_rollout(model: AnemoiModelDisentangledEncProcDec) -> None:
    assert model.supports_latent_rollout


def test_forward_equals_one_latent_step(model: AnemoiModelDisentangledEncProcDec) -> None:
    x = _inputs()
    y_latent, _ = model.step_latent(model.encode(x), x)
    assert torch.allclose(model(x)["data"], y_latent["data"])


def test_step_latent_shifts_the_state(model: AnemoiModelDisentangledEncProcDec) -> None:
    x = _inputs()
    state = model.encode(x)
    _, next_state = model.step_latent(state, x)
    assert next_state.latents.shape == state.latents.shape
    assert torch.equal(next_state.latents[:, :-1], state.latents[:, 1:])


def test_latent_step_differs_from_reencoding(model: AnemoiModelDisentangledEncProcDec) -> None:
    x, x_next = _inputs(), _inputs()
    _, state = model.step_latent(model.encode(x), x)
    y_latent, _ = model.step_latent(state, x_next)
    assert not torch.allclose(y_latent["data"], model(x_next)["data"])


def test_step_latent_keeps_no_hidden_state(model: AnemoiModelDisentangledEncProcDec) -> None:
    x = _inputs()
    state = model.encode(x)
    y_first, _ = model.step_latent(state, x)
    model(_inputs())  # unrelated call in between
    y_again, _ = model.step_latent(state, x)
    assert torch.equal(y_first["data"], y_again["data"])


def test_gradients_flow_through_the_latent_state(model: AnemoiModelDisentangledEncProcDec) -> None:
    x = _inputs()
    _, state = model.step_latent(model.encode(x), x)
    y_second, _ = model.step_latent(state, x)
    y_second["data"].pow(2).mean().backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.latent_blender.parameters() if p.requires_grad)
