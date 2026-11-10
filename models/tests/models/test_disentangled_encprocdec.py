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


def _model_config(target: str, latent_rollout: bool):
    processor = _mapper("anemoi.models.layers.processor.GraphTransformerProcessor")
    processor["num_layers"] = 2
    return OmegaConf.create(
        {
            "num_channels": 16,
            "model": {
                "_target_": target,
                "hidden_nodes_name": "hidden",
                "latent_skip": True,
                "latent_rollout": latent_rollout,
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


def _build(model_class, n_step_input: int, latent_rollout: bool = True, graph: HeteroData | None = None):
    name_to_index = {v: i for i, v in enumerate(VARIABLES)}
    data_indices = {"data": IndexCollection(OmegaConf.create({"forcing": ["f"]}), name_to_index)}
    statistics = {"data": {k: np.ones(len(VARIABLES)) for k in ("mean", "stdev", "minimum", "maximum")}}
    target = f"{model_class.__module__}.{model_class.__name__}"
    return model_class(
        model_config=_model_config(target, latent_rollout),
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
    y = model(_inputs(), rollout_step=0)["data"]
    assert y.shape == (1, 1, 1, N_DATA, 3)


def test_latent_rollout_advances_buffer_instead_of_reencoding(model: AnemoiModelDisentangledEncProcDec) -> None:
    model(_inputs(), rollout_step=0)
    buffer = model._latent_buffer.clone()
    x_next = _inputs()
    y_latent = model(x_next, rollout_step=1)["data"]
    assert torch.equal(model._latent_buffer[:, 0], buffer[:, 1])
    y_reencoded = model(x_next, rollout_step=0)["data"]
    assert not torch.allclose(y_latent, y_reencoded)


def test_without_latent_rollout_every_step_reencodes(model: AnemoiModelDisentangledEncProcDec) -> None:
    model.latent_rollout = False
    x, x_next = _inputs(), _inputs()
    model(x, rollout_step=0)
    assert torch.allclose(model(x_next, rollout_step=1)["data"], model(x_next, rollout_step=0)["data"])


def test_gradients_flow_through_latent_rollout(model: AnemoiModelDisentangledEncProcDec) -> None:
    loss = sum(model(_inputs(), rollout_step=step)["data"].pow(2).mean() for step in (0, 1))
    loss.backward()
    assert all(p.grad is not None for p in model.latent_blender.parameters() if p.requires_grad)


def test_autoencoder_weights_load_by_name(model: AnemoiModelDisentangledEncProcDec) -> None:
    autoencoder = _build(AnemoiModelEncProcDec, n_step_input=1, graph=model._graph_data)
    missing, unexpected = model.load_state_dict(autoencoder.state_dict(), strict=False)
    assert not unexpected
    assert {key.split(".")[0] for key in missing} == {"latent_blender"}
