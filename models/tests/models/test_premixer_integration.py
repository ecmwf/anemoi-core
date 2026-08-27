# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""End-to-end checks of the data -> data pre-mixer inside a real model.

`tests/layers/test_premixer.py` covers the module in isolation. These tests
build an actual AnemoiModelAutoEncoder to verify the wiring: that the config
key builds the modules, that the fork contract holds through the whole model,
and that the pre-mixer is a genuine no-op when it is not configured.
"""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.models.autoencoder import AnemoiModelAutoEncoder

N_DATA = 32
N_HIDDEN = 8
N_VARS = 1
EDGE_ATTRS = ["edge_length", "edge_dirs"]
BACKEND = "pyg"  # triton needs a GPU

LAYER_KERNELS = {
    "LayerNorm": {"_target_": "torch.nn.LayerNorm"},
    "Linear": {"_target_": "torch.nn.Linear"},
    "Activation": {"_target_": "torch.nn.GELU"},
    "QueryNorm": {"_target_": "anemoi.models.layers.normalization.AutocastLayerNorm", "bias": False},
    "KeyNorm": {"_target_": "anemoi.models.layers.normalization.AutocastLayerNorm", "bias": False},
}


class _IndexGroup(SimpleNamespace):
    def __len__(self):
        return len(self.prognostic)


def _make_data_indices() -> dict:
    idx = SimpleNamespace(
        model=SimpleNamespace(
            input=_IndexGroup(prognostic=[0]),
            output=_IndexGroup(prognostic=[0], full=[0], diagnostic=[], name_to_index={"var": 0}),
            _forcing=[],
        ),
        data=SimpleNamespace(input=SimpleNamespace(name_to_index={"var": 0})),
        name_to_index={"var": 0},
    )
    return {"data": idx}


def _dst_sorted_edges(n_src: int, n_dst: int, k: int, seed: int):
    """k edges per destination node, sorted by destination."""
    g = torch.Generator().manual_seed(seed)
    dst = torch.arange(n_dst).repeat_interleave(k)
    src = torch.randint(0, n_src, (n_dst * k,), generator=g)
    return torch.stack([src, dst], dim=0), n_dst * k


def _make_graph() -> HeteroData:
    torch.manual_seed(0)
    graph = HeteroData()
    graph["data"].x = torch.rand(N_DATA, 2)
    graph["data"].num_nodes = N_DATA
    graph["hidden"].x = torch.rand(N_HIDDEN, 2)
    graph["hidden"].num_nodes = N_HIDDEN

    for name, n_src, n_dst, k, seed in [
        (("data", "to", "data"), N_DATA, N_DATA, 4, 1),
        (("data", "to", "hidden"), N_DATA, N_HIDDEN, 4, 2),
        (("hidden", "to", "hidden"), N_HIDDEN, N_HIDDEN, 3, 3),
        (("hidden", "to", "data"), N_HIDDEN, N_DATA, 2, 4),
    ]:
        edge_index, n_edges = _dst_sorted_edges(n_src, n_dst, k, seed)
        graph[name].edge_index = edge_index
        graph[name].edge_length = torch.rand(n_edges, 1)
        graph[name].edge_dirs = torch.rand(n_edges, 2)
    return graph


def _component(target: str, **extra) -> dict:
    base = {
        "_target_": target,
        "trainable_size": 0,
        "sub_graph_edge_attributes": EDGE_ATTRS,
        "num_chunks": 1,
        "mlp_hidden_ratio": 2,
        "num_heads": 2,
        "layer_kernels": LAYER_KERNELS,
        "graph_attention_backend": BACKEND,
    }
    base.update(extra)
    return base


def _make_config(with_premixer: bool, initialise_out_zero: bool = True):
    cfg = {
        "model": {
            "num_channels": 16,
            "trainable_parameters": {"data": 0, "hidden": 0, "data2hidden": 0, "hidden2data": 0, "hidden2hidden": 0},
            "model": {"hidden_nodes_name": "hidden", "latent_skip": False},
            "residual": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1},
            "bounding": [],
            "layer_kernels": LAYER_KERNELS,
            "processor": _component(
                "anemoi.models.layers.processor.GraphTransformerProcessor", num_layers=2
            ),
            "encoder": _component("anemoi.models.layers.mapper.GraphTransformerForwardMapper"),
            "decoder": _component("anemoi.models.layers.mapper.GraphTransformerBackwardMapper"),
        },
    }
    if with_premixer:
        cfg["model"]["premixer"] = _component(
            "anemoi.models.layers.premixer.GraphTransformerPreMixer",
            num_channels=16,
            num_layers=2,
            num_chunks=2,
            initialise_out_zero=initialise_out_zero,
        )
    return OmegaConf.create(cfg)


def _build(with_premixer: bool, initialise_out_zero: bool = True, seed: int = 42) -> AnemoiModelAutoEncoder:
    torch.manual_seed(seed)
    return AnemoiModelAutoEncoder(
        model_config=_make_config(with_premixer, initialise_out_zero),
        data_indices=_make_data_indices(),
        statistics={"data": None},
        n_step_input=1,
        n_step_output=1,
        graph_data=_make_graph(),
    )


@pytest.fixture
def x() -> dict:
    torch.manual_seed(7)
    return {"data": torch.rand(1, 1, 1, N_DATA, N_VARS)}


def test_absent_config_builds_no_premixer():
    """Without a model.premixer key the model must be completely unchanged."""
    model = _build(with_premixer=False)
    assert len(model.premixer) == 0
    assert len(model.premixer_graph_provider) == 0


def test_premixer_is_built_per_dataset():
    model = _build(with_premixer=True)
    assert "data" in model.premixer
    assert "data" in model.premixer_graph_provider


def test_output_shape_is_unchanged(x):
    baseline = _build(with_premixer=False).eval()
    premixed = _build(with_premixer=True).eval()
    with torch.no_grad():
        assert premixed(x)["data"].shape == baseline(x)["data"].shape


def test_fork_contract_model_is_bit_identical_at_init(x):
    """The whole model must reproduce the no-pre-mixer model bit-for-bit.

    This is what lets an existing checkpoint be forked into a pre-mixer run:
    every shared weight transfers, and the pre-mixer contributes exactly zero
    until it is trained.
    """
    baseline = _build(with_premixer=False).eval()
    premixed = _build(with_premixer=True, initialise_out_zero=True).eval()

    incompatible = premixed.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(k.startswith("premixer") for k in incompatible.missing_keys)

    with torch.no_grad():
        torch.testing.assert_close(premixed(x)["data"], baseline(x)["data"], rtol=0, atol=0)


def test_premixer_changes_output_once_trained(x):
    """With a non-zero output projection the pre-mixer must affect the model."""
    baseline = _build(with_premixer=False).eval()
    active = _build(with_premixer=True, initialise_out_zero=False).eval()
    active.load_state_dict(baseline.state_dict(), strict=False)

    with torch.no_grad():
        assert not torch.allclose(active(x)["data"], baseline(x)["data"])


def test_gradients_reach_the_premixer(x):
    model = _build(with_premixer=True, initialise_out_zero=False)
    model(x)["data"].sum().backward()

    premixer_params = [(n, p) for n, p in model.named_parameters() if n.startswith("premixer")]
    assert premixer_params, "no pre-mixer parameters found"
    assert all(p.grad is not None for _, p in premixer_params)
