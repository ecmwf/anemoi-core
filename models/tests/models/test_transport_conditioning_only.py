# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Real-instantiation tests for conditioning-only datasets in the base transport model.

A conditioning-only dataset is encoded (its latent joins the additive merge) but is never
predicted: it has no decoder, no noised (corrupted) target concatenated to its encoder input,
and never appears in the forward output dict. The feature is general to the transport family and
is driven purely by the explicit ``model.model.conditioning_only_datasets`` config list.

These tests construct :class:`AnemoiTransportModelEncProcDec` through its real ``__init__`` with a
real graph and real graph-transformer encoders/decoders/processor, with two datasets:

- ``aux``  — conditioning-only (its own small grid), and
- ``output`` — predicted.

Design note verified here: an input-only dataset still carries a non-empty ``model.output``
(``aux``'s prognostic ``u, v``), so conditioning-only cannot be inferred from ``data_indices`` — it
must be declared explicitly. That is exactly why the config list is the source of truth.
"""

import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec

# ── grid sizes ────────────────────────────────────────────────────────────────
N_AUX = 5  # conditioning-only dataset grid
N_OUTPUT = 8  # predicted dataset grid (different size on purpose)
N_HIDDEN = 4  # tiny hidden mesh

NUM_CHANNELS = 16
NOISE_CHANNELS = 8
NOISE_COND_DIM = 4
N_STEP_INPUT = 2
N_STEP_OUTPUT = 1

# ── index collections ─────────────────────────────────────────────────────────
# aux "conditioning-only": u(0) v(1) sforce(2), forcing=[sforce] -> prognostic=[u, v]
#   model.input = [u, v, sforce] (3), model.output = [u, v] (2) -> NON-EMPTY output even though
#   this dataset is only ever an input. This is what makes explicit config necessary.
# output "predicted": u(0) v(1) force(2), forcing=[force] -> prognostic=[u, v]
#   model.input = [u, v, force] (3), model.output = [u, v] (2)
_AUX_NAME_TO_INDEX = {"u": 0, "v": 1, "sforce": 2}
_OUTPUT_NAME_TO_INDEX = {"u": 0, "v": 1, "force": 2}


def _aux_indices() -> IndexCollection:
    cfg = DictConfig({"forcing": ["sforce"], "diagnostic": [], "target": []})
    return IndexCollection(cfg, dict(_AUX_NAME_TO_INDEX))


def _output_indices() -> IndexCollection:
    cfg = DictConfig({"forcing": ["force"], "diagnostic": [], "target": []})
    return IndexCollection(cfg, dict(_OUTPUT_NAME_TO_INDEX))


def _stats(n: int) -> dict:
    import numpy as np

    return {
        "mean": np.zeros(n, dtype=np.float32),
        "stdev": np.ones(n, dtype=np.float32),
        "minimum": np.zeros(n, dtype=np.float32),
        "maximum": np.ones(n, dtype=np.float32),
    }


def _coords(n: int, offset: float) -> torch.Tensor:
    lats = torch.linspace(-1.0, 1.0, n) + offset
    lons = torch.linspace(0.0, 2.0, n) - offset
    return torch.stack((lats, lons), dim=-1).float()


def _dense_bipartite_edges(n_src: int, n_dst: int) -> torch.Tensor:
    src = torch.arange(n_src).repeat_interleave(n_dst)
    dst = torch.arange(n_dst).repeat(n_src)
    return torch.stack((src, dst), dim=0)


def _graph() -> HeteroData:
    graph = HeteroData()
    for name, n, offset in (("aux", N_AUX, 0.0), ("output", N_OUTPUT, 0.1), ("hidden", N_HIDDEN, 0.2)):
        graph[name].x = _coords(n, offset)
        graph[name].num_nodes = n

    sizes = {"aux": N_AUX, "output": N_OUTPUT, "hidden": N_HIDDEN}
    # aux has NO hidden->aux decoder edges on purpose: conditioning-only datasets are never decoded.
    pairs = [
        ("aux", "hidden"),
        ("output", "hidden"),
        ("hidden", "output"),
        ("hidden", "hidden"),
    ]
    generator = torch.Generator().manual_seed(7)
    for src, dst in pairs:
        edge_index = _dense_bipartite_edges(sizes[src], sizes[dst])
        store = graph[(src, "to", dst)]
        store.edge_index = edge_index
        store.edge_length = torch.rand((edge_index.shape[1], 1), generator=generator)
    return graph


def _layer_kernels() -> dict:
    return {
        "LayerNorm": {
            "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
            "normalized_shape": NUM_CHANNELS,
            "condition_shape": NOISE_COND_DIM,
            "zero_init": True,
            "autocast": False,
        },
        "Linear": {"_target_": "torch.nn.Linear"},
        "Activation": {"_target_": "torch.nn.GELU"},
        "QueryNorm": {"_target_": "anemoi.models.layers.normalization.AutocastLayerNorm", "bias": False},
        "KeyNorm": {"_target_": "anemoi.models.layers.normalization.AutocastLayerNorm", "bias": False},
    }


def _model_config(conditioning_only_datasets: list[str]) -> DictConfig:
    layer_kernels = _layer_kernels()
    mapper_common = {
        "trainable_size": 0,
        "sub_graph_edge_attributes": ["edge_length"],
        "num_chunks": 1,
        "mlp_hidden_ratio": 2,
        "mlp_implementation": "mlp",
        "num_heads": 4,
        "qk_norm": True,
        "cpu_offload": False,
        "gradient_checkpointing": False,
        "layer_kernels": layer_kernels,
        "shard_strategy": "edges",
        "graph_attention_backend": "pyg",
    }
    return DictConfig(
        {
            "model": {
                "num_channels": NUM_CHANNELS,
                "cpu_offload": False,
                "keep_batch_sharded": False,
                "model": {
                    "hidden_nodes_name": "hidden",
                    "latent_skip": True,
                    "conditioning_only_datasets": list(conditioning_only_datasets),
                    "transport": {
                        "objective": "edm_diffusion",
                        "sigma_data": 1.0,
                        "noise_channels": NOISE_CHANNELS,
                        "noise_cond_dim": NOISE_COND_DIM,
                        "noise_embedder": {
                            "_target_": "anemoi.models.layers.diffusion.SinusoidalEmbeddings",
                            "num_channels": NOISE_CHANNELS,
                            "max_period": 1000,
                        },
                    },
                },
                "layer_kernels": layer_kernels,
                "encoder": {
                    "_target_": "anemoi.models.layers.mapper.GraphTransformerForwardMapper",
                    **mapper_common,
                },
                "decoder": {
                    "_target_": "anemoi.models.layers.mapper.GraphTransformerBackwardMapper",
                    "initialise_data_extractor_zero": False,
                    **mapper_common,
                },
                "processor": {
                    "_target_": "anemoi.models.layers.processor.GraphTransformerProcessor",
                    "num_layers": 1,
                    **mapper_common,
                },
                "residual": {"_target_": "anemoi.models.layers.residual.SkipConnection"},
                "trainable_parameters": {
                    "data": 0,
                    "hidden": 0,
                    "data2hidden": 0,
                    "hidden2data": 0,
                    "hidden2hidden": 0,
                },
                "attributes": {"edges": ["edge_length"], "nodes": []},
                "bounding": [],
            },
        },
    )


def _build_model(conditioning_only_datasets: list[str]) -> AnemoiTransportModelEncProcDec:
    data_indices = {"aux": _aux_indices(), "output": _output_indices()}
    statistics = {"aux": _stats(len(_AUX_NAME_TO_INDEX)), "output": _stats(len(_OUTPUT_NAME_TO_INDEX))}
    return AnemoiTransportModelEncProcDec(
        model_config=_model_config(conditioning_only_datasets),
        data_indices=data_indices,
        statistics=statistics,
        n_step_input=N_STEP_INPUT,
        n_step_output=N_STEP_OUTPUT,
        graph_data=_graph(),
    )


@pytest.fixture
def model() -> AnemoiTransportModelEncProcDec:
    return _build_model(["aux"])


def _expected_input_dim(model: AnemoiTransportModelEncProcDec, dataset: str, conditioning_only: bool) -> int:
    """Hand formula for the encoder input dim.

    history = n_step_input * len(model.input) + node-attribute dims; predicted datasets add a
    corrupted-target term n_step_output * len(model.output); conditioning-only datasets do not.
    """
    history = N_STEP_INPUT * len(model.data_indices[dataset].model.input) + model.node_attributes.attr_ndims[dataset]
    if conditioning_only:
        return history
    return history + N_STEP_OUTPUT * len(model.data_indices[dataset].model.output)


# ── test 1: construction, decoder membership, encoder input dims ───────────────
def test_construction_decoder_membership_and_input_dims(model: AnemoiTransportModelEncProcDec) -> None:
    # The conditioning-only design cannot be inferred from data_indices: aux still has model.output.
    assert len(model.data_indices["aux"].model.output) == 2  # u, v — non-empty despite input-only role
    assert model.conditioning_only_datasets == {"aux"}

    # aux is encoded but never decoded; output is decoded.
    assert "aux" in model.encoder
    assert "output" in model.encoder
    assert "aux" not in model.decoder
    assert "output" in model.decoder
    assert "aux" not in model.decoder_graph_provider
    assert "output" in model.decoder_graph_provider

    # aux input dim has NO corrupted-target term; output input dim is unchanged (has the term).
    expected_aux = _expected_input_dim(model, "aux", conditioning_only=True)
    expected_output = _expected_input_dim(model, "output", conditioning_only=False)
    assert model.input_dim["aux"] == expected_aux
    assert model.input_dim["output"] == expected_output

    # The corrupted-target term is exactly what separates the two: aux lacks it, output has it.
    aux_history = N_STEP_INPUT * len(model.data_indices["aux"].model.input) + model.node_attributes.attr_ndims["aux"]
    assert model.input_dim["aux"] == aux_history  # no + n_step_output * num_output_channels
    output_corrupted_term = N_STEP_OUTPUT * len(model.data_indices["output"].model.output)
    output_history = (
        N_STEP_INPUT * len(model.data_indices["output"].model.input) + model.node_attributes.attr_ndims["output"]
    )
    assert model.input_dim["output"] == output_history + output_corrupted_term

    # The real encoders were built with those dims (only a real instantiation can check this).
    assert model.encoder["aux"].in_channels_src == expected_aux
    assert model.encoder["output"].in_channels_src == expected_output


def _forward_inputs(batch_size: int = 2) -> tuple[dict, dict, dict]:
    generator = torch.Generator().manual_seed(11)
    n_in_aux = len(_aux_indices().model.input)  # 3
    n_in_output = len(_output_indices().model.input)  # 3
    n_out_output = len(_output_indices().model.output)  # 2
    x = {
        "aux": torch.randn(batch_size, N_STEP_INPUT, 1, N_AUX, n_in_aux, generator=generator),
        "output": torch.randn(batch_size, N_STEP_INPUT, 1, N_OUTPUT, n_in_output, generator=generator),
    }
    # conditioned_target and condition contain ONLY the predicted dataset (mirrors the objective).
    conditioned_target = {
        "output": torch.randn(batch_size, N_STEP_OUTPUT, 1, N_OUTPUT, n_out_output, generator=generator),
    }
    condition = {"output": torch.full((batch_size, 1, 1, 1, 1), 0.5)}
    return x, conditioned_target, condition


# ── test 2: full forward returns only the decoded dataset ──────────────────────
def test_forward_returns_only_predicted_dataset(model: AnemoiTransportModelEncProcDec) -> None:
    x, conditioned_target, condition = _forward_inputs()
    n_out_output = len(model.data_indices["output"].model.output)

    out = model.forward(x, conditioned_target, condition)

    assert set(out.keys()) == {"output"}  # aux is absent from the result
    assert out["output"].shape == (2, N_STEP_OUTPUT, 1, N_OUTPUT, n_out_output)
    assert torch.isfinite(out["output"]).all()


# ── test 3: a predicted dataset missing its conditioned target still raises loudly ──
def test_missing_conditioned_target_for_predicted_dataset_raises(model: AnemoiTransportModelEncProcDec) -> None:
    x, _, condition = _forward_inputs()

    # "output" is NOT conditioning-only, so a missing corrupted target must raise (loud contract).
    with pytest.raises(AssertionError, match="not conditioning-only"):
        model._forward_transport_network(x, {}, condition)


# ── test 4: an unknown conditioning-only name raises at construction ───────────
def test_unknown_conditioning_only_dataset_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="conditioning_only_datasets references unknown datasets"):
        _build_model(["ghost"])
