# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``model.categorical_embeddings`` swaps raw code columns for learned embeddings in the encoder input."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import HeteroData

import anemoi.models.models.autoencoder  # noqa: F401
import anemoi.models.models.ens_encoder_processor_decoder  # noqa: F401
import anemoi.models.models.hierarchical  # noqa: F401
import anemoi.models.models.transport_encoder_processor_decoder  # noqa: F401
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.categorical import CategoricalEmbedding
from anemoi.models.layers.categorical import check_categorical_preprocessing
from anemoi.models.models.autoencoder import AnemoiModelAutoEncoder
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec
from anemoi.models.models.hierarchical import AnemoiModelEncProcDecHierarchical
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec

_GRID, _HIDDEN, _STEPS = 6, 3, 2
# Data order: prognostic a, b; forcing f; corrector rt (codes) and geom.
_NAME_TO_INDEX = {"a": 0, "rt": 1, "f": 2, "geom": 3, "b": 4}
_CODES = [49001, 21009, 1004]


def _data_indices() -> dict:
    data_config = OmegaConf.create({"forcing": ["f"], "diagnostic": [], "corrector": ["rt", "geom"]})
    return {"data": IndexCollection(data_config, _NAME_TO_INDEX)}


def _graph() -> HeteroData:
    torch.manual_seed(0)
    graph = HeteroData()
    graph["data"].x = torch.rand(_GRID, 2)
    graph["hidden"].x = torch.rand(_HIDDEN, 2)
    for src, dst, n_src, n_dst in [
        ("data", "hidden", _GRID, _HIDDEN),
        ("hidden", "hidden", _HIDDEN, _HIDDEN),
        ("hidden", "data", _HIDDEN, _GRID),
    ]:
        dst_idx = torch.arange(max(n_src, n_dst)) % n_dst
        src_idx = torch.arange(max(n_src, n_dst)) % n_src
        graph[src, "to", dst].edge_index = torch.stack([src_idx, dst_idx])
        graph[src, "to", dst].edge_length = torch.rand(src_idx.numel(), 1)
    return graph


def _config(categorical: dict | None = None) -> OmegaConf:
    block = {
        "trainable_size": 0,
        "sub_graph_edge_attributes": ["edge_length"],
        "num_chunks": 1,
        "mlp_extra_layers": 0,
        "mlp_hidden_ratio": 1.0,
        "cpu_offload": False,
        "gradient_checkpointing": False,
        "layer_kernels": {},
    }
    model = {
        "num_channels": 8,
        "trainable_parameters": {"data": 0, "hidden": 0, "data2hidden": 0, "hidden2data": 0, "hidden2hidden": 0},
        "model": {"hidden_nodes_name": "hidden", "latent_skip": True},
        "encoder": {"_target_": "anemoi.models.layers.mapper.GNNForwardMapper", **block},
        "processor": {"_target_": "anemoi.models.layers.processor.GNNProcessor", "num_layers": 1, **block},
        "decoder": {"_target_": "anemoi.models.layers.mapper.GNNBackwardMapper", **block},
        "residual": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1},
        "bounding": [],
    }
    if categorical is not None:
        model["categorical_embeddings"] = categorical
    return OmegaConf.create({"model": model})


def _build(categorical: dict | None = None) -> AnemoiModelEncProcDec:
    return AnemoiModelEncProcDec(
        model_config=_config(categorical),
        data_indices=_data_indices(),
        statistics={"data": None},
        n_step_input=_STEPS,
        n_step_output=1,
        graph_data=_graph(),
    )


_SPEC = {"data": {"rt": {"codes": _CODES, "embedding_dim": 4, "unknown_prob": 0.0}}}


def _encoder_in_features(model: AnemoiModelEncProcDec) -> int:
    return next(m for m in model.encoder["data"].emb_nodes_src.modules() if isinstance(m, nn.Linear)).in_features


def _input(rt_codes: list[float]) -> torch.Tensor:
    """(batch, time, ensemble, grid, vars) in model-input order a, rt, f, geom, b."""
    x = torch.randn(1, _STEPS, 1, _GRID, 5)
    x[..., 1] = torch.tensor(rt_codes)
    return x


def test_input_dim_without_key_is_unchanged() -> None:
    model = _build()
    attr = model.node_attributes.attr_ndims["data"]
    assert model.input_dim["data"] == _STEPS * 5 + attr
    assert _encoder_in_features(model) == model.input_dim["data"]
    assert not any("categorical" in key for key in model.state_dict())


def test_input_dim_and_forward_with_embedding() -> None:
    model = _build(_SPEC)
    attr = model.node_attributes.attr_ndims["data"]
    # One raw rt channel becomes 4 embedding channels at each input step.
    assert model.input_dim["data"] == _STEPS * (5 - 1 + 4) + attr
    assert _encoder_in_features(model) == model.input_dim["data"]
    assert "categorical_embeddings.data.rt.embedding.weight" in model.state_dict()

    out = model({"data": _input([49001, 21009, 1004, 0, 7, 49001])})
    assert out["data"].shape == (1, 1, 1, _GRID, 2)


def test_zeroed_codes_use_missing_row() -> None:
    model = _build(_SPEC)
    x = _input([0.0] * _GRID)
    embedded = model._embed_categorical_inputs(x, "data")
    # Continuous columns a, f, geom, b first, then the 4 embedding channels.
    torch.testing.assert_close(embedded[..., :4], x[..., [0, 2, 3, 4]])
    missing_row = model.categorical_embeddings["data"]["rt"].embedding.weight[CategoricalEmbedding.MISSING]
    torch.testing.assert_close(embedded[..., 4:], missing_row.expand_as(embedded[..., 4:]))


def test_gradient_reaches_only_used_rows() -> None:
    model = _build(_SPEC)
    model({"data": _input([49001, 49001, 0, 0, 0, 0])})["data"].sum().backward()
    grad = model.categorical_embeddings["data"]["rt"].embedding.weight.grad
    used = (grad.abs().sum(-1) > 0).tolist()
    assert used == [True, False, True, False, False]  # MISSING and 49001 only


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:  # noqa: ARG002, FBT001, FBT002
        return x


def test_predict_step_with_raw_codes() -> None:
    model = _build(_SPEC).eval()
    batch = _input([49001, 21009, 1004, 0, 7, 49001])[:, :, 0]  # 4-D raw batch
    processors = nn.ModuleDict({"data": _Identity()})
    out = model.predict_step({"data": batch}, processors, processors, n_step_input=_STEPS)
    assert out["data"].shape == (1, 1, 1, _GRID, 2)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"other": {"rt": {"codes": _CODES}}}, "unknown dataset"),
        ({"data": {"nope": {"codes": _CODES}}}, "not model inputs"),
    ],
)
def test_invalid_config(spec: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _build(spec)


def test_unsupported_models_refuse_the_key() -> None:
    assert AnemoiModelEncProcDec.supports_categorical_embeddings
    assert AnemoiModelEncProcDecHierarchical.supports_categorical_embeddings
    for cls in (AnemoiEnsModelEncProcDec, AnemoiModelAutoEncoder, AnemoiTransportModelEncProcDec):
        assert not cls.supports_categorical_embeddings, cls.__name__


def _processors(norm_mul: list[float], norm_add: list[float], fill: list[float]) -> SimpleNamespace:
    normalizer = SimpleNamespace(_norm_mul=torch.tensor(norm_mul), _norm_add=torch.tensor(norm_add))
    imputer = SimpleNamespace(imputation_values_training=torch.tensor(fill))
    return SimpleNamespace(processors={"normalizer": normalizer, "imputer": imputer})


def test_preprocessing_check() -> None:
    indices = _data_indices()["data"]
    identity = [1.0] * 5
    zeros = [0.0] * 5
    check_categorical_preprocessing("data", ["rt"], _processors(identity, zeros, zeros), indices)

    scaled = [1.0, 0.5, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="`none` list"):
        check_categorical_preprocessing("data", ["rt"], _processors(scaled, zeros, zeros), indices)


def test_preprocessing_check_warns_on_nonzero_fill(caplog: pytest.LogCaptureFixture) -> None:
    indices = _data_indices()["data"]
    fill = [0.0, 3.0, 0.0, 0.0, 0.0]
    check_categorical_preprocessing("data", ["rt"], _processors([1.0] * 5, [0.0] * 5, fill), indices)
    assert "MISSING" in caplog.text
