# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

import anemoi.models.models.base as base_model_module
from anemoi.models.models import AnemoiModelEncProcDecHierarchical
from anemoi.models.models.base import BaseGraphModel


class DummyGraphModel(BaseGraphModel):
    def _build_networks(self, model_config) -> None:
        self.seen_hidden_name = model_config.model.hidden_nodes_name

    def _assemble_input(self, x, batch_size, grid_shard_sizes=None, model_comm_group=None):
        return x

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        return x_out

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class DummyHierarchicalGraphModel(DummyGraphModel):
    _build_hidden_node_routing = AnemoiModelEncProcDecHierarchical._build_hidden_node_routing


class _IndexGroup(SimpleNamespace):
    def __len__(self):
        return len(self.prognostic)


def _make_data_indices() -> dict:
    dataset_indices = SimpleNamespace(
        model=SimpleNamespace(
            input=_IndexGroup(prognostic=[0], forcing=[]),
            output=_IndexGroup(prognostic=[0], full=[0], diagnostic=[], name_to_index={"var": 0}),
            _forcing=[],
        ),
        data=SimpleNamespace(
            input=SimpleNamespace(
                name_to_index={"var": 0},
            ),
        ),
        name_to_index={"var": 0},
    )
    return {"data": dataset_indices}


def _make_graph() -> HeteroData:
    graph = HeteroData()
    graph["data"].x = torch.zeros(2, 2)
    graph["data"].num_nodes = 2
    graph["hidden"].x = torch.zeros(1, 2)
    graph["hidden"].num_nodes = 1
    return graph


def _make_hierarchical_graph() -> HeteroData:
    graph = HeteroData()
    graph["data"].x = torch.zeros(2, 2)
    graph["data"].num_nodes = 2
    for hidden_name in ["hidden_1", "hidden_2", "hidden_3"]:
        graph[hidden_name].x = torch.zeros(1, 2)
        graph[hidden_name].num_nodes = 1
    return graph


def test_base_graph_model_builds_with_omegaconf_config() -> None:
    model_config = OmegaConf.create(
        {
            "node_trainable_parameters": {
                "data": 0,
                "hidden": 0,
            },
            "model": {
                "hidden_nodes_name": "hidden",
                "latent_skip": False,
            },
            "encoders": {
                0: {
                    "source_datasets": ["data"],
                    "dataset_fusing_strategy": "not_supported",
                    "mapper": {},
                },
            },
            "decoders": {
                0: {
                    "target_datasets": ["data"],
                    "target_node_features": ["coordinates"],
                    "mapper": {},
                },
            },
            "residual": {
                "datasets": {"data": {"_target_": "anemoi.models.layers.residual.SkipConnection"}},
            },
            "bounding": {"datasets": {"data": []}},
        },
    )

    model = DummyGraphModel(
        model_config=model_config,
        data_indices=_make_data_indices(),
        statistics={"data": None},
        n_step_input=1,
        n_step_output=1,
        graph_data=_make_graph(),
    )

    assert model.seen_hidden_name == "hidden"
    assert "data" in model.residual


def test_hierarchical_routing_accepts_omegaconf_hidden_node_lists() -> None:
    model_config = OmegaConf.create(
        {
            "num_channels": 8,
            "node_trainable_parameters": {
                "data": 0,
                "hidden": 0,
            },
            "model": {
                "hidden_nodes_name": ["hidden_1", "hidden_2", "hidden_3"],
                "latent_skip": False,
            },
            "encoders": {
                0: {
                    "source_datasets": ["data"],
                    "dataset_fusing_strategy": "not_supported",
                    "mapper": {},
                },
            },
            "decoders": {
                0: {
                    "target_datasets": ["data"],
                    "target_node_features": ["coordinates"],
                    "mapper": {},
                },
            },
            "residual": {
                "datasets": {"data": {"_target_": "anemoi.models.layers.residual.SkipConnection"}},
            },
            "bounding": {"datasets": {"data": []}},
        },
    )

    model = DummyHierarchicalGraphModel(
        model_config=model_config,
        data_indices=_make_data_indices(),
        statistics={"data": None},
        n_step_input=1,
        n_step_output=1,
        graph_data=_make_hierarchical_graph(),
    )

    assert model.hidden_levels == ["hidden_1", "hidden_2", "hidden_3"]
    assert model.node_attributes.num_nodes["hidden_3"] == 1


# ---------------------------------------------------------------------------
# predict_step — spatial preprocessor ordering
# ---------------------------------------------------------------------------


def _make_minimal_model():
    """Return a DummyGraphModel with a working predict_step."""
    model_config = OmegaConf.create(
        {
            "num_channels": 8,
            "node_trainable_parameters": {"data": 0, "hidden": 0},
            "model": {"hidden_nodes_name": "hidden", "latent_skip": False},
            "encoders": {
                0: {
                    "source_datasets": ["data"],
                    "dataset_fusing_strategy": "not_supported",
                    "mapper": {},
                },
            },
            "decoders": {
                0: {
                    "target_datasets": ["data"],
                    "target_node_features": ["coordinates"],
                    "mapper": {},
                },
            },
            "residual": {
                "datasets": {"data": {"_target_": "anemoi.models.layers.residual.SkipConnection"}},
            },
            "bounding": {"datasets": {"data": []}},
        }
    )
    return DummyGraphModel(
        model_config=model_config,
        data_indices=_make_data_indices(),
        statistics={"data": None},
        n_step_input=1,
        n_step_output=1,
        graph_data=_make_graph(),
    )


def _route_hidden_nodes(hidden_nodes_name, dataset_names=("a", "b"), model_class=BaseGraphModel) -> SimpleNamespace:
    """Run the hidden-node routing on a bare namespace, without building a model."""
    model = SimpleNamespace(dataset_names=list(dataset_names))
    model_class._build_hidden_node_routing(model, hidden_nodes_name)
    return model


def test_hidden_node_routing_string_applies_to_all_datasets() -> None:
    model = _route_hidden_nodes("hidden")

    assert model.dataset2hidden == {"a": "hidden", "b": "hidden"}
    assert model.hidden_names == ["hidden"]


@pytest.mark.parametrize("as_omegaconf", [False, True])
def test_hidden_node_routing_mapping_keeps_order_and_deduplicates(as_omegaconf) -> None:
    mapping = {"a": "hidden_1", "b": "hidden_2", "c": "hidden_1"}
    model = _route_hidden_nodes(OmegaConf.create(mapping) if as_omegaconf else mapping, ("a", "b", "c"))

    assert model.dataset2hidden == mapping
    assert model.hidden_names == ["hidden_1", "hidden_2"]


@pytest.mark.parametrize("as_omegaconf", [False, True])
def test_hierarchical_routing_encodes_every_dataset_into_the_first_level(as_omegaconf) -> None:
    levels = ["hidden_1", "hidden_2"]
    model = _route_hidden_nodes(
        OmegaConf.create(levels) if as_omegaconf else levels,
        model_class=AnemoiModelEncProcDecHierarchical,
    )

    assert model.hidden_levels == levels
    assert model.dataset2hidden == {"a": "hidden_1", "b": "hidden_1"}
    assert model.hidden_names == levels


def test_hidden_node_routing_rejects_mapping_missing_a_dataset() -> None:
    with pytest.raises(AssertionError, match=r"missing for datasets \['b'\]"):
        _route_hidden_nodes({"a": "hidden"})


def test_hidden_node_routing_rejects_mapping_with_unknown_dataset() -> None:
    with pytest.raises(AssertionError, match=r"unknown datasets \['c'\]"):
        _route_hidden_nodes({"a": "hidden", "b": "hidden", "c": "hidden"})


def test_hidden_node_routing_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="Hidden nodes name must be"):
        _route_hidden_nodes(3)


def test_active_hidden_name_for_datasets_sharing_a_mesh() -> None:
    model = _route_hidden_nodes({"a": "hidden_1", "b": "hidden_2", "c": "hidden_1"}, ("a", "b", "c"))

    assert BaseGraphModel._get_active_hidden_name(model, ["a"]) == "hidden_1"
    assert BaseGraphModel._get_active_hidden_name(model, ["a", "c"]) == "hidden_1"
    assert BaseGraphModel._get_active_hidden_name(model, ["b"]) == "hidden_2"


def test_active_hidden_name_rejects_batch_mixing_meshes() -> None:
    model = _route_hidden_nodes({"a": "hidden_1", "b": "hidden_2"})

    with pytest.raises(ValueError, match="same hidden mesh"):
        BaseGraphModel._get_active_hidden_name(model, ["a", "b"])


def _identity_pre_processor():
    """Pre-processor that returns its input unchanged (identity)."""

    class _Proc:
        def __call__(self, x, in_place=False):
            return x

    return _Proc()


def test_predict_step_spatial_preprocessors_called_before_normalization(monkeypatch):
    """Spatial preprocessors must be called before normalization preprocessors."""
    call_order = []

    class RecordingSpatialProcessor(nn.Module):
        def forward(self, x, model_comm_group=None, grid_shard_sizes=None):
            call_order.append("spatial")
            return x, grid_shard_sizes

    class RecordingPreProcessor:
        def __call__(self, x, in_place=False):
            call_order.append("pre")
            return x

    model = _make_minimal_model()

    # Patch forward so predict_step can complete without a real graph network.
    BATCH, TIME, GRID, VARS = 1, 1, 4, 1
    dummy_out = torch.zeros(BATCH, 1, 1, GRID, VARS)  # (b, t, ens, grid, vars)
    monkeypatch.setattr(model, "forward", lambda x, **kw: {"data": dummy_out})

    spatial_processors = nn.ModuleDict({"data": RecordingSpatialProcessor()})
    pre_processors = {"data": RecordingPreProcessor()}
    post_processors = {"data": _identity_pre_processor()}

    batch = {"data": torch.zeros(BATCH, TIME, GRID, VARS)}

    with torch.no_grad():
        model.predict_step(
            batch,
            pre_processors=pre_processors,
            post_processors=post_processors,
            n_step_input=TIME,
            spatial_pre_processors=spatial_processors,
        )

    assert call_order == ["spatial", "pre"], f"Expected spatial before pre, got order: {call_order}"


def test_predict_step_replaces_source_grid_shard_sizes(monkeypatch):
    source_grid_shard_sizes = [4, 4]
    target_grid_shard_sizes = [2, 2]

    class RegriddingSpatialProcessor(nn.Module):
        def forward(self, x, model_comm_group=None, grid_shard_sizes=None):
            assert model_comm_group is comm_group
            assert grid_shard_sizes == source_grid_shard_sizes
            return x[..., :2, :], target_grid_shard_sizes

    model = _make_minimal_model()
    comm_group = object()

    def forward(x, *, grid_shard_sizes=None, **_kwargs):
        assert grid_shard_sizes == {"data": target_grid_shard_sizes}
        return x

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(base_model_module, "get_shard_sizes", lambda *_args, **_kwargs: source_grid_shard_sizes)
    monkeypatch.setattr(base_model_module, "shard_tensor", lambda tensor, *_args, **_kwargs: tensor)

    out = model.predict_step(
        {"data": torch.zeros(1, 1, 8, 1)},
        pre_processors={"data": _identity_pre_processor()},
        post_processors={"data": _identity_pre_processor()},
        n_step_input=1,
        model_comm_group=comm_group,
        gather_out=False,
        spatial_pre_processors=nn.ModuleDict({"data": RegriddingSpatialProcessor()}),
    )

    assert out["data"].shape[-2] == 2


def test_predict_step_skips_input_only_datasets(monkeypatch):
    """Datasets without a decoder (input-only) have no output and must not be post-processed."""
    model = _make_minimal_model()

    # `forcing` feeds an encoder but has no decoder, so forward only returns `data`
    monkeypatch.setattr(model, "forward", lambda x, **kw: {"data": x["data"][:, -1:] + 1.0})

    processors = {"data": _identity_pre_processor(), "forcing": _identity_pre_processor()}
    batch = {"data": torch.zeros(1, 1, 4, 1), "forcing": torch.zeros(1, 1, 4, 2)}  # (batch, time, grid, vars)

    out = model.predict_step(batch, pre_processors=processors, post_processors=processors, n_step_input=1)

    assert set(out) == {"data"}
    assert torch.equal(out["data"], torch.ones(1, 1, 1, 4, 1))
