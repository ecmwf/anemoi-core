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


class _IndexGroup(SimpleNamespace):
    def __len__(self):
        return len(self.prognostic)


def _make_data_indices(dataset_names: tuple[str, ...] = ("data",)) -> dict:
    def _indices() -> SimpleNamespace:
        return SimpleNamespace(
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

    return {name: _indices() for name in dataset_names}


def _make_graph(dataset_names: tuple[str, ...] = ("data",)) -> HeteroData:
    graph = HeteroData()
    for name in dataset_names:
        graph[name].x = torch.zeros(2, 2)
        graph[name].num_nodes = 2
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
            "model": {
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


def test_base_graph_model_accepts_omegaconf_hidden_node_lists() -> None:
    model_config = OmegaConf.create(
        {
            "model": {
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
        },
    )

    model = DummyGraphModel(
        model_config=model_config,
        data_indices=_make_data_indices(),
        statistics={"data": None},
        n_step_input=1,
        n_step_output=1,
        graph_data=_make_hierarchical_graph(),
    )

    assert list(model.seen_hidden_name) == ["hidden_1", "hidden_2", "hidden_3"]
    assert model.node_attributes.num_nodes["hidden_3"] == 1


# ---------------------------------------------------------------------------
# encoder routing and dataset fusion
# ---------------------------------------------------------------------------


class FusingGraphModel(DummyGraphModel):
    supports_encoder_fusion = True


class CrossDatasetInputDimModel(DummyGraphModel):
    """Sizes every encoder from the channel counts of *all* datasets."""

    def _calculate_input_dim(self, dataset_name: str) -> int:
        return sum(self.num_input_channels.values())


def _build_dummy_model(
    *,
    source_datasets: tuple[str, ...] = ("data",),
    fusing_strategy: str = "not_supported",
    fusion_anchor: str | None = None,
    model_cls: type[BaseGraphModel] = DummyGraphModel,
) -> BaseGraphModel:
    encoder_config: dict = {
        "source_datasets": list(source_datasets),
        "dataset_fusing_strategy": fusing_strategy,
        "mapper": {},
    }
    if fusion_anchor is not None:
        encoder_config["fusion_anchor"] = fusion_anchor

    model_config = OmegaConf.create(
        {
            "model": {
                "node_trainable_parameters": {name: 0 for name in (*source_datasets, "hidden")},
                "model": {"hidden_nodes_name": "hidden", "latent_skip": False},
                "encoders": {0: encoder_config},
                "decoders": {
                    0: {
                        "target_datasets": [source_datasets[0]],
                        "target_node_features": ["coordinates"],
                        "mapper": {},
                    },
                },
                "residual": {
                    "datasets": {
                        name: {"_target_": "anemoi.models.layers.residual.SkipConnection"} for name in source_datasets
                    },
                },
                "bounding": {"datasets": {name: [] for name in source_datasets}},
            },
        },
    )
    return model_cls(
        model_config=model_config,
        data_indices=_make_data_indices(source_datasets),
        statistics={name: None for name in source_datasets},
        n_step_input=1,
        n_step_output=1,
        graph_data=_make_graph(source_datasets),
    )


# Each value is a substring of "not_supported", so they all slip through a
# membership test written against the bare string instead of a tuple of strings.
@pytest.mark.parametrize("fusing_strategy", ["", "not", "supported", "_supported", "t_suppo"])
def test_rejects_fusing_strategy_that_is_a_substring_of_a_supported_one(fusing_strategy: str) -> None:
    with pytest.raises(ValueError, match="unsupported fusing strategy"):
        _build_dummy_model(fusing_strategy=fusing_strategy)


def test_without_fusion_every_source_dataset_is_its_own_anchor() -> None:
    model = _build_dummy_model(source_datasets=("data", "extra"))

    assert model.encoder2anchors == {0: ["data", "extra"]}
    assert model.input_datasets == ["data", "extra"]


def test_fusion_anchor_without_a_fusion_strategy_is_rejected() -> None:
    """Silently ignoring the anchor would hide a real misconfiguration."""
    with pytest.raises(ValueError, match="fusion_anchor"):
        _build_dummy_model(source_datasets=("data", "extra"), fusion_anchor="data")


def test_fusion_is_rejected_by_model_classes_that_do_not_implement_it() -> None:
    with pytest.raises(ValueError, match="does not support"):
        _build_dummy_model(
            source_datasets=("data", "extra"),
            fusing_strategy="concatenate_inputs_along_variable_dim",
            fusion_anchor="data",
        )


def test_fusion_routes_every_source_dataset_through_the_anchor_node_set() -> None:
    model = _build_dummy_model(
        source_datasets=("data", "extra"),
        fusing_strategy="concatenate_inputs_along_variable_dim",
        fusion_anchor="data",
        model_cls=FusingGraphModel,
    )

    # Only the anchor owns encoder edges and an encoder input dimension.
    assert model.encoder2anchors == {0: ["data"]}
    assert model.input_datasets == ["data"]
    # Every fused dataset still resolves to the encoder, on the anchor's node set.
    assert model.encoder2datasets == {0: ["data", "extra"]}
    assert model.dataset2encoder == {"data": 0, "extra": 0}


def test_fusion_requires_an_anchor() -> None:
    with pytest.raises(ValueError, match="must set fusion_anchor"):
        _build_dummy_model(
            source_datasets=("data", "extra"),
            fusing_strategy="concatenate_inputs_along_variable_dim",
            model_cls=FusingGraphModel,
        )


def test_fusion_anchor_must_be_one_of_the_source_datasets() -> None:
    with pytest.raises(ValueError, match="not in source_datasets"):
        _build_dummy_model(
            source_datasets=("data", "extra"),
            fusing_strategy="concatenate_inputs_along_variable_dim",
            fusion_anchor="hidden",
            model_cls=FusingGraphModel,
        )


def test_calculate_shapes_and_indices_fills_channel_counts_before_any_dimension() -> None:
    """Dimension hooks may read channel counts of datasets other than their own.

    If dims were computed in the same loop that fills ``num_input_channels``,
    the first dataset would only ever see its own count.
    """
    model = _build_dummy_model(source_datasets=("data", "extra"), model_cls=CrossDatasetInputDimModel)

    assert model.input_dim == {"data": 2, "extra": 2}


# ---------------------------------------------------------------------------
# predict_step — spatial preprocessor ordering
# ---------------------------------------------------------------------------


def _make_minimal_model():
    """Return a DummyGraphModel with a working predict_step."""
    model_config = OmegaConf.create(
        {
            "model": {
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
            },
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
