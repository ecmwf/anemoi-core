# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

import anemoi.models.models.base as base_model_module
from anemoi.models.models.base import BaseGraphModel


class DummyGraphModel(BaseGraphModel):
    def _build_networks(self, model_config) -> None:
        self.seen_hidden_name = model_config.model.model.hidden_nodes_name

    def _assemble_input(self, x, batch_size, grid_shard_sizes=None, model_comm_group=None):
        return x

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        return x_out

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _IndexGroup(SimpleNamespace):
    def __len__(self):
        return len(self.prognostic)


def _make_data_indices() -> dict:
    dataset_indices = SimpleNamespace(
        model=SimpleNamespace(
            input=_IndexGroup(prognostic=[0]),
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
            "model": {
                "num_channels": 8,
                "trainable_parameters": {
                    "data": 0,
                    "hidden": 0,
                },
                "model": {
                    "hidden_nodes_name": "hidden",
                    "latent_skip": False,
                },
                "residual": {
                    "_target_": "anemoi.models.layers.residual.SkipConnection",
                },
                "bounding": [],
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
                "trainable_parameters": {
                    "data": 0,
                    "hidden": 0,
                },
                "model": {
                    "hidden_nodes_name": ["hidden_1", "hidden_2", "hidden_3"],
                    "latent_skip": False,
                },
                "residual": {
                    "_target_": "anemoi.models.layers.residual.SkipConnection",
                },
                "bounding": [],
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
# predict_step — spatial preprocessor ordering
# ---------------------------------------------------------------------------


def _make_minimal_model():
    """Return a DummyGraphModel with a working predict_step."""
    model_config = OmegaConf.create(
        {
            "model": {
                "num_channels": 8,
                "trainable_parameters": {"data": 0, "hidden": 0},
                "model": {"hidden_nodes_name": "hidden", "latent_skip": False},
                "residual": {"_target_": "anemoi.models.layers.residual.SkipConnection"},
                "bounding": [],
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
