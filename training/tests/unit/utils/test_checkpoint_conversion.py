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
from torch_geometric.data import HeteroData

from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils import checkpoint as checkpoint_utils
from anemoi.training.utils.checkpoint import load_and_prepare_model

CKPT_PATH = "training.ckpt"


def _graph(*node_types: str) -> HeteroData:
    graph = HeteroData()
    for node_type in node_types:
        graph[node_type].x = torch.zeros(2, 1)
    return graph


@pytest.fixture
def load_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record the kwargs given to ``load_from_checkpoint`` and return a minimal module."""
    calls: list[dict] = []

    def fake_load_from_checkpoint(_path: str, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(model=SimpleNamespace(metadata={"uuid": "abc"}, config=object()))

    monkeypatch.setattr(BaseTrainingModule, "load_from_checkpoint", fake_load_from_checkpoint)
    return calls


def _set_hyper_parameters(monkeypatch: pytest.MonkeyPatch, hyper_parameters: dict) -> None:
    method = SimpleNamespace(_target_="anemoi.training.train.methods.base.BaseTrainingModule")
    hyper_parameters = {"config": SimpleNamespace(training=SimpleNamespace(method=method)), **hyper_parameters}
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: {"hyper_parameters": hyper_parameters})


def test_stored_graph_is_used_by_default(monkeypatch: pytest.MonkeyPatch, load_calls: list[dict]) -> None:
    _set_hyper_parameters(monkeypatch, {"graph_data": _graph("data"), "data_indices": {"data": None}})

    _, metadata = load_and_prepare_model(CKPT_PATH)

    assert metadata == {"uuid": "abc"}
    assert "graph_data" not in load_calls[0]


def test_graph_argument_overrides_stored_graph(monkeypatch: pytest.MonkeyPatch, load_calls: list[dict]) -> None:
    graph = _graph("data")
    _set_hyper_parameters(monkeypatch, {"graph_data": _graph("data"), "data_indices": {"data": None}})
    monkeypatch.setattr(checkpoint_utils, "load_graph_from_file", lambda _path: graph)

    load_and_prepare_model(CKPT_PATH, graph_path="graph.pt")

    assert load_calls[0]["graph_data"] is graph


def test_graph_argument_replaces_missing_stored_graph(monkeypatch: pytest.MonkeyPatch, load_calls: list[dict]) -> None:
    graph = _graph("data")
    _set_hyper_parameters(monkeypatch, {"data_indices": {"data": None}})
    monkeypatch.setattr(checkpoint_utils, "load_graph_from_file", lambda _path: graph)

    load_and_prepare_model(CKPT_PATH, graph_path="graph.pt")

    assert load_calls[0]["graph_data"] is graph


def test_graph_argument_with_fused_datasets(monkeypatch: pytest.MonkeyPatch, load_calls: list[dict]) -> None:
    graph = _graph("era", "cerra", "hidden")
    _set_hyper_parameters(monkeypatch, {"data_indices": {"era": None, "cerra": None}})
    monkeypatch.setattr(checkpoint_utils, "load_graph_from_file", lambda _path: graph)

    load_and_prepare_model(CKPT_PATH, graph_path="graph.pt")

    assert load_calls[0]["graph_data"] is graph


def test_missing_graph_everywhere_raises(monkeypatch: pytest.MonkeyPatch, load_calls: list[dict]) -> None:
    _set_hyper_parameters(monkeypatch, {"data_indices": {"data": None}})

    with pytest.raises(ValueError, match="--graph"):
        load_and_prepare_model(CKPT_PATH)

    assert not load_calls


def test_graph_without_required_dataset_nodes_raises(monkeypatch: pytest.MonkeyPatch, load_calls: list[dict]) -> None:
    _set_hyper_parameters(monkeypatch, {"data_indices": {"era": None, "cerra": None}})
    monkeypatch.setattr(checkpoint_utils, "load_graph_from_file", lambda _path: _graph("era", "hidden"))

    with pytest.raises(ValueError, match="missing dataset node types"):
        load_and_prepare_model(CKPT_PATH, graph_path="graph.pt")

    assert not load_calls
