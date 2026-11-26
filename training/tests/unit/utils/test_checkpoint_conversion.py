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
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils.checkpoint import load_and_prepare_model

CKPT_PATH = "training.ckpt"


@pytest.fixture
def load_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record the kwargs given to ``load_from_checkpoint`` and return a minimal module."""
    calls: list[dict] = []

    def fake_load_from_checkpoint(_path: str, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(model=SimpleNamespace(metadata={"uuid": "abc"}, config=object()))

    monkeypatch.setattr(BaseTrainingModule, "load_from_checkpoint", fake_load_from_checkpoint)
    return calls


def _set_hyper_parameters(monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    """Store a minimal training config (graph built from nodes) as the checkpoint hyperparameters."""
    config = OmegaConf.create(
        {
            "training": {"method": {"_target_": "anemoi.training.train.methods.base.BaseTrainingModule"}},
            "task": {"_target_": "types.SimpleNamespace", "name": "forecaster"},
            "system": {"input": {"graph": None}},
            "graph": {"nodes": {"data": {}, "hidden": {}}, "edges": []},
        },
    )
    OmegaConf.set_struct(config, True)
    hyper_parameters = {"config": config, "data_indices": {"data": None}}
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: {"hyper_parameters": hyper_parameters})
    return config


def test_graph_is_built_from_the_config_by_default(monkeypatch: pytest.MonkeyPatch, load_calls: list[dict]) -> None:
    _set_hyper_parameters(monkeypatch)

    _, metadata = load_and_prepare_model(CKPT_PATH)

    assert metadata == {"uuid": "abc"}
    assert set(load_calls[0]) == {"weights_only", "task"}  # the stored config is used as is
    # the task is not stored with the hyperparameters: it is built from config.task
    assert load_calls[0]["task"] == SimpleNamespace(name="forecaster")


def test_graph_argument_loads_the_graph_from_file(monkeypatch: pytest.MonkeyPatch, load_calls: list[dict]) -> None:
    stored_config = _set_hyper_parameters(monkeypatch)

    load_and_prepare_model(CKPT_PATH, graph_path="graph.pt")

    config = load_calls[0]["config"]
    assert load_calls[0]["task"] == SimpleNamespace(name="forecaster")
    assert config.system.input.graph == "graph.pt"
    assert config.graph.nodes is None  # existing-graph mode: the model loads system.input.graph
    # the stored config is not modified
    assert stored_config.system.input.graph is None
    assert set(stored_config.graph.nodes) == {"data", "hidden"}
