# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.training.builder import ModelRuntimeArtifacts
from anemoi.training.builder import build_anemoi_model
from anemoi.training.builder import build_direct_model
from anemoi.training.config_bundle import ModelConfigBundle
from anemoi.training.config_bundle import TaskConfigBundle
from anemoi.training.runtime import TaskRuntimeArtifacts
from anemoi.training.train.train import AnemoiTrainer

if TYPE_CHECKING:
    import pytest


class FakeAnemoiModel:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class DummyTrainer(AnemoiTrainer):
    @cached_property
    def profiler(self) -> None:
        return None


def test_build_anemoi_model_uses_injected_runtime_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = HeteroData()
    graph["data"].x = torch.zeros(1, 1)

    def _fake_instantiate(config: Any, **kwargs: Any) -> Any:
        if getattr(config, "_target_", None) == "tests.FakeAnemoiModel":
            return FakeAnemoiModel(**kwargs)
        return config

    monkeypatch.setattr("anemoi.training.builder.instantiate", _fake_instantiate)

    statistics = {"data": {}}
    data_indices = {"data": object()}
    metadata = {"metadata_inference": {"data": {"timesteps": {"relative_date_indices_training": [0]}}}}
    supporting_arrays = {"data": {}}
    runtime_artifacts = ModelRuntimeArtifacts(
        graph_data=graph,
        statistics=statistics,
        statistics_tendencies=None,
        data_indices=data_indices,
        metadata=metadata,
        supporting_arrays=supporting_arrays,
    )

    model = build_anemoi_model(
        config_bundle=ModelConfigBundle(
            training=OmegaConf.create({"multistep_input": 1, "multistep_output": 1}),
            data=OmegaConf.create({"processors": {}}),
            model=OmegaConf.create(
                {
                    "wrapper": {"_target_": "tests.FakeAnemoiModel"},
                    "backbone": {"_target_": "unused"},
                    "keep_batch_sharded": False,
                    "output_mask": {"_target_": "anemoi.training.utils.masks.NoOutputMask"},
                },
            ),
        ),
        runtime_artifacts=runtime_artifacts,
    )

    assert model.kwargs["graph_data"] is graph
    assert model.kwargs["statistics"] is statistics
    assert model.kwargs["data_indices"] is data_indices
    assert model.kwargs["metadata"] is metadata
    assert model.kwargs["supporting_arrays"] == supporting_arrays


def test_trainer_model_passes_runtime_artifacts_to_model_instantiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = HeteroData()
    graph["data"].x = torch.zeros(1, 1)

    captured_kwargs = {}
    fake_model = object()

    def _capture_instantiate(config: Any, **kwargs: Any) -> Any:
        captured_kwargs["target"] = config._target_
        captured_kwargs["kwargs"] = kwargs
        return fake_model

    class FakeTask:
        task_type = "forecaster"

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr("anemoi.training.train.train.instantiate", _capture_instantiate)
    monkeypatch.setattr("anemoi.training.train.train.get_class", lambda _path: FakeTask)

    trainer = DummyTrainer.__new__(DummyTrainer)
    runtime_artifacts = ModelRuntimeArtifacts(
        graph_data=graph,
        statistics={"data": {}},
        statistics_tendencies=None,
        data_indices={"data": object()},
        metadata={"metadata_inference": {}},
        supporting_arrays={"data": {}},
    )
    trainer.config = OmegaConf.create(
        {
            "training": {
                "model_task": "unused",
                "transfer_learning": False,
            },
            "data": {},
            "system": {"hardware": {}},
            "dataloader": {},
            "graph": {},
            "model_builder": {
                "_target_": "anemoi.training.builder.build_direct_model",
            },
            "model": {
                "_target_": "anemoi.models.models.naive.NaiveModel",
            },
        },
    )
    trainer.runtime_artifacts = runtime_artifacts
    trainer.load_weights_only = False

    task = trainer.model

    assert captured_kwargs["target"] == "anemoi.training.builder.build_direct_model"
    assert captured_kwargs["kwargs"]["runtime_artifacts"] is runtime_artifacts
    assert isinstance(captured_kwargs["kwargs"]["config_bundle"], ModelConfigBundle)
    assert task.kwargs["model"] is fake_model
    assert isinstance(task.kwargs["runtime_artifacts"], TaskRuntimeArtifacts)
    assert isinstance(task.kwargs["config_bundle"], TaskConfigBundle)
    assert task.kwargs["runtime_artifacts"].graph_data is runtime_artifacts.graph_data
    assert task.kwargs["runtime_artifacts"].data_indices is runtime_artifacts.data_indices
    assert runtime_artifacts.metadata["metadata_inference"]["task"] == "forecaster"


def test_build_direct_model_uses_model_spec_from_config_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = HeteroData()
    graph["data"].x = torch.zeros(1, 1)

    captured_kwargs = {}
    fake_model = object()

    def _capture_instantiate(config: Any, **kwargs: Any) -> Any:
        captured_kwargs["target"] = config._target_
        captured_kwargs["kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr("anemoi.training.builder.instantiate", _capture_instantiate)

    runtime_artifacts = ModelRuntimeArtifacts(
        graph_data=graph,
        statistics={"data": {}},
        statistics_tendencies=None,
        data_indices={"data": object()},
        metadata={"metadata_inference": {}},
        supporting_arrays={"data": {}},
    )

    model = build_direct_model(
        config_bundle=ModelConfigBundle(
            training=OmegaConf.create({"multistep_input": 1, "multistep_output": 1}),
            data=OmegaConf.create({}),
            model=OmegaConf.create({"_target_": "anemoi.models.models.naive.NaiveModel", "n_input": 1}),
        ),
        runtime_artifacts=runtime_artifacts,
    )

    assert model is fake_model
    assert captured_kwargs["target"] == "anemoi.models.models.naive.NaiveModel"
    assert captured_kwargs["kwargs"]["runtime_artifacts"] is runtime_artifacts
