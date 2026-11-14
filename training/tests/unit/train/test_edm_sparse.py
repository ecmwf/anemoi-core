# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from anemoi.models.data.layout import TensorLayout
from anemoi.models.transport.settings import EdmSettings
from anemoi.training.train.methods.edm_diffusion import EDMDiffusionTransportObjective
from anemoi.training.train.methods.flow_matching import FlowMatchingTransportObjective
from anemoi.training.train.methods.transport_base import PreparedPredictionTarget
from anemoi.training.utils.index_space import IndexSpace
from tests.batch_builders import build_batch

if TYPE_CHECKING:
    import pytest


def _module() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                edm=EdmSettings(sigma_data=0.5),
                training_condition={"distribution": "linear", "sigma_max": 1.0, "sigma_min": 0.1},
            ),
        ),
        model_comm_group=None,
    )


def _fm_module() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                training_condition={"distribution": "uniform_time"},
            ),
        ),
        model_comm_group=None,
    )


def _prepared_sparse_target() -> PreparedPredictionTarget:
    """Model targets of a grid and of observations, two samples each.

    The observations of each sample are ``(ensemble, observations, variables)``, all in one output window.
    """
    obs = [torch.ones(1, 2, 1), torch.full((1, 4, 1), 2.0)]
    batch = build_batch(
        data={"grid": torch.ones(2, 1, 1, 3, 1), "obs": obs},
        coordinates={"grid": torch.zeros(3, 2), "obs": [torch.zeros(sample.shape[1], 2) for sample in obs]},
        timedeltas={"obs": [torch.zeros(sample.shape[1]) for sample in obs]},
        metadata={"static_coords": frozenset({"grid"}), "obs": {"boundaries": [(slice(0, s.shape[1]),) for s in obs]}},
        layouts={
            "grid": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4),
            "obs": TensorLayout(ensemble=0, grid=1, variables=2),
        },
        variables={"grid": ["value"], "obs": ["value"]},
        statistics={"grid": {}, "obs": {}},
    )
    return PreparedPredictionTarget(
        model_target=batch,
        loss_target=batch,
        loss_target_layout=IndexSpace.MODEL_OUTPUT,
        metric_target=batch,
        model_target_missing=batch,
    )


def test_edm_prepare_accepts_sparse_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, device=None, dtype=None: torch.zeros(shape, device=device, dtype=dtype),
    )
    objective = EDMDiffusionTransportObjective(_module())
    prepared = _prepared_sparse_target()

    objective_data = objective.prepare(prepared)

    assert objective_data.condition["grid"].shape == (2, 1, 1, 1, 1)
    assert objective_data.condition["obs"].shape == (2, 1, 1, 1, 1)
    torch.testing.assert_close(objective_data.conditioned_target["grid"].data, prepared.model_target["grid"].data)
    torch.testing.assert_close(objective_data.conditioned_target["obs"].data[0], prepared.model_target["obs"].data[0])
    torch.testing.assert_close(objective_data.conditioned_target["obs"].data[1], prepared.model_target["obs"].data[1])
    # One weight per sample, shaped to spread over its (ensemble, observations, variables) values.
    assert isinstance(objective_data.weights["obs"], list)
    assert [weight.shape for weight in objective_data.weights["obs"]] == [torch.Size([1, 1, 1])] * 2


def _fixed_training_time(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, device=None, dtype=None: torch.full(shape, value, device=device, dtype=dtype),
    )


def _assert_sparse_close(actual: list[torch.Tensor], expected: list[torch.Tensor]) -> None:
    assert len(actual) == len(expected)
    for actual_sample, expected_sample in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_sample, expected_sample)


def test_flow_matching_prepare_accepts_sparse_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_training_time(monkeypatch, 0.25)
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, device=None, dtype=None: torch.full(shape, 2.0, device=device, dtype=dtype),
    )
    objective = FlowMatchingTransportObjective(_fm_module())
    prepared = _prepared_sparse_target()

    objective_data = objective.prepare(prepared)

    grid = prepared.model_target["grid"].data
    assert objective_data.condition["grid"].shape == (2, 1, 1, 1, 1)
    assert objective_data.condition["obs"].shape == (2, 1, 1, 1, 1)
    torch.testing.assert_close(objective_data.conditioned_target["grid"].data, 0.75 * 2.0 + 0.25 * grid)
    torch.testing.assert_close(objective_data.loss_target["grid"].data, grid - 2.0)

    obs = prepared.model_target["obs"].data
    _assert_sparse_close(objective_data.conditioned_target["obs"].data, [0.75 * 2.0 + 0.25 * sample for sample in obs])
    _assert_sparse_close(objective_data.loss_target["obs"].data, [sample - 2.0 for sample in obs])

    reconstructed = objective.reconstruct_endpoint(objective_data.loss_target, objective_data)
    torch.testing.assert_close(reconstructed["grid"].data, grid)
    _assert_sparse_close(reconstructed["obs"].data, obs)
