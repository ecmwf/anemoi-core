# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from anemoi.models.transport.settings import EdmSettings
from anemoi.models.transport.settings import StochasticInterpolantSettings
from anemoi.models.transport.settings import TransportSourceSettings
from anemoi.models.transport.sources import TransportSourceBuilder
from anemoi.training.train.methods.edm_diffusion import EDMDiffusionTransportObjective
from anemoi.training.train.methods.stochastic_interpolant import StochasticInterpolantTransportObjective
from anemoi.training.train.methods.transport_base import PreparedPredictionTarget
from anemoi.training.utils.index_space import IndexSpace


class _Batch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, dataset_name):
        return self.data[dataset_name]

    def keys(self):
        return self.data.keys()

    def with_data(self, data):
        return _Batch(data)


def _no_grid_shards(_source: object) -> None:
    return None


def _module(source_kind: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                edm=EdmSettings(sigma_data=0.5),
                training_condition={"distribution": "linear", "sigma_max": 1.0, "sigma_min": 0.1},
                transport_source=TransportSourceBuilder(TransportSourceSettings(kind=source_kind)),
            ),
        ),
        model_comm_group=None,
        grid_shard_sizes=None,
        _grid_shard_sizes=_no_grid_shards,
    )


def _si_module(
    source_kind: str,
    *,
    noise_scale: float = 0.0,
    beta_schedule: str = "linear",
    sigma_schedule: str = "quadratic_bridge",
) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                stochastic_interpolant=StochasticInterpolantSettings(
                    beta_schedule=beta_schedule,
                    sigma_schedule=sigma_schedule,
                    noise_scale=noise_scale,
                ),
                training_condition={"distribution": "uniform_time"},
                transport_source=TransportSourceBuilder(TransportSourceSettings(kind=source_kind)),
            ),
        ),
        model_comm_group=None,
        grid_shard_sizes=None,
        _grid_shard_sizes=_no_grid_shards,
    )


def _prepared_sparse_target() -> PreparedPredictionTarget:
    data = {
        "grid": torch.ones(2, 1, 1, 3, 1),
        "obs": [torch.ones(2, 1), torch.full((4, 1), 2.0)],
    }
    batch = _Batch(data)
    return PreparedPredictionTarget(
        model_target=batch,
        loss_target=batch,
        loss_target_layout=IndexSpace.MODEL_OUTPUT,
        metric_target=batch,
        aux={},
    )


def test_edm_prepare_accepts_sparse_targets_with_zero_source() -> None:
    objective = EDMDiffusionTransportObjective(_module("zero"))
    prepared = _prepared_sparse_target()

    objective_data = objective.prepare(prepared)

    assert objective_data.condition["grid"].shape == (2, 1, 1, 1, 1)
    assert objective_data.condition["obs"].shape == (2, 1, 1, 1, 1)
    torch.testing.assert_close(objective_data.conditioned_target.data["grid"], prepared.model_target.data["grid"])
    torch.testing.assert_close(objective_data.conditioned_target.data["obs"][0], prepared.model_target.data["obs"][0])
    torch.testing.assert_close(objective_data.conditioned_target.data["obs"][1], prepared.model_target.data["obs"][1])
    assert isinstance(objective_data.weights["obs"], list)
    assert [weight.shape for weight in objective_data.weights["obs"]] == [torch.Size([]), torch.Size([])]


def test_edm_prepare_rejects_reference_state_for_sparse_targets() -> None:
    objective = EDMDiffusionTransportObjective(_module("reference_state"))

    with pytest.raises(NotImplementedError, match="reference_state.*sparse"):
        objective.prepare(_prepared_sparse_target())


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


def test_stochastic_interpolant_prepare_accepts_sparse_targets_with_zero_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_training_time(monkeypatch, 0.25)
    objective = StochasticInterpolantTransportObjective(_si_module("zero", noise_scale=0.0))
    prepared = _prepared_sparse_target()

    objective_data = objective.prepare(prepared)

    assert objective_data.condition["grid"].shape == (2, 1, 1, 1, 1)
    assert objective_data.condition["obs"].shape == (2, 1, 1, 1, 1)
    torch.testing.assert_close(
        objective_data.conditioned_target.data["grid"],
        prepared.model_target.data["grid"] * 0.25,
    )
    torch.testing.assert_close(objective_data.loss_target.data["grid"], prepared.model_target.data["grid"])
    _assert_sparse_close(
        objective_data.conditioned_target.data["obs"],
        [sample * 0.25 for sample in prepared.model_target.data["obs"]],
    )
    _assert_sparse_close(objective_data.loss_target.data["obs"], prepared.model_target.data["obs"])

    reconstructed = objective.reconstruct_endpoint(objective_data.loss_target, objective_data)
    torch.testing.assert_close(reconstructed.data["grid"], prepared.model_target.data["grid"])
    _assert_sparse_close(reconstructed.data["obs"], prepared.model_target.data["obs"])


def test_stochastic_interpolant_prepare_accepts_sparse_targets_with_gaussian_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_training_time(monkeypatch, 0.25)
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, device=None, dtype=None: torch.full(shape, 2.0, device=device, dtype=dtype),
    )
    objective = StochasticInterpolantTransportObjective(_si_module("gaussian", noise_scale=0.0))
    prepared = _prepared_sparse_target()

    objective_data = objective.prepare(prepared)

    expected_grid_source = torch.full_like(prepared.model_target.data["grid"], 2.0)
    expected_grid_interpolant = 0.75 * expected_grid_source + 0.25 * prepared.model_target.data["grid"]
    expected_grid_drift = -expected_grid_source + prepared.model_target.data["grid"]
    torch.testing.assert_close(objective_data.aux["source"]["grid"], expected_grid_source)
    torch.testing.assert_close(objective_data.conditioned_target.data["grid"], expected_grid_interpolant)
    torch.testing.assert_close(objective_data.loss_target.data["grid"], expected_grid_drift)

    expected_obs_source = [torch.full_like(sample, 2.0) for sample in prepared.model_target.data["obs"]]
    expected_obs_interpolant = [
        0.75 * source_sample + 0.25 * clean_sample
        for source_sample, clean_sample in zip(expected_obs_source, prepared.model_target.data["obs"], strict=True)
    ]
    expected_obs_drift = [
        -source_sample + clean_sample
        for source_sample, clean_sample in zip(expected_obs_source, prepared.model_target.data["obs"], strict=True)
    ]
    _assert_sparse_close(objective_data.aux["source"]["obs"], expected_obs_source)
    _assert_sparse_close(objective_data.conditioned_target.data["obs"], expected_obs_interpolant)
    _assert_sparse_close(objective_data.loss_target.data["obs"], expected_obs_drift)

    reconstructed = objective.reconstruct_endpoint(objective_data.loss_target, objective_data)
    torch.testing.assert_close(reconstructed.data["grid"], prepared.model_target.data["grid"])
    _assert_sparse_close(reconstructed.data["obs"], prepared.model_target.data["obs"])


def test_stochastic_interpolant_prepare_rejects_sparse_reference_state_source() -> None:
    objective = StochasticInterpolantTransportObjective(_si_module("reference_state"))

    with pytest.raises(NotImplementedError, match="reference_state.*sparse"):
        objective.prepare(_prepared_sparse_target())


def test_stochastic_interpolant_prepare_accepts_sparse_bridge_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_training_time(monkeypatch, 0.25)
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, device=None, dtype=None: torch.ones(shape, device=device, dtype=dtype),
    )
    monkeypatch.setattr(torch, "randn_like", lambda sample: torch.ones_like(sample))
    objective = StochasticInterpolantTransportObjective(
        _si_module("zero", noise_scale=1.0, sigma_schedule="quadratic_bridge"),
    )
    prepared = _prepared_sparse_target()

    objective_data = objective.prepare(prepared)

    bridge_noise = 0.25 * 0.75
    drift_noise = 0.5
    torch.testing.assert_close(
        objective_data.conditioned_target.data["grid"],
        prepared.model_target.data["grid"] * 0.25 + bridge_noise,
    )
    torch.testing.assert_close(
        objective_data.loss_target.data["grid"],
        prepared.model_target.data["grid"] + drift_noise,
    )
    _assert_sparse_close(
        objective_data.conditioned_target.data["obs"],
        [sample * 0.25 + bridge_noise for sample in prepared.model_target.data["obs"]],
    )
    _assert_sparse_close(
        objective_data.loss_target.data["obs"],
        [sample + drift_noise for sample in prepared.model_target.data["obs"]],
    )

    reconstructed = objective.reconstruct_endpoint(objective_data.loss_target, objective_data)
    torch.testing.assert_close(reconstructed.data["grid"], prepared.model_target.data["grid"])
    _assert_sparse_close(reconstructed.data["obs"], prepared.model_target.data["obs"])
