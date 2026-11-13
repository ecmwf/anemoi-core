# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import pytest
import torch

from anemoi.models.data import TensorLayout
from anemoi.models.transport import objectives
from anemoi.models.transport.objectives import EDMDiffusionModelObjective
from anemoi.models.transport.paths import edm_loss_weight
from anemoi.models.transport.schedules import SIGMA_SCHEDULES
from anemoi.models.transport.schedules import TIME_SCHEDULES
from tests.batch_builders import build_batch


class _Model:
    def __init__(self):
        self.scaled_noised = None

    def _forward_transport_network(self, _x, scaled_noised, _c_noise, **_kwargs):
        self.scaled_noised = scaled_noised
        return scaled_noised.with_data(
            {"obs": [torch.zeros_like(sample) for sample in scaled_noised["obs"].data]},
        )


def test_edm_model_preconditioning_handles_sparse_targets() -> None:
    model = _Model()
    noised = build_batch(
        data={"obs": [torch.ones(2, 1), torch.full((3, 1), 2.0)]},
        coordinates={"obs": [torch.zeros(2, 2), torch.zeros(3, 2)]},
        timedeltas={"obs": [torch.zeros(2), torch.zeros(3)]},
        boundaries={"obs": [(slice(0, 2),), (slice(0, 3),)]},
        layouts={"obs": TensorLayout(grid=0, variables=1)},
        variables={"obs": ["x"]},
        statistics={"obs": {}},
    )
    sigma = {"obs": torch.ones(2, 1, 1, 1, 1)}

    out = EDMDiffusionModelObjective(sigma_data=1.0).forward(
        model._forward_transport_network,
        noised.with_data({}),
        noised,
        sigma,
    )

    torch.testing.assert_close(model.scaled_noised["obs"].data[0], torch.full((2, 1), 1.0 / math.sqrt(2.0)))
    torch.testing.assert_close(model.scaled_noised["obs"].data[1], torch.full((3, 1), math.sqrt(2.0)))
    torch.testing.assert_close(out["obs"].data[0], torch.full((2, 1), 0.5))
    torch.testing.assert_close(out["obs"].data[1], torch.full((3, 1), 1.0))


@pytest.mark.parametrize("device", ["cpu", "cuda", "meta"])
def test_sampling_runs_in_float64_where_the_device_has_it(device: str) -> None:
    """The schedule and solver state are float64 on every device that has float64."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert objectives._sampling_dtype(torch.device(device)) == torch.float64


def test_sampling_runs_in_float32_with_a_warning_on_mps() -> None:
    """MPS has no float64, so the sampler falls back to float32."""
    with pytest.warns(UserWarning, match="float32 on MPS"):
        assert objectives._sampling_dtype(torch.device("mps")) == torch.float32


def test_inference_schedule_takes_the_sampling_dtype(monkeypatch) -> None:
    """The inference schedule, which sets the sampler's dtype, is built in the device's sampling dtype."""
    monkeypatch.setattr(objectives, "_sampling_dtype", lambda device: torch.float32)
    model = SimpleNamespace(inference_defaults={"sampling_schedule": {"schedule_type": "unit_time", "num_steps": 4}})
    schedule = objectives._build_inference_schedule(
        model.inference_defaults,
        TIME_SCHEDULES,
        "test schedule",
        None,
        torch.device("cpu"),
    )
    assert schedule.dtype == torch.float32


def test_inference_override_keeps_the_other_defaults() -> None:
    model = SimpleNamespace(
        inference_defaults={"sampling_schedule": {"schedule_type": "karras", "rho": 7.0, "num_steps": 50}},
    )

    config = objectives._get_inference_config(model.inference_defaults, "sampling_schedule", {"num_steps": 20})

    assert config == {"schedule_type": "karras", "rho": 7.0, "num_steps": 20}


def test_inference_override_of_another_schedule_type_keeps_the_settings_it_takes() -> None:
    """Switching from Karras to linear keeps the noise range and step count; the Karras ``rho`` is left out."""
    model = SimpleNamespace(
        inference_defaults={
            "sampling_schedule": {
                "schedule_type": "karras",
                "sigma_max": 1.0,
                "sigma_min": 0.25,
                "rho": 7.0,
                "num_steps": 4,
            },
        },
    )

    schedule = objectives._build_inference_schedule(
        model.inference_defaults,
        SIGMA_SCHEDULES,
        "test schedule",
        {"schedule_type": "linear"},
        torch.device("cpu"),
    )

    torch.testing.assert_close(schedule, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float64))


def test_edm_preconditioning_gives_unit_variance_inputs_and_targets() -> None:
    """Karras et al. (2022), table 1: with data of standard deviation sigma_data, the scaled network input
    has unit variance, the network target has unit variance, and the loss weight is 1 / c_out^2.
    """
    sigma_data = 0.5
    sigma_values = torch.tensor([0.02, 0.5, 80.0], dtype=torch.float64)
    sigma = {"data": sigma_values.view(3, 1, 1, 1, 1)}
    c_skip, c_out, c_in, c_noise = objectives.edm_preconditioning(sigma, sigma_data)
    c_skip, c_out, c_in, c_noise = (factor["data"].flatten() for factor in (c_skip, c_out, c_in, c_noise))

    # Noised input y = x + n has variance sigma_data^2 + sigma^2.
    torch.testing.assert_close(c_in**2 * (sigma_data**2 + sigma_values**2), torch.ones(3, dtype=torch.float64))
    # Target F = (x - c_skip y) / c_out has unit variance.
    target_variance = ((1 - c_skip) ** 2 * sigma_data**2 + c_skip**2 * sigma_values**2) / c_out**2
    torch.testing.assert_close(target_variance, torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(edm_loss_weight(sigma_values, sigma_data), 1 / c_out**2)
    torch.testing.assert_close(c_noise, sigma_values.log() / 4)
    # At sigma = sigma_data the output is half skip, half network.
    assert c_skip[1].item() == pytest.approx(0.5)
