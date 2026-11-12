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
from anemoi.models.transport.schedules import TIME_SCHEDULES
from anemoi.models.transport.settings import EdmSettings
from tests.batch_builders import build_batch


class _Model:
    edm = EdmSettings(sigma_data=1.0)

    def __init__(self):
        self.scaled_noised = None

    def _assert_condition_shapes(self, sigma):
        return 2, 1

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

    out = EDMDiffusionModelObjective().forward(model, noised.with_data({}), noised, sigma)

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
    schedule = objectives._build_inference_schedule(model, TIME_SCHEDULES, "test schedule", None, torch.device("cpu"))
    assert schedule.dtype == torch.float32
