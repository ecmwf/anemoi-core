# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for diffusion samplers: Heun, DPM++2M, piecewise scheduler, Karras terminal zero."""

from __future__ import annotations

from typing import Optional

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.samplers.diffusion_samplers import DPMpp2MSampler
from anemoi.models.samplers.diffusion_samplers import EDMHeunSampler
from anemoi.models.samplers.diffusion_samplers import ExperimentalSamplerScheduler
from anemoi.models.samplers.diffusion_samplers import KarrasScheduler


# ============================================================
# Helpers
# ============================================================

DATASET_NAME = "test_dataset"


class MockDenoisingFunction:
    """Deterministic mock: reduces noise proportionally to sigma."""

    def __init__(self, noise_reduction_factor: float = 0.9):
        self.noise_reduction_factor = noise_reduction_factor
        self.call_count = 0

    def __call__(
        self,
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[dict[str, Optional[list]]] = None,
    ) -> dict[str, torch.Tensor]:
        self.call_count += 1
        denoised = {}
        for dataset_name, y_ in y.items():
            sigma_val = sigma[dataset_name]
            sigma_normalized = sigma_val / (sigma_val.max() + 1e-8)
            denoised[dataset_name] = (1 - sigma_normalized * self.noise_reduction_factor) * y_
        return denoised


def _make_data(shape=(2, 3, 1, 10, 5), num_steps=5):
    """Return (x, y, sigmas) for sampler tests."""
    x = {DATASET_NAME: torch.randn(*shape)}
    y = {DATASET_NAME: torch.randn(*shape)}
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
    return x, y, sigmas


# ============================================================
# Tests
# ============================================================


def test_heun_sampler_basic():
    """EDMHeunSampler: dict in/out, correct shape, finite, denoising fn called."""
    x, y, sigmas = _make_data()
    mock_fn = MockDenoisingFunction()

    result = EDMHeunSampler().sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_fn)

    assert set(result.keys()) == {DATASET_NAME}
    assert result[DATASET_NAME].shape == y[DATASET_NAME].shape
    assert torch.isfinite(result[DATASET_NAME]).all()
    assert mock_fn.call_count > 0
    # Heun does first-order + correction per step (except possibly last)
    assert mock_fn.call_count >= 5


def test_dpmpp2m_sampler_basic():
    """DPMpp2MSampler: dict in/out, correct shape, finite, one call per step."""
    x, y, sigmas = _make_data()
    mock_fn = MockDenoisingFunction()

    result = DPMpp2MSampler().sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_fn)

    assert set(result.keys()) == {DATASET_NAME}
    assert result[DATASET_NAME].shape == y[DATASET_NAME].shape
    assert torch.isfinite(result[DATASET_NAME]).all()
    assert mock_fn.call_count == 5  # DPM++2M: exactly one denoising call per step


def test_piecewise_schedule():
    """ExperimentalSamplerScheduler: monotonic, correct endpoints, transition point, terminal zero."""
    sched = ExperimentalSamplerScheduler(
        sigma_max=100000.0, sigma_min=0.02, num_steps=30,
        sigma_transition=10.0, num_steps_high=10, num_steps_low=20,
    )
    sigmas = sched.get_schedule()

    # Length = num_steps + terminal zero
    assert len(sigmas) == 31
    # Monotonically decreasing
    assert all(sigmas[i] >= sigmas[i + 1] for i in range(len(sigmas) - 1))
    # Endpoints
    assert torch.isclose(sigmas[0], torch.tensor(100000.0, dtype=torch.float64))
    assert torch.isclose(sigmas[-2], torch.tensor(0.02, dtype=torch.float64), atol=1e-6)
    assert sigmas[-1] == 0.0
    # Transition point at boundary between high and low segments
    assert torch.isclose(sigmas[10], torch.tensor(10.0, dtype=torch.float64), atol=1e-4)

    # Also works end-to-end with a sampler
    x, y, _ = _make_data()
    mock_fn = MockDenoisingFunction()
    result = EDMHeunSampler().sample(x, y, sigmas.float(), mock_fn)
    assert torch.isfinite(result[DATASET_NAME]).all()


def test_karras_terminal_zero():
    """KarrasScheduler appends terminal zero so samplers terminate properly."""
    sched = KarrasScheduler(sigma_max=80.0, sigma_min=0.002, num_steps=10)
    sigmas = sched.get_schedule()

    assert len(sigmas) == 11  # num_steps + terminal zero
    assert sigmas[-1] == 0.0
    assert sigmas[-2] > 0.0
    # All non-terminal values are positive and monotonically decreasing
    assert all(sigmas[i] > sigmas[i + 1] for i in range(len(sigmas) - 2))
