# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Callable

import pytest
import torch

from anemoi.models.samplers.diffusion_samplers import DEFAULT_FINAL_SIGMA_EPS
from anemoi.models.samplers.diffusion_samplers import CosineScheduler
from anemoi.models.samplers.diffusion_samplers import DPMpp2MSampler
from anemoi.models.samplers.diffusion_samplers import EDMHeunSampler
from anemoi.models.samplers.diffusion_samplers import ExponentialScheduler
from anemoi.models.samplers.diffusion_samplers import KarrasScheduler
from anemoi.models.samplers.diffusion_samplers import LinearScheduler
from anemoi.models.samplers.diffusion_samplers import NoiseScheduler

DATASET_NAME = "test_dataset"


class DummyScheduler(NoiseScheduler):
    def __init__(
        self,
        schedule: torch.Tensor,
        *,
        num_steps: int,
        sigma_max: float = 1.0,
        sigma_min: float = 0.1,
    ) -> None:
        super().__init__(
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            num_steps=num_steps,
        )
        self.schedule = schedule

    def _build_schedule(
        self,
        device: torch.device = None,
        dtype_compute: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        return self.schedule.to(device=device, dtype=dtype_compute)


class RecordingZeroDenoiser:
    def __init__(self, validator: Callable | None = None) -> None:
        self.validator = validator
        self.call_count = 0

    def __call__(
        self,
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
        model_comm_group=None,
        grid_shard_shapes=None,
    ) -> dict[str, torch.Tensor]:
        del model_comm_group, grid_shard_shapes
        self.call_count += 1
        if self.validator is not None:
            self.validator(x, y, sigma)
        return {dataset_name: torch.zeros_like(y_data) for dataset_name, y_data in y.items()}


def make_inputs(dtype: torch.dtype = torch.float32) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    x = {DATASET_NAME: torch.randn(2, 3, 1, 5, 4, dtype=dtype)}
    y = {DATASET_NAME: torch.randn(2, 3, 1, 5, 4, dtype=dtype)}
    return x, y


@pytest.mark.parametrize(
    ("scheduler_cls", "scheduler_kwargs"),
    [
        (KarrasScheduler, {"rho": 7.0}),
        (LinearScheduler, {}),
        (CosineScheduler, {"s": 0.008}),
        (ExponentialScheduler, {}),
    ],
)
def test_builtin_noise_schedulers_return_descending_schedule_with_final_zero(
    scheduler_cls: type[NoiseScheduler],
    scheduler_kwargs: dict[str, float],
) -> None:
    scheduler = scheduler_cls(
        sigma_max=1.0,
        sigma_min=0.02,
        num_steps=6,
        **scheduler_kwargs,
    )

    sigmas = scheduler.get_schedule(dtype_compute=torch.float64)
    prefinal = sigmas[:-1]

    assert sigmas.shape == (7,)
    assert sigmas.dtype == torch.float64
    assert sigmas[-1].item() == 0.0
    assert torch.isfinite(sigmas).all()
    assert torch.all(prefinal > 0)
    assert torch.all(prefinal[:-1] > prefinal[1:])
    assert prefinal[0].item() == pytest.approx(1.0)
    assert prefinal[-1].item() == pytest.approx(0.02)


def test_karras_scheduler_single_step_returns_sigma_max_and_final_zero() -> None:
    sigmas = KarrasScheduler(
        sigma_max=1.0,
        sigma_min=0.02,
        num_steps=1,
        rho=7.0,
    ).get_schedule(dtype_compute=torch.float64)

    assert torch.equal(sigmas, torch.tensor([1.0, 0.0], dtype=torch.float64))


@pytest.mark.parametrize(
    ("scheduler_cls", "scheduler_kwargs"),
    [
        (KarrasScheduler, {"rho": 7.0}),
        (LinearScheduler, {}),
        (CosineScheduler, {"s": 0.008}),
        (ExponentialScheduler, {}),
    ],
)
@pytest.mark.parametrize("sigma_min", [0.0, -0.1])
def test_builtin_noise_schedulers_require_strictly_positive_sigma_min(
    scheduler_cls: type[NoiseScheduler],
    scheduler_kwargs: dict[str, float],
    sigma_min: float,
) -> None:
    with pytest.raises(ValueError, match="sigma_min must be strictly positive"):
        scheduler_cls(
            sigma_max=1.0,
            sigma_min=sigma_min,
            num_steps=6,
            **scheduler_kwargs,
        )


def test_noise_scheduler_validates_common_constructor_contract() -> None:
    with pytest.raises(ValueError, match="sigma_max must be strictly positive"):
        DummyScheduler(torch.linspace(1.0, 0.1, 4), num_steps=4, sigma_max=0.0)

    with pytest.raises(ValueError, match="sigma_max must be greater than or equal to sigma_min"):
        DummyScheduler(torch.linspace(1.0, 0.1, 4), num_steps=4, sigma_max=0.05, sigma_min=0.1)

    with pytest.raises(ValueError, match="num_steps must be at least 1"):
        DummyScheduler(torch.empty(0), num_steps=0)


def test_noise_scheduler_appends_exact_final_zero_when_missing() -> None:
    base_schedule = torch.linspace(1.0, 0.1, 4, dtype=torch.float64)
    scheduler = DummyScheduler(base_schedule, num_steps=4)

    sigmas = scheduler.get_schedule(dtype_compute=torch.float64)

    assert torch.equal(sigmas[:-1], base_schedule)
    assert sigmas[-1].item() == 0.0


def test_noise_scheduler_preserves_explicit_near_zero_final_value() -> None:
    final_sigma = DEFAULT_FINAL_SIGMA_EPS / 10
    scheduler = DummyScheduler(
        torch.tensor([1.0, 0.7, 0.4, 0.1, final_sigma], dtype=torch.float64),
        num_steps=4,
    )

    sigmas = scheduler.get_schedule(dtype_compute=torch.float64)

    assert sigmas.shape == (5,)
    assert sigmas[-1].item() == pytest.approx(final_sigma)


def test_noise_scheduler_rejects_explicit_final_value_outside_tolerance() -> None:
    scheduler = DummyScheduler(
        torch.tensor([1.0, 0.7, 0.4, 0.1, DEFAULT_FINAL_SIGMA_EPS * 10], dtype=torch.float64),
        num_steps=4,
    )

    with pytest.raises(ValueError, match="explicit final value"):
        scheduler.get_schedule(dtype_compute=torch.float64)


def test_noise_scheduler_rejects_negative_explicit_final_value() -> None:
    scheduler = DummyScheduler(
        torch.tensor([1.0, 0.7, 0.4, 0.1, -DEFAULT_FINAL_SIGMA_EPS / 10], dtype=torch.float64),
        num_steps=4,
    )

    with pytest.raises(ValueError, match="explicit final value"):
        scheduler.get_schedule(dtype_compute=torch.float64)


@pytest.mark.parametrize("sampler_cls", [EDMHeunSampler, DPMpp2MSampler])
def test_samplers_expand_sigma_to_model_dtype_and_return_model_dtype(
    sampler_cls: type[EDMHeunSampler] | type[DPMpp2MSampler],
) -> None:
    x, y = make_inputs(dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.0], dtype=torch.float64)

    def _validate_sigma(
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
    ) -> None:
        assert set(sigma.keys()) == set(y.keys())
        sigma_expanded = sigma[DATASET_NAME]
        assert sigma_expanded.dtype == x[DATASET_NAME].dtype == y[DATASET_NAME].dtype
        assert sigma_expanded.shape == (
            y[DATASET_NAME].shape[0],
            1,
            y[DATASET_NAME].shape[2],
            1,
            1,
        )

    denoiser = RecordingZeroDenoiser(validator=_validate_sigma)
    sampler = sampler_cls(dtype=torch.float64)

    result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=denoiser)

    assert denoiser.call_count == 1
    assert result[DATASET_NAME].shape == y[DATASET_NAME].shape
    assert result[DATASET_NAME].dtype == x[DATASET_NAME].dtype
    assert torch.allclose(result[DATASET_NAME], torch.zeros_like(result[DATASET_NAME]))


def test_heun_uses_corrector_before_final_step() -> None:
    x, y = make_inputs(dtype=torch.float64)
    sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)

    denoiser = RecordingZeroDenoiser()
    sampler = EDMHeunSampler(dtype=torch.float64, eps_prec=0.0)
    sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=denoiser)

    assert denoiser.call_count == 3


def test_heun_uses_final_sigma_tolerance_for_final_step() -> None:
    x = {DATASET_NAME: torch.ones(1, 1, 1, 1, 1, dtype=torch.float64)}
    y = {DATASET_NAME: 2 * torch.ones(1, 1, 1, 1, 1, dtype=torch.float64)}
    final_sigma = DEFAULT_FINAL_SIGMA_EPS / 10
    sigmas = torch.tensor([1.0, final_sigma], dtype=torch.float64)

    denoiser = RecordingZeroDenoiser()
    sampler = EDMHeunSampler(dtype=torch.float64, eps_prec=0.0)
    result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=denoiser)

    assert denoiser.call_count == 1
    assert torch.allclose(result[DATASET_NAME], y[DATASET_NAME] * final_sigma, atol=1e-14, rtol=0.0)


def test_dpmpp_uses_final_sigma_tolerance_for_final_step() -> None:
    x = {DATASET_NAME: torch.ones(1, 1, 1, 1, 1, dtype=torch.float64)}
    y = {DATASET_NAME: 2 * torch.ones(1, 1, 1, 1, 1, dtype=torch.float64)}
    sigmas = torch.tensor([1.0, DEFAULT_FINAL_SIGMA_EPS / 10], dtype=torch.float64)

    denoiser = RecordingZeroDenoiser()
    sampler = DPMpp2MSampler(dtype=torch.float64)
    result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=denoiser)

    assert denoiser.call_count == 1
    assert torch.allclose(result[DATASET_NAME], torch.zeros_like(result[DATASET_NAME]))
