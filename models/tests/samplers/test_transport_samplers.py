# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math
from collections.abc import Callable

import pytest
import torch

from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.samplers.transport_samplers import DPMpp2MSampler
from anemoi.models.samplers.transport_samplers import EDMDiffusionSampler
from anemoi.models.samplers.transport_samplers import EDMHeunSampler
from anemoi.models.samplers.transport_samplers import VectorFieldEulerSampler
from anemoi.models.samplers.transport_samplers import VectorFieldHeunSampler
from anemoi.models.samplers.transport_samplers import _expand_scalar_condition
from anemoi.models.transport.schedules import CosineSigmaSchedule
from anemoi.models.transport.schedules import ExponentialSigmaSchedule
from anemoi.models.transport.schedules import KarrasSigmaSchedule
from anemoi.models.transport.schedules import LinearSigmaSchedule
from anemoi.models.transport.schedules import SigmaSchedule
from tests.batch_builders import build_batch

DATASET_NAME = "test_dataset"


class DummySigmaSchedule(SigmaSchedule):
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
    ) -> torch.Tensor:
        return self.schedule.to(device=device, dtype=dtype_compute)


class RecordingZeroDenoiser:
    def __init__(self, validator: Callable | None = None) -> None:
        self.validator = validator
        self.call_count = 0

    def __call__(
        self,
        x: Batch,
        y: Batch,
        sigma: dict[str, torch.Tensor],
        model_comm_group=None,
    ) -> Batch:
        del model_comm_group
        self.call_count += 1
        if self.validator is not None:
            self.validator(x, y, sigma)
        return y.with_data({name: torch.zeros_like(src.data) for name, src in y.items()})


def _batch(data: dict[str, torch.Tensor]) -> Batch:
    return build_batch(
        data=data,
        coordinates={DATASET_NAME: torch.zeros(data[DATASET_NAME].shape[-2], 2)},
        metadata={"static_coords": frozenset({DATASET_NAME})},
        layouts={DATASET_NAME: TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)},
        variables={DATASET_NAME: [f"var_{idx}" for idx in range(data[DATASET_NAME].shape[-1])]},
        statistics={DATASET_NAME: {}},
    )


def make_inputs(dtype: torch.dtype = torch.float32) -> tuple[Batch, Batch]:
    x = _batch({DATASET_NAME: torch.randn(2, 3, 1, 5, 4, dtype=dtype)})
    y = _batch({DATASET_NAME: torch.randn(2, 3, 1, 5, 4, dtype=dtype)})
    return x, y


@pytest.mark.parametrize(
    ("scheduler_cls", "scheduler_kwargs"),
    [
        (KarrasSigmaSchedule, {"rho": 7.0}),
        (LinearSigmaSchedule, {}),
        (CosineSigmaSchedule, {}),
        (ExponentialSigmaSchedule, {}),
    ],
)
def test_builtin_sigma_schedules_return_descending_schedule_with_final_zero(
    scheduler_cls: type[SigmaSchedule],
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


def test_karras_sigma_schedule_single_step_returns_sigma_max_and_final_zero() -> None:
    sigmas = KarrasSigmaSchedule(
        sigma_max=1.0,
        sigma_min=0.02,
        num_steps=1,
        rho=7.0,
    ).get_schedule(dtype_compute=torch.float64)

    assert torch.equal(sigmas, torch.tensor([1.0, 0.0], dtype=torch.float64))


@pytest.mark.parametrize(
    ("scheduler_cls", "scheduler_kwargs"),
    [
        (KarrasSigmaSchedule, {"rho": 7.0}),
        (LinearSigmaSchedule, {}),
        (CosineSigmaSchedule, {}),
        (ExponentialSigmaSchedule, {}),
    ],
)
@pytest.mark.parametrize("sigma_min", [0.0, -0.1])
def test_builtin_sigma_schedules_require_strictly_positive_sigma_min(
    scheduler_cls: type[SigmaSchedule],
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


def test_sigma_schedule_validates_common_constructor_contract() -> None:
    with pytest.raises(ValueError, match="sigma_max must be strictly positive"):
        DummySigmaSchedule(torch.linspace(1.0, 0.1, 4), num_steps=4, sigma_max=0.0)

    with pytest.raises(ValueError, match="sigma_max must be greater than or equal to sigma_min"):
        DummySigmaSchedule(torch.linspace(1.0, 0.1, 4), num_steps=4, sigma_max=0.05, sigma_min=0.1)

    with pytest.raises(ValueError, match="num_steps must be at least 1"):
        DummySigmaSchedule(torch.empty(0), num_steps=0)


def test_sigma_schedule_appends_exact_final_zero_when_missing() -> None:
    base_schedule = torch.linspace(1.0, 0.1, 4, dtype=torch.float64)
    scheduler = DummySigmaSchedule(base_schedule, num_steps=4)

    sigmas = scheduler.get_schedule(dtype_compute=torch.float64)

    assert torch.equal(sigmas[:-1], base_schedule)
    assert sigmas[-1].item() == 0.0


@pytest.mark.parametrize("sampler_cls", [EDMHeunSampler, DPMpp2MSampler])
def test_samplers_expand_sigma_to_model_dtype_and_return_model_dtype(
    sampler_cls: type[EDMHeunSampler] | type[DPMpp2MSampler],
) -> None:
    x, y = make_inputs(dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.0], dtype=torch.float64)

    def _validate_sigma(
        x: Batch,
        y: Batch,
        sigma: dict[str, torch.Tensor],
    ) -> None:
        assert set(sigma.keys()) == set(y.keys())
        sigma_expanded = sigma[DATASET_NAME]
        assert sigma_expanded.dtype == x[DATASET_NAME].data.dtype == y[DATASET_NAME].data.dtype
        assert sigma_expanded.shape == (
            y[DATASET_NAME].data.shape[0],
            1,
            y[DATASET_NAME].data.shape[2],
            1,
            1,
        )

    denoiser = RecordingZeroDenoiser(validator=_validate_sigma)
    sampler = sampler_cls(dtype=torch.float64)

    result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=denoiser)

    assert denoiser.call_count == 1
    assert result[DATASET_NAME].data.shape == y[DATASET_NAME].data.shape
    assert result[DATASET_NAME].data.dtype == x[DATASET_NAME].data.dtype
    assert torch.allclose(result[DATASET_NAME].data, torch.zeros_like(result[DATASET_NAME].data))


def test_heun_uses_corrector_before_final_step() -> None:
    x, y = make_inputs(dtype=torch.float64)
    sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)

    denoiser = RecordingZeroDenoiser()
    sampler = EDMHeunSampler(dtype=torch.float64)
    sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=denoiser)

    assert denoiser.call_count == 3


@pytest.mark.parametrize("sampler_cls", [VectorFieldEulerSampler, VectorFieldHeunSampler])
def test_vector_field_samplers_integrate_constant_velocity(
    sampler_cls: type[VectorFieldEulerSampler] | type[VectorFieldHeunSampler],
) -> None:
    x, y = make_inputs(dtype=torch.float32)
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)

    def velocity_fn(
        x: Batch,
        y: Batch,
        time: dict[str, torch.Tensor],
        model_comm_group=None,
    ) -> Batch:
        del model_comm_group
        time_expanded = time[DATASET_NAME]
        assert time_expanded.dtype == x[DATASET_NAME].data.dtype == y[DATASET_NAME].data.dtype
        assert time_expanded.shape == (
            y[DATASET_NAME].data.shape[0],
            1,
            y[DATASET_NAME].data.shape[2],
            1,
            1,
        )
        return y.with_data({name: torch.ones_like(src.data) for name, src in y.items()})

    sampler = sampler_cls(dtype=torch.float64)
    result = sampler.sample(x=x, y=y, times=times, vector_field_fn=velocity_fn)

    assert result[DATASET_NAME].data.dtype == x[DATASET_NAME].data.dtype
    assert torch.allclose(result[DATASET_NAME].data, y[DATASET_NAME].data + 1.0)


def test_vector_field_heun_matches_linear_ode_euler_final_step() -> None:
    x = _batch({DATASET_NAME: torch.zeros(1, 1, 1, 1, 1, dtype=torch.float64)})
    y = _batch({DATASET_NAME: torch.full((1, 1, 1, 1, 1), 2.0, dtype=torch.float64)})
    times = torch.tensor([0.0, 0.25, 0.5], dtype=torch.float64)

    def vector_field_fn(
        x: Batch,
        y: Batch,
        time: dict[str, torch.Tensor],
        model_comm_group=None,
    ) -> Batch:
        del x, model_comm_group
        return y.with_data({DATASET_NAME: 2.0 * y[DATASET_NAME].data + time[DATASET_NAME]})

    sampler = VectorFieldHeunSampler(dtype=torch.float64, euler_final_step=True)
    result = sampler.sample(x=x, y=y, times=times, vector_field_fn=vector_field_fn)

    first_dt = times[1] - times[0]
    first_f1 = 2.0 * y[DATASET_NAME].data + times[0]
    first_predictor = y[DATASET_NAME].data + first_dt * first_f1
    first_f2 = 2.0 * first_predictor + times[1]
    first_corrected = y[DATASET_NAME].data + first_dt * (first_f1 + first_f2) / 2.0

    final_dt = times[2] - times[1]
    final_f1 = 2.0 * first_corrected + times[1]
    expected = first_corrected + final_dt * final_f1
    torch.testing.assert_close(result[DATASET_NAME].data, expected)


def test_vector_field_heun_can_use_predictor_corrector_on_final_step() -> None:
    x = _batch({DATASET_NAME: torch.zeros(1, 1, 1, 1, 1, dtype=torch.float64)})
    y = _batch({DATASET_NAME: torch.full((1, 1, 1, 1, 1), 2.0, dtype=torch.float64)})
    times = torch.tensor([0.0, 0.25], dtype=torch.float64)

    def vector_field_fn(
        x: Batch,
        y: Batch,
        time: dict[str, torch.Tensor],
        model_comm_group=None,
    ) -> Batch:
        del x, model_comm_group
        return y.with_data({DATASET_NAME: 2.0 * y[DATASET_NAME].data + time[DATASET_NAME]})

    sampler = VectorFieldHeunSampler(dtype=torch.float64, euler_final_step=False)
    result = sampler.sample(x=x, y=y, times=times, vector_field_fn=vector_field_fn)

    dt = times[1] - times[0]
    f1 = 2.0 * y[DATASET_NAME].data + times[0]
    y_predictor = y[DATASET_NAME].data + dt * f1
    f2 = 2.0 * y_predictor + times[1]
    expected = y[DATASET_NAME].data + dt * (f1 + f2) / 2.0
    torch.testing.assert_close(result[DATASET_NAME].data, expected)


def test_scalar_condition_keeps_at_least_float32() -> None:
    """A time or noise level stays float32 for bfloat16 data, and float64 for float64 data."""
    _, y_bfloat16 = make_inputs(dtype=torch.bfloat16)
    _, y_float64 = make_inputs(dtype=torch.float64)
    value = torch.tensor(0.123456789, dtype=torch.float64)

    condition = _expand_scalar_condition(value, y_bfloat16)[DATASET_NAME]

    assert condition.dtype == torch.float32
    assert condition.flatten()[0].item() == pytest.approx(value.item(), rel=1e-7)
    assert _expand_scalar_condition(value, y_float64)[DATASET_NAME].dtype == torch.float64


@pytest.mark.parametrize("sampler_cls", [EDMHeunSampler, DPMpp2MSampler])
def test_edm_samplers_follow_the_exact_flow_of_gaussian_data(sampler_cls: type[EDMDiffusionSampler]) -> None:
    """For data drawn from N(0, sd^2) the ideal denoiser is known, D(y, sigma) = y sd^2 / (sd^2 + sigma^2).

    The probability-flow ODE then scales each value by sqrt(sd^2 + sigma^2), so a sample that starts at
    sigma_max ends at y_start * sd / sqrt(sd^2 + sigma_max^2). Both samplers are second order: doubling the
    number of steps divides their error by about four (about 1 % at 40 Karras steps, 0.06 % at 160).
    """
    sigma_data, sigma_max = 0.5, 80.0
    x, y = make_inputs(dtype=torch.float64)
    start = y[DATASET_NAME].data

    def gaussian_denoiser(x: Batch, y: Batch, sigma: dict[str, torch.Tensor], model_comm_group=None) -> Batch:
        del x, model_comm_group
        return y.with_data(
            {name: source.data * sigma_data**2 / (sigma_data**2 + sigma[name] ** 2) for name, source in y.items()},
        )

    def relative_error(num_steps: int) -> float:
        sigmas = KarrasSigmaSchedule(sigma_max=sigma_max, sigma_min=0.002, num_steps=num_steps, rho=7.0).get_schedule()
        result = sampler_cls(dtype=torch.float64).sample(x=x, y=y, sigmas=sigmas, denoising_fn=gaussian_denoiser)
        # The flow is linear, so the sample is the start scaled by one factor.
        factor = (result[DATASET_NAME].data * start).sum() / (start * start).sum()
        torch.testing.assert_close(result[DATASET_NAME].data, factor * start)
        return (factor / (sigma_data / math.sqrt(sigma_data**2 + sigma_max**2)) - 1).item()

    error_40, error_80, error_160 = relative_error(40), relative_error(80), relative_error(160)

    assert 3.5 < error_40 / error_80 < 4.5
    assert 3.5 < error_80 / error_160 < 4.5
    assert abs(error_160) < 1e-3


def test_dpmpp_hands_the_denoiser_its_solver_state_rounded_once() -> None:
    """DPM++ keeps its state in float64 and only rounds the copy handed to a float32 denoiser.

    With a zero denoiser each step scales the state by sigma_next / sigma, so at step k the denoiser must see
    the float64 state y_0 * sigma_k / sigma_0 rounded once to float32, without rounding errors carried over.
    """
    torch.manual_seed(0)
    x, _ = make_inputs(dtype=torch.float32)
    start = torch.randn(2, 3, 1, 5, 4, dtype=torch.float64)
    y = _batch({DATASET_NAME: start})
    sigmas = KarrasSigmaSchedule(sigma_max=80.0, sigma_min=0.002, num_steps=12, rho=7.0).get_schedule()
    received = []

    def zero_denoiser(x: Batch, y: Batch, sigma: dict[str, torch.Tensor], model_comm_group=None) -> Batch:
        del x, sigma, model_comm_group
        received.append(y[DATASET_NAME].data.clone())
        return y.with_data({name: torch.zeros_like(source.data) for name, source in y.items()})

    DPMpp2MSampler(dtype=torch.float64).sample(x=x, y=y, sigmas=sigmas, denoising_fn=zero_denoiser)

    state = start.clone()
    for step, seen in enumerate(received):
        assert seen.dtype == torch.float32
        torch.testing.assert_close(seen, state.to(torch.float32), rtol=0.0, atol=0.0)
        if step + 1 < len(sigmas) - 1:
            state = state * (sigmas[step + 1] / sigmas[step])
