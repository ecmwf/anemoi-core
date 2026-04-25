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
from anemoi.models.samplers.diffusion_samplers import ExperimentalSamplerScheduler
from anemoi.models.samplers.diffusion_samplers import ExponentialScheduler
from anemoi.models.samplers.diffusion_samplers import KarrasScheduler
from anemoi.models.samplers.diffusion_samplers import LinearScheduler
from anemoi.models.samplers.diffusion_samplers import NoiseScheduler
from anemoi.models.samplers.diffusion_samplers import NOISE_SCHEDULERS

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


def test_noise_scheduler_canonicalizes_explicit_near_zero_final_value_to_zero() -> None:
    final_sigma = DEFAULT_FINAL_SIGMA_EPS / 10
    scheduler = DummyScheduler(
        torch.tensor([1.0, 0.7, 0.4, 0.1, final_sigma], dtype=torch.float64),
        num_steps=4,
    )

    sigmas = scheduler.get_schedule(dtype_compute=torch.float64)

    assert sigmas.shape == (5,)
    assert sigmas[-1].item() == 0.0


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


class TestEDMHeunSampler:
    """Test suite for EDM Heun sampler."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, time_steps, ensemble_size, grid_size, vars_size = 2, 3, 1, 10, 5

        x = torch.randn(batch_size, time_steps, ensemble_size, grid_size, vars_size)
        y = torch.randn(batch_size, ensemble_size, grid_size, vars_size)

        # Create a simple noise schedule
        num_steps = 5
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

        return x, y, sigmas

    @pytest.fixture
    def mock_denoising_fn(self):
        """Create mock denoising function."""
        return MockDenoisingFunction(deterministic=True)

    def test_basic_functionality(self, sample_data, mock_denoising_fn):
        """Test basic functionality of EDM Heun sampler."""
        x, y, sigmas = sample_data

        sampler = EDMHeunSampler()
        result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_denoising_fn)

        # Check output shape
        assert result.shape == y.shape

        # Check that denoising function was called
        assert mock_denoising_fn.call_count > 0

        # Check that result is finite
        assert torch.isfinite(result).all()

    def test_output_shape_consistency(self, mock_denoising_fn):
        """Test that output shape matches input shape for various dimensions."""
        test_shapes = [
            (1, 2, 1, 5, 3),  # Small
            (3, 4, 2, 20, 10),  # Medium
            (2, 1, 1, 8, 8),  # Square grid
        ]

        for shape in test_shapes:
            batch_size, time_steps, ensemble_size, grid_size, vars_size = shape
            x = torch.randn(batch_size, time_steps, ensemble_size, grid_size, vars_size)
            y = torch.randn(batch_size, ensemble_size, grid_size, vars_size)
            sigmas = torch.linspace(1.0, 0.0, 6)  # 5 steps

            mock_denoising_fn.call_count = 0  # Reset counter

            sampler = EDMHeunSampler()
            result = sampler.sample(x, y, sigmas, mock_denoising_fn)

            assert result.shape == y.shape
            assert torch.isfinite(result).all()

    @pytest.mark.parametrize("num_steps", [1, 3, 10, 20])
    def test_different_step_counts(self, mock_denoising_fn, num_steps):
        """Test sampler with different numbers of steps."""
        x = torch.randn(1, 2, 1, 5, 3)
        y = torch.randn(1, 1, 5, 3)
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

        mock_denoising_fn.call_count = 0

        sampler = EDMHeunSampler()
        result = sampler.sample(x, y, sigmas, mock_denoising_fn)

        assert result.shape == y.shape
        # For Heun method, we expect roughly 2 calls per step (first order + correction)
        # except for the last step which might not have correction
        expected_min_calls = num_steps
        expected_max_calls = num_steps * 2
        assert expected_min_calls <= mock_denoising_fn.call_count <= expected_max_calls

    @pytest.mark.parametrize("S_churn", [0.0, 0.1, 0.5])
    def test_stochastic_churn_parameter(self, sample_data, S_churn):
        """Test different stochastic churn values."""
        x, y, sigmas = sample_data
        mock_denoising_fn = MockDenoisingFunction(deterministic=True)

        sampler = EDMHeunSampler(S_churn=S_churn)
        result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_denoising_fn)

        assert result.shape == y.shape
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("S_min,S_max", [(0.0, 1.0), (0.1, 0.8), (0.0, float("inf"))])
    def test_churn_range_parameters(self, sample_data, S_min, S_max):
        """Test different churn range parameters."""
        x, y, sigmas = sample_data
        mock_denoising_fn = MockDenoisingFunction(deterministic=True)

        sampler = EDMHeunSampler(S_churn=0.2, S_min=S_min, S_max=S_max)
        result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_denoising_fn)

        assert result.shape == y.shape
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("S_noise", [0.5, 1.0, 1.5])
    def test_noise_scale_parameter(self, sample_data, S_noise):
        """Test different noise scale values."""
        x, y, sigmas = sample_data
        mock_denoising_fn = MockDenoisingFunction(deterministic=True)

        sampler = EDMHeunSampler(S_noise=S_noise)
        result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_denoising_fn)

        assert result.shape == y.shape
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, sample_data, dtype):
        """Test sampler with different data types."""
        x, y, sigmas = sample_data
        mock_denoising_fn = MockDenoisingFunction(deterministic=True)

        sampler = EDMHeunSampler(dtype=dtype)
        result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_denoising_fn)

        assert result.shape == y.shape
        assert torch.isfinite(result).all()

    def test_deterministic_behavior(self, sample_data):
        """Test that sampler produces deterministic results with same inputs."""
        x, y, sigmas = sample_data

        # Run twice with same seed
        torch.manual_seed(42)
        mock_fn1 = MockDenoisingFunction(deterministic=True)
        sampler1 = EDMHeunSampler(S_churn=0.0)
        result1 = sampler1.sample(x, y.clone(), sigmas, mock_fn1)

        torch.manual_seed(42)
        mock_fn2 = MockDenoisingFunction(deterministic=True)
        sampler2 = EDMHeunSampler(S_churn=0.0)
        result2 = sampler2.sample(x, y.clone(), sigmas, mock_fn2)

        assert torch.allclose(result1, result2, atol=1e-6)

    def test_noise_reduction_progression(self, sample_data):
        """Test that sampler progressively reduces noise."""
        x, y, sigmas = sample_data
        mock_denoising_fn = MockDenoisingFunction(noise_reduction_factor=0.8, deterministic=True)

        # Store initial noise level
        initial_norm = torch.norm(y)

        sampler = EDMHeunSampler()
        result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_denoising_fn)

        final_norm = torch.norm(result)

        # With our mock function that reduces noise by 20% each step,
        # the final result should have lower norm than initial
        assert torch.isfinite(result).all()
        assert final_norm >= 0  # Basic sanity check
        assert (
            final_norm < initial_norm
        ), f"Expected noise reduction: final_norm ({final_norm}) should be < initial_norm ({initial_norm})"


class TestDPMPP2MSampler:
    """Test suite for DPM++ 2M sampler."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, time_steps, ensemble_size, grid_size, vars_size = 2, 3, 1, 10, 5

        x = torch.randn(batch_size, time_steps, ensemble_size, grid_size, vars_size)
        y = torch.randn(batch_size, ensemble_size, grid_size, vars_size)

        # Create a simple noise schedule
        num_steps = 5
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

        return x, y, sigmas

    @pytest.fixture
    def mock_denoising_fn(self):
        """Create mock denoising function."""
        return MockDenoisingFunction(deterministic=True)

    def test_basic_functionality(self, sample_data, mock_denoising_fn):
        """Test basic functionality of DPM++ 2M sampler."""
        x, y, sigmas = sample_data

        sampler = DPMpp2MSampler()
        result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_denoising_fn)

        # Check output shape
        assert result.shape == y.shape

        # Check that denoising function was called
        assert mock_denoising_fn.call_count > 0

        # Check that result is finite
        assert torch.isfinite(result).all()

    def test_output_shape_consistency(self, mock_denoising_fn):
        """Test that output shape matches input shape for various dimensions."""
        test_shapes = [
            (1, 2, 1, 5, 3),  # Small
            (3, 4, 2, 20, 10),  # Medium
            (2, 1, 1, 8, 8),  # Square grid
        ]

        for shape in test_shapes:
            batch_size, time_steps, ensemble_size, grid_size, vars_size = shape
            x = torch.randn(batch_size, time_steps, ensemble_size, grid_size, vars_size)
            y = torch.randn(batch_size, ensemble_size, grid_size, vars_size)
            sigmas = torch.linspace(1.0, 0.0, 6)  # 5 steps

            mock_denoising_fn.call_count = 0  # Reset counter

            sampler = DPMpp2MSampler()
            result = sampler.sample(x, y, sigmas, mock_denoising_fn)

            assert result.shape == y.shape
            assert torch.isfinite(result).all()

    @pytest.mark.parametrize("num_steps", [1, 3, 10, 20])
    def test_different_step_counts(self, mock_denoising_fn, num_steps):
        """Test sampler with different numbers of steps."""
        x = torch.randn(1, 2, 1, 5, 3)
        y = torch.randn(1, 1, 5, 3)
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

        mock_denoising_fn.call_count = 0

        sampler = DPMpp2MSampler()
        result = sampler.sample(x, y, sigmas, mock_denoising_fn)

        assert result.shape == y.shape
        # DPM++ 2M should call denoising function once per step
        assert mock_denoising_fn.call_count == num_steps

    def test_deterministic_behavior(self, sample_data):
        """Test that sampler produces deterministic results with same inputs."""
        x, y, sigmas = sample_data

        # Run twice with same inputs
        mock_fn1 = MockDenoisingFunction(deterministic=True)
        sampler1 = DPMpp2MSampler()
        result1 = sampler1.sample(x, y.clone(), sigmas, mock_fn1)

        mock_fn2 = MockDenoisingFunction(deterministic=True)
        sampler2 = DPMpp2MSampler()
        result2 = sampler2.sample(x, y.clone(), sigmas, mock_fn2)

        assert torch.allclose(result1, result2, atol=1e-6)

    def test_zero_final_sigma(self, sample_data, mock_denoising_fn):
        """Test behavior when final sigma is zero."""
        x, y, sigmas = sample_data

        # Ensure final sigma is exactly zero
        sigmas[-1] = 0.0

        sampler = DPMpp2MSampler()
        result = sampler.sample(x, y, sigmas, mock_denoising_fn)

        assert result.shape == y.shape
        assert torch.isfinite(result).all()

    def test_numerical_stability_small_sigmas(self, mock_denoising_fn):
        """Test numerical stability with very small sigma values."""
        x = torch.randn(1, 2, 1, 5, 3)
        y = torch.randn(1, 1, 5, 3)

        # Create schedule with very small sigmas
        sigmas = torch.tensor([1e-3, 1e-4, 1e-5, 0.0])

        sampler = DPMpp2MSampler()
        result = sampler.sample(x, y, sigmas, mock_denoising_fn)

        assert result.shape == y.shape
        assert torch.isfinite(result).all()
        assert not torch.isnan(result).any()


class TestSamplerComparison:
    """Test suite comparing different samplers."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, time_steps, ensemble_size, grid_size, vars_size = 2, 3, 1, 10, 5

        x = torch.randn(batch_size, time_steps, ensemble_size, grid_size, vars_size)
        y = torch.randn(batch_size, ensemble_size, grid_size, vars_size)

        # Create a simple noise schedule
        num_steps = 5
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

        return x, y, sigmas

    def test_samplers_produce_different_results(self, sample_data):
        """Test that different samplers produce different results."""
        x, y, sigmas = sample_data

        # Use different mock functions to ensure different behavior
        mock_fn1 = MockDenoisingFunction(deterministic=True, noise_reduction_factor=0.8)
        mock_fn2 = MockDenoisingFunction(deterministic=True, noise_reduction_factor=0.8)

        sampler_heun = EDMHeunSampler(S_churn=0.0)
        result_heun = sampler_heun.sample(x, y.clone(), sigmas, mock_fn1)

        sampler_dpmpp = DPMpp2MSampler()
        result_dpmpp = sampler_dpmpp.sample(x, y.clone(), sigmas, mock_fn2)

        # Convert to same dtype for comparison
        result_heun = result_heun.to(result_dpmpp.dtype)

        # Results should be different (unless by coincidence)
        assert not torch.allclose(result_heun, result_dpmpp, atol=1e-6)

    def test_samplers_same_output_shape(self, sample_data):
        """Test that all samplers produce the same output shape."""
        x, y, sigmas = sample_data

        mock_fn1 = MockDenoisingFunction(deterministic=True)
        mock_fn2 = MockDenoisingFunction(deterministic=True)

        sampler_heun = EDMHeunSampler()
        result_heun = sampler_heun.sample(x, y.clone(), sigmas, mock_fn1)
        sampler_dpmpp = DPMpp2MSampler()
        result_dpmpp = sampler_dpmpp.sample(x, y.clone(), sigmas, mock_fn2)

        assert result_heun.shape == result_dpmpp.shape == y.shape

    @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    def test_device_compatibility(self, sample_data, device):
        """Test that samplers work on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x, y, sigmas = sample_data
        x = x.to(device)
        y = y.to(device)
        sigmas = sigmas.to(device)

        # Create device-aware mock function
        class DeviceMockDenoisingFunction(MockDenoisingFunction):
            def __call__(self, x, y, sigma, model_comm_group=None, grid_shard_shapes=None):
                result = super().__call__(x, y, sigma, model_comm_group, grid_shard_shapes)
                return result.to(device)

        mock_fn1 = DeviceMockDenoisingFunction(deterministic=True)
        mock_fn2 = DeviceMockDenoisingFunction(deterministic=True)

        sampler_heun = EDMHeunSampler()
        result_heun = sampler_heun.sample(x, y.clone(), sigmas, mock_fn1)
        sampler_dpmpp = DPMpp2MSampler()
        result_dpmpp = sampler_dpmpp.sample(x, y.clone(), sigmas, mock_fn2)

        assert result_heun.device.type == device
        assert result_dpmpp.device.type == device
        assert torch.isfinite(result_heun).all()
        assert torch.isfinite(result_dpmpp).all()


class TestSamplerEdgeCases:
    """Test edge cases and error conditions for samplers."""

    def test_single_step_sampling(self):
        """Test samplers with only one step."""
        x = torch.randn(1, 2, 1, 5, 3)
        y = torch.randn(1, 1, 5, 3)
        sigmas = torch.tensor([1.0, 0.0])  # Only one step

        mock_fn1 = MockDenoisingFunction(deterministic=True)
        mock_fn2 = MockDenoisingFunction(deterministic=True)

        sampler_heun = EDMHeunSampler()
        result_heun = sampler_heun.sample(x, y.clone(), sigmas, mock_fn1)
        sampler_dpmpp = DPMpp2MSampler()
        result_dpmpp = sampler_dpmpp.sample(x, y.clone(), sigmas, mock_fn2)

        assert result_heun.shape == y.shape
        assert result_dpmpp.shape == y.shape
        assert torch.isfinite(result_heun).all()
        assert torch.isfinite(result_dpmpp).all()

    def test_large_batch_sizes(self):
        """Test samplers with large batch sizes."""
        batch_size = 10
        x = torch.randn(batch_size, 2, 1, 5, 3)
        y = torch.randn(batch_size, 1, 5, 3)
        sigmas = torch.linspace(1.0, 0.0, 4)  # 3 steps

        mock_fn1 = MockDenoisingFunction(deterministic=True)
        mock_fn2 = MockDenoisingFunction(deterministic=True)

        sampler_heun = EDMHeunSampler()
        result_heun = sampler_heun.sample(x, y.clone(), sigmas, mock_fn1)
        sampler_dpmpp = DPMpp2MSampler()
        result_dpmpp = sampler_dpmpp.sample(x, y.clone(), sigmas, mock_fn2)

        assert result_heun.shape == y.shape
        assert result_dpmpp.shape == y.shape
        assert torch.isfinite(result_heun).all()
        assert torch.isfinite(result_dpmpp).all()

    def test_multiple_ensemble_members(self):
        """Test samplers with multiple ensemble members."""
        ensemble_size = 5
        x = torch.randn(2, 3, ensemble_size, 10, 5)
        y = torch.randn(2, ensemble_size, 10, 5)
        sigmas = torch.linspace(1.0, 0.0, 4)  # 3 steps

        mock_fn1 = MockDenoisingFunction(deterministic=True)
        mock_fn2 = MockDenoisingFunction(deterministic=True)

        sampler_heun = EDMHeunSampler()
        result_heun = sampler_heun.sample(x, y.clone(), sigmas, mock_fn1)
        sampler_dpmpp = DPMpp2MSampler()
        result_dpmpp = sampler_dpmpp.sample(x, y.clone(), sigmas, mock_fn2)

        assert result_heun.shape == y.shape
        assert result_dpmpp.shape == y.shape
        assert torch.isfinite(result_heun).all()
        assert torch.isfinite(result_dpmpp).all()


class TestExperimentalSamplerScheduler:
    """Test suite for the piecewise experimental scheduler."""

    def test_piecewise_schedule_shape_and_endpoints(self):
        scheduler = ExperimentalSamplerScheduler(
            sigma_max=1e5,
            sigma_min=0.03,
            sigma_transition=10.0,
            num_steps=40,
            num_steps_high=20,
            num_steps_low=20,
        )

        sigmas = scheduler.get_schedule()

        assert sigmas.shape == (41,)
        assert torch.isclose(sigmas[0], torch.tensor(1e5, dtype=sigmas.dtype))
        assert torch.isclose(sigmas[20], torch.tensor(10.0, dtype=sigmas.dtype))
        assert torch.isclose(sigmas[-2], torch.tensor(0.03, dtype=sigmas.dtype))
        assert torch.isclose(sigmas[-1], torch.tensor(0.0, dtype=sigmas.dtype))

    def test_piecewise_schedule_is_monotone(self):
        scheduler = ExperimentalSamplerScheduler(
            sigma_max=1e5,
            sigma_min=0.03,
            sigma_transition=20.0,
            num_steps=40,
            num_steps_high=20,
            num_steps_low=20,
        )

        sigmas = scheduler.get_schedule()

        assert torch.all(sigmas[:-1] >= sigmas[1:])
        assert torch.all(sigmas[:-2] > 0.0)

    def test_piecewise_schedule_respects_even_split_default(self):
        scheduler = ExperimentalSamplerScheduler(
            sigma_max=1e5,
            sigma_min=0.03,
            sigma_transition=10.0,
            num_steps=40,
        )

        sigmas = scheduler.get_schedule()

        assert sigmas.shape == (41,)
        assert torch.isclose(sigmas[20], torch.tensor(10.0, dtype=sigmas.dtype))

    def test_piecewise_schedule_registry_name(self):
        scheduler_cls = NOISE_SCHEDULERS["experimental_sampler"]
        scheduler = scheduler_cls(
            sigma_max=1e5,
            sigma_min=0.03,
            sigma_transition=10.0,
            num_steps=40,
            num_steps_high=20,
            num_steps_low=20,
        )

        sigmas = scheduler.get_schedule()

        assert sigmas.shape == (41,)


def test_downscaler_sample_seeded_init_noise_is_deterministic(monkeypatch):
    from types import SimpleNamespace

    from anemoi.models.models.downscaler_encoder_processor_decoder import AnemoiDownscalingModelEncProcDec
    from anemoi.models.samplers import diffusion_samplers

    captured = []

    class _FixedScheduler:
        def __init__(self, **_kwargs):
            pass

        def get_schedule(self, device=None, dtype_compute=torch.float64, **_kwargs):
            return torch.tensor([5.0, 0.0], device=device, dtype=dtype_compute)

    class _CaptureSampler:
        def __init__(self, dtype, **_kwargs):
            self.dtype = dtype

        def sample(self, x, y, y_init, sigmas, denoising_fn, **kwargs):
            captured.append(y_init.detach().cpu().clone())
            return y_init

    monkeypatch.setitem(diffusion_samplers.NOISE_SCHEDULERS, "karras", _FixedScheduler)
    monkeypatch.setitem(diffusion_samplers.DIFFUSION_SAMPLERS, "heun", _CaptureSampler)

    model = AnemoiDownscalingModelEncProcDec.__new__(AnemoiDownscalingModelEncProcDec)
    model.inference_defaults = SimpleNamespace(
        noise_scheduler={
            "schedule_type": "karras",
            "sigma_max": 5.0,
            "sigma_min": 0.03,
            "num_steps": 1,
            "rho": 7.0,
        },
        diffusion_sampler={
            "sampler": "heun",
            "S_churn": 0.0,
            "S_min": 0.0,
            "S_max": 5.0,
            "S_noise": 1.0,
        },
    )
    model.num_output_channels = 2
    model.fwd_with_preconditioning = lambda *args, **kwargs: args[1]

    x = torch.ones(1, 1, 1, 3, 2)
    x_hres = torch.ones(1, 1, 1, 3, 2)

    out1 = model.sample(x, x_hres, seed=42)
    out2 = model.sample(x, x_hres, seed=42)
    out3 = model.sample(x, x_hres, seed=7)

    assert out1.shape == out2.shape == out3.shape
    assert torch.allclose(captured[0], captured[1])
    assert not torch.allclose(captured[0], captured[2])
