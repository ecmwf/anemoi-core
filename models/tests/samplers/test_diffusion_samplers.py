# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional

import pytest
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.samplers.diffusion_samplers import DPMpp2MSampler
from anemoi.models.samplers.diffusion_samplers import EDMHeunSampler

DATASET_NAME = "test_dataset"


class MockDenoisingFunction:
    """Mock denoising function for testing samplers."""

    def __init__(self, noise_reduction_factor: float = 0.9, deterministic: bool = False):
        """Initialize mock denoising function.

        Parameters
        ----------
        noise_reduction_factor : float
            Factor by which to reduce noise at each step (default: 0.9)
        deterministic : bool
            If True, use deterministic denoising; if False, add some randomness
        """
        self.noise_reduction_factor = noise_reduction_factor
        self.deterministic = deterministic
        self.call_count = 0

    def __call__(
        self,
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        sigma: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[dict[str, Optional[list]]] = None,
    ) -> torch.Tensor:
        """Mock denoising function that reduces noise proportionally to sigma."""
        self.call_count += 1

        # Simple denoising: reduce noise proportional to sigma
        # At high sigma (high noise), return mostly the conditioning x
        # At low sigma (low noise), return mostly the noisy y
        sigma_normalized = sigma / (sigma.max() + 1e-8)

        denoised = {}
        for dataset_name, y_ in y.items():
            if self.deterministic:
                # Deterministic denoising for reproducible tests
                denoised[dataset_name] = (1 - sigma_normalized * self.noise_reduction_factor) * y_
            else:
                # Add some controlled randomness
                denoised = (1 - sigma_normalized * self.noise_reduction_factor) * y_
                denoised += 0.01 * sigma_normalized * torch.randn_like(y_)

        return denoised


class TestEDMHeunSampler:
    """Test suite for EDM Heun sampler."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, time_steps, ensemble_size, grid_size, vars_size = 2, 3, 1, 10, 5

        x = {DATASET_NAME: torch.randn(batch_size, time_steps, ensemble_size, grid_size, vars_size)}
        y = {DATASET_NAME: torch.randn(batch_size, ensemble_size, grid_size, vars_size)}

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

        assert set(result.keys()) == set(y.keys())
        for dataset_name in result:
            # Check output shape
            assert result[dataset_name].shape == y[dataset_name].shape

            # Check that result is finite
            assert torch.isfinite(result[dataset_name]).all()

        # Check that denoising function was called
        assert mock_denoising_fn.call_count > 0

    def test_output_shape_consistency(self, mock_denoising_fn):
        """Test that output shape matches input shape for various dimensions."""
        test_shapes = [
            (1, 2, 1, 5, 3),  # Small
            (3, 4, 2, 20, 10),  # Medium
            (2, 1, 1, 8, 8),  # Square grid
        ]

        for shape in test_shapes:
            batch_size, time_steps, ensemble_size, grid_size, vars_size = shape
            x = {DATASET_NAME: torch.randn(batch_size, time_steps, ensemble_size, grid_size, vars_size)}
            y = {DATASET_NAME: torch.randn(batch_size, ensemble_size, grid_size, vars_size)}
            sigmas = torch.linspace(1.0, 0.0, 6)  # 5 steps

            mock_denoising_fn.call_count = 0  # Reset counter

            sampler = EDMHeunSampler()
            result = sampler.sample(x, y, sigmas, mock_denoising_fn)

            assert set(result.keys()) == set(y.keys())
            for dataset_name in result:
                assert result[dataset_name].shape == y[dataset_name].shape
                assert torch.isfinite(result[dataset_name]).all()

    @pytest.mark.parametrize("num_steps", [1, 3, 10, 20])
    def test_different_step_counts(self, mock_denoising_fn, num_steps):
        """Test sampler with different numbers of steps."""
        x = {DATASET_NAME: torch.randn(1, 2, 1, 5, 3)}
        y = {DATASET_NAME: torch.randn(1, 1, 5, 3)}
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

        mock_denoising_fn.call_count = 0

        sampler = EDMHeunSampler()
        result = sampler.sample(x, y, sigmas, mock_denoising_fn)

        assert set(result.keys()) == set(y.keys())
        for dataset_name in result:
            assert result[dataset_name].shape == y[dataset_name].shape

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

        assert set(result.keys()) == set(y.keys())
        for dataset_name in result:
            assert result[dataset_name].shape == y[dataset_name].shape
            assert torch.isfinite(result[dataset_name]).all()

    @pytest.mark.parametrize("S_min,S_max", [(0.0, 1.0), (0.1, 0.8), (0.0, float("inf"))])
    def test_churn_range_parameters(self, sample_data, S_min, S_max):
        """Test different churn range parameters."""
        x, y, sigmas = sample_data
        mock_denoising_fn = MockDenoisingFunction(deterministic=True)

        sampler = EDMHeunSampler(S_churn=0.2, S_min=S_min, S_max=S_max)
        result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_denoising_fn)

        assert set(result.keys()) == set(y.keys())
        for dataset_name in result:
            assert result[dataset_name].shape == y[dataset_name].shape
            assert torch.isfinite(result[dataset_name]).all()

    @pytest.mark.parametrize("S_noise", [0.5, 1.0, 1.5])
    def test_noise_scale_parameter(self, sample_data, S_noise):
        """Test different noise scale values."""
        x, y, sigmas = sample_data
        mock_denoising_fn = MockDenoisingFunction(deterministic=True)

        sampler = EDMHeunSampler(S_noise=S_noise)
        result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_denoising_fn)

        assert set(result.keys()) == set(y.keys())
        for dataset_name in result:
            assert result[dataset_name].shape == y[dataset_name].shape
            assert torch.isfinite(result[dataset_name]).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, sample_data, dtype):
        """Test sampler with different data types."""
        x, y, sigmas = sample_data
        mock_denoising_fn = MockDenoisingFunction(deterministic=True)

        sampler = EDMHeunSampler(dtype=dtype)
        result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_denoising_fn)

        assert set(result.keys()) == set(y.keys())
        for dataset_name in result:
            assert result[dataset_name].shape == y[dataset_name].shape
            assert torch.isfinite(result[dataset_name]).all()

    def test_deterministic_behavior(self, sample_data):
        """Test that sampler produces deterministic results with same inputs."""
        x, y, sigmas = sample_data

        # Run twice with same seed
        torch.manual_seed(42)
        mock_fn1 = MockDenoisingFunction(deterministic=True)
        sampler1 = EDMHeunSampler(S_churn=0.0)
        y_cloned = {k: v.clone() for k, v in y.items()}
        result1 = sampler1.sample(x, y_cloned, sigmas, mock_fn1)

        torch.manual_seed(42)
        mock_fn2 = MockDenoisingFunction(deterministic=True)
        sampler2 = EDMHeunSampler(S_churn=0.0)
        y_cloned = {k: v.clone() for k, v in y.items()}
        result2 = sampler2.sample(x, y_cloned, sigmas, mock_fn2)

        assert set(result1.keys()) == set(result2.keys())
        for dataset_name in result1:
            assert torch.allclose(result1[dataset_name], result2[dataset_name], atol=1e-6)

    def test_noise_reduction_progression(self, sample_data):
        """Test that sampler progressively reduces noise."""
        x, y, sigmas = sample_data
        mock_denoising_fn = MockDenoisingFunction(noise_reduction_factor=0.8, deterministic=True)

        # Store initial noise level
        initial_norm = {dataset_name: torch.norm(y_) for dataset_name, y_ in y.items()}

        sampler = EDMHeunSampler()
        result = sampler.sample(x=x, y=y, sigmas=sigmas, denoising_fn=mock_denoising_fn)

        final_norm = {dataset_name: torch.norm(result_) for dataset_name, result_ in result.items()}

        assert set(final_norm.keys()) == set(initial_norm.keys())

        # With our mock function that reduces noise by 20% each step,
        # the final result should have lower norm than initial
        for dataset_name in result:
            assert torch.isfinite(result[dataset_name]).all()
            assert final_norm[dataset_name] >= 0  # Basic sanity check
            assert (
                final_norm[dataset_name] < initial_norm[dataset_name]
            ), f"Expected noise reduction: final_norm ({final_norm[dataset_name]}) should be < initial_norm ({initial_norm[dataset_name]})"


class TestDPMPP2MSampler:
    """Test suite for DPM++ 2M sampler."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, time_steps, ensemble_size, grid_size, vars_size = 2, 3, 1, 10, 5

        x = {DATASET_NAME: torch.randn(batch_size, time_steps, ensemble_size, grid_size, vars_size)}
        y = {DATASET_NAME: torch.randn(batch_size, ensemble_size, grid_size, vars_size)}

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

        assert set(result.keys()) == set(y.keys())
        for dataset_name in result:
            # Check output shape
            assert result[dataset_name].shape == y[dataset_name].shape

            # Check that result is finite
            assert torch.isfinite(result[dataset_name]).all()

        # Check that denoising function was called
        assert mock_denoising_fn.call_count > 0

    def test_output_shape_consistency(self, mock_denoising_fn):
        """Test that output shape matches input shape for various dimensions."""
        test_shapes = [
            (1, 2, 1, 5, 3),  # Small
            (3, 4, 2, 20, 10),  # Medium
            (2, 1, 1, 8, 8),  # Square grid
        ]

        for shape in test_shapes:
            batch_size, time_steps, ensemble_size, grid_size, vars_size = shape
            x = {DATASET_NAME: torch.randn(batch_size, time_steps, ensemble_size, grid_size, vars_size)}
            y = {DATASET_NAME: torch.randn(batch_size, ensemble_size, grid_size, vars_size)}
            sigmas = torch.linspace(1.0, 0.0, 6)  # 5 steps

            mock_denoising_fn.call_count = 0  # Reset counter

            sampler = DPMpp2MSampler()
            result = sampler.sample(x, y, sigmas, mock_denoising_fn)
            assert set(result.keys()) == set(y.keys())
            for dataset_name in result:
                assert result[dataset_name].shape == y[dataset_name].shape
                assert torch.isfinite(result[dataset_name]).all()

    @pytest.mark.parametrize("num_steps", [1, 3, 10, 20])
    def test_different_step_counts(self, mock_denoising_fn, num_steps):
        """Test sampler with different numbers of steps."""
        x = {DATASET_NAME: torch.randn(1, 2, 1, 5, 3)}
        y = {DATASET_NAME: torch.randn(1, 1, 5, 3)}
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

        mock_denoising_fn.call_count = 0

        sampler = DPMpp2MSampler()
        result = sampler.sample(x, y, sigmas, mock_denoising_fn)

        assert set(result.keys()) == set(y.keys())
        for dataset_name in result:
            assert result[dataset_name].shape == y[dataset_name].shape

        # DPM++ 2M should call denoising function once per step
        assert mock_denoising_fn.call_count == num_steps

    def test_deterministic_behavior(self, sample_data):
        """Test that sampler produces deterministic results with same inputs."""
        x, y, sigmas = sample_data

        # Run twice with same inputs
        mock_fn1 = MockDenoisingFunction(deterministic=True)
        sampler1 = DPMpp2MSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result1 = sampler1.sample(x, y_cloned, sigmas, mock_fn1)

        mock_fn2 = MockDenoisingFunction(deterministic=True)
        sampler2 = DPMpp2MSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result2 = sampler2.sample(x, y_cloned, sigmas, mock_fn2)

        assert set(result1.keys()) == set(result2.keys())
        for dataset_name in result1:
            assert torch.allclose(result1[dataset_name], result2[dataset_name], atol=1e-6)

    def test_zero_final_sigma(self, sample_data, mock_denoising_fn):
        """Test behavior when final sigma is zero."""
        x, y, sigmas = sample_data

        # Ensure final sigma is exactly zero
        sigmas[-1] = 0.0

        sampler = DPMpp2MSampler()
        result = sampler.sample(x, y, sigmas, mock_denoising_fn)

        assert set(result.keys()) == set(y.keys())
        for dataset_name in result:
            assert result[dataset_name].shape == y[dataset_name].shape
            assert torch.isfinite(result[dataset_name]).all()

    def test_numerical_stability_small_sigmas(self, mock_denoising_fn):
        """Test numerical stability with very small sigma values."""
        x = {DATASET_NAME: torch.randn(1, 2, 1, 5, 3)}
        y = {DATASET_NAME: torch.randn(1, 1, 5, 3)}

        # Create schedule with very small sigmas
        sigmas = torch.tensor([1e-3, 1e-4, 1e-5, 0.0])

        sampler = DPMpp2MSampler()
        result = sampler.sample(x, y, sigmas, mock_denoising_fn)
        assert set(result.keys()) == set(y.keys())
        for dataset_name in result:
            assert result[dataset_name].shape == y[dataset_name].shape
            assert torch.isfinite(result[dataset_name]).all()
            assert not torch.isnan(result[dataset_name]).any()


class TestSamplerComparison:
    """Test suite comparing different samplers."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, time_steps, ensemble_size, grid_size, vars_size = 2, 3, 1, 10, 5

        x = {DATASET_NAME: torch.randn(batch_size, time_steps, ensemble_size, grid_size, vars_size)}
        y = {DATASET_NAME: torch.randn(batch_size, ensemble_size, grid_size, vars_size)}

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
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_heun = sampler_heun.sample(x, y_cloned, sigmas, mock_fn1)

        sampler_dpmpp = DPMpp2MSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_dpmpp = sampler_dpmpp.sample(x, y_cloned, sigmas, mock_fn2)

        # Convert to same dtype for comparison
        result_heun = {k: v.to(result_dpmpp[k].dtype) for k, v in result_heun.items()}

        assert set(result_heun.keys()) == set(result_dpmpp.keys())
        for dataset_name in result_heun:
            # Results should be different (unless by coincidence)
            assert not torch.allclose(result_heun[dataset_name], result_dpmpp[dataset_name], atol=1e-6)

    def test_samplers_same_output_shape(self, sample_data):
        """Test that all samplers produce the same output shape."""
        x, y, sigmas = sample_data

        mock_fn1 = MockDenoisingFunction(deterministic=True)
        mock_fn2 = MockDenoisingFunction(deterministic=True)

        sampler_heun = EDMHeunSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_heun = sampler_heun.sample(x, y_cloned, sigmas, mock_fn1)
        sampler_dpmpp = DPMpp2MSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_dpmpp = sampler_dpmpp.sample(x, y_cloned, sigmas, mock_fn2)

        assert set(result_heun.keys()) == set(result_dpmpp.keys())
        for dataset_name in result_heun:
            assert result_heun[dataset_name].shape == result_dpmpp[dataset_name].shape == y[dataset_name].shape

    @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    def test_device_compatibility(self, sample_data, device):
        """Test that samplers work on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x, y, sigmas = sample_data
        for dataset_name in x:
            x[dataset_name] = x[dataset_name].to(device)
            y[dataset_name] = y[dataset_name].to(device)
        sigmas = sigmas.to(device)

        # Create device-aware mock function
        class DeviceMockDenoisingFunction(MockDenoisingFunction):
            def __call__(self, x, y, sigma, model_comm_group=None, grid_shard_shapes=None):
                result = super().__call__(x, y, sigma, model_comm_group, grid_shard_shapes)
                for dataset_name in result:
                    result[dataset_name] = result[dataset_name].to(device)
                return result

        mock_fn1 = DeviceMockDenoisingFunction(deterministic=True)
        mock_fn2 = DeviceMockDenoisingFunction(deterministic=True)

        sampler_heun = EDMHeunSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_heun = sampler_heun.sample(x, y_cloned, sigmas, mock_fn1)
        sampler_dpmpp = DPMpp2MSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_dpmpp = sampler_dpmpp.sample(x, y_cloned, sigmas, mock_fn2)

        assert set(result_heun.keys()) == set(result_dpmpp.keys())
        for dataset_name in result_heun:
            assert result_heun[dataset_name].device.type == device
            assert result_dpmpp[dataset_name].device.type == device
            assert torch.isfinite(result_heun[dataset_name]).all()
            assert torch.isfinite(result_dpmpp[dataset_name]).all()


class TestSamplerEdgeCases:
    """Test edge cases and error conditions for samplers."""

    def test_single_step_sampling(self):
        """Test samplers with only one step."""
        x = {DATASET_NAME: torch.randn(1, 2, 1, 5, 3)}
        y = {DATASET_NAME: torch.randn(1, 1, 5, 3)}
        sigmas = torch.tensor([1.0, 0.0])  # Only one step

        mock_fn1 = MockDenoisingFunction(deterministic=True)
        mock_fn2 = MockDenoisingFunction(deterministic=True)

        sampler_heun = EDMHeunSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_heun = sampler_heun.sample(x, y_cloned, sigmas, mock_fn1)
        sampler_dpmpp = DPMpp2MSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_dpmpp = sampler_dpmpp.sample(x, y_cloned, sigmas, mock_fn2)

        assert set(result_heun.keys()) == set(result_dpmpp.keys())
        for dataset_name in result_heun:
            assert result_heun[dataset_name].shape == y[dataset_name].shape
            assert result_dpmpp[dataset_name].shape == y[dataset_name].shape
            assert torch.isfinite(result_heun[dataset_name]).all()
            assert torch.isfinite(result_dpmpp[dataset_name]).all()

    def test_large_batch_sizes(self):
        """Test samplers with large batch sizes."""
        batch_size = 10
        x = {DATASET_NAME: torch.randn(batch_size, 2, 1, 5, 3)}
        y = {DATASET_NAME: torch.randn(batch_size, 1, 5, 3)}
        sigmas = torch.linspace(1.0, 0.0, 4)  # 3 steps

        mock_fn1 = MockDenoisingFunction(deterministic=True)
        mock_fn2 = MockDenoisingFunction(deterministic=True)

        sampler_heun = EDMHeunSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_heun = sampler_heun.sample(x, y_cloned, sigmas, mock_fn1)
        sampler_dpmpp = DPMpp2MSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_dpmpp = sampler_dpmpp.sample(x, y_cloned, sigmas, mock_fn2)

        assert set(result_heun.keys()) == set(result_dpmpp.keys())
        for dataset_name in result_heun:
            assert result_heun[dataset_name].shape == y[dataset_name].shape
            assert result_dpmpp[dataset_name].shape == y[dataset_name].shape
            assert torch.isfinite(result_heun[dataset_name]).all()
            assert torch.isfinite(result_dpmpp[dataset_name]).all()

    def test_multiple_ensemble_members(self):
        """Test samplers with multiple ensemble members."""
        ensemble_size = 5
        x = {DATASET_NAME: torch.randn(2, 3, ensemble_size, 10, 5)}
        y = {DATASET_NAME: torch.randn(2, ensemble_size, 10, 5)}
        sigmas = torch.linspace(1.0, 0.0, 4)  # 3 steps

        mock_fn1 = MockDenoisingFunction(deterministic=True)
        mock_fn2 = MockDenoisingFunction(deterministic=True)

        sampler_heun = EDMHeunSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_heun = sampler_heun.sample(x, y_cloned, sigmas, mock_fn1)
        sampler_dpmpp = DPMpp2MSampler()
        y_cloned = {k: v.clone() for k, v in y.items()}
        result_dpmpp = sampler_dpmpp.sample(x, y_cloned, sigmas, mock_fn2)

        assert set(result_heun.keys()) == set(result_dpmpp.keys())
        for dataset_name in result_heun:
            assert result_heun[dataset_name].shape == y[dataset_name].shape
            assert result_dpmpp[dataset_name].shape == y[dataset_name].shape
            assert torch.isfinite(result_heun[dataset_name]).all()
            assert torch.isfinite(result_dpmpp[dataset_name]).all()
