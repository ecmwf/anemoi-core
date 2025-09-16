# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import pytest
import torch

from anemoi.models.layers.diffusion import RandomFourierEmbeddings
from anemoi.models.layers.diffusion import SinusoidalEmbeddings


class TestRandomFourierEmbeddings:
    """Test suite for RandomFourierEmbeddings."""

    def test_initialization_default(self):
        """Test RandomFourierEmbeddings initialization with default parameters."""
        embedder = RandomFourierEmbeddings()

        assert embedder.frequencies.shape == (16,)  # num_channels // 2 = 32 // 2
        assert abs(embedder.pi.item() - math.pi) < 1e-6
        assert isinstance(embedder.frequencies, torch.Tensor)
        assert isinstance(embedder.pi, torch.Tensor)

    def test_initialization_custom(self):
        """Test RandomFourierEmbeddings initialization with custom parameters."""
        num_channels = 64
        scale = 32
        embedder = RandomFourierEmbeddings(num_channels=num_channels, scale=scale)

        assert embedder.frequencies.shape == (num_channels // 2,)
        assert abs(embedder.pi.item() - math.pi) < 1e-6

    @pytest.mark.parametrize("num_channels", [16, 32, 64, 128])
    def test_output_shape(self, num_channels):
        """Test that output shape matches expected dimensions."""
        embedder = RandomFourierEmbeddings(num_channels=num_channels)

        batch_size = 4
        zdim = num_channels // 2
        x = torch.randn(batch_size, zdim)

        output = embedder(x)

        assert output.shape == (batch_size, num_channels)
        assert output.dtype == torch.float32

    def test_forward_computation(self):
        """Test that forward computation produces expected values."""
        torch.manual_seed(42)  # For reproducible results
        embedder = RandomFourierEmbeddings(num_channels=4, scale=1.0)

        x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])  # Shape: (2, 2) for num_channels=4
        output = embedder(x)

        # Check that output contains both sin and cos components
        assert output.shape == (2, 4)

        # Check that output is finite
        assert torch.isfinite(output).all()

    def test_different_inputs(self):
        """Test embedder with different input values."""
        embedder = RandomFourierEmbeddings(num_channels=32)
        zdim = 16  # num_channels // 2

        # Test with various input shapes and values
        inputs = [
            torch.randn(1, zdim),
            torch.randn(3, zdim),
            torch.randn(10, zdim),
            torch.randn(5, zdim),
        ]

        for x in inputs:
            output = embedder(x)
            assert output.shape == (x.shape[0], 32)
            assert not torch.isnan(output).any()
            assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test that gradients flow through the embedding layer."""
        embedder = RandomFourierEmbeddings(num_channels=32)
        x = torch.randn(4, 16, requires_grad=True)  # Shape: (4, 16) for num_channels=32

        output = embedder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        torch.manual_seed(42)
        embedder1 = RandomFourierEmbeddings(num_channels=32, scale=16)

        torch.manual_seed(42)
        embedder2 = RandomFourierEmbeddings(num_channels=32, scale=16)

        x = torch.randn(5, 16)  # Shape: (5, 16) for num_channels=32
        output1 = embedder1(x)
        output2 = embedder2(x)

        assert torch.allclose(output1, output2)

    def test_scale_effect(self):
        """Test that scale parameter affects the output."""
        torch.manual_seed(42)
        embedder_small = RandomFourierEmbeddings(num_channels=32, scale=1)

        torch.manual_seed(42)
        embedder_large = RandomFourierEmbeddings(num_channels=32, scale=100)

        x = torch.randn(1, 16)  # Shape: (1, 16) for num_channels=32
        output_small = embedder_small(x)
        output_large = embedder_large(x)

        # Outputs should be different due to different scales
        assert not torch.allclose(output_small, output_large)


class TestSinusoidalEmbeddings:
    """Test suite for SinusoidalEmbeddings."""

    def test_initialization_default(self):
        """Test SinusoidalEmbeddings initialization with default parameters."""
        embedder = SinusoidalEmbeddings()

        assert embedder.frequencies.shape == (16,)  # num_channels // 2 = 32 // 2
        assert isinstance(embedder.frequencies, torch.Tensor)

    def test_initialization_custom(self):
        """Test SinusoidalEmbeddings initialization with custom parameters."""
        num_channels = 64
        max_period = 5000
        embedder = SinusoidalEmbeddings(num_channels=num_channels, max_period=max_period)

        assert embedder.frequencies.shape == (num_channels // 2,)

    @pytest.mark.parametrize("num_channels", [16, 32, 64, 128])
    def test_output_shape(self, num_channels):
        """Test that output shape matches expected dimensions."""
        embedder = SinusoidalEmbeddings(num_channels=num_channels)

        batch_size = 4
        zdim = num_channels // 2
        x = torch.randn(batch_size, zdim)

        output = embedder(x)

        assert output.shape == (batch_size, num_channels)
        assert output.dtype == torch.float32

    def test_forward_computation(self):
        """Test that forward computation produces expected values."""
        embedder = SinusoidalEmbeddings(num_channels=4, max_period=10000)

        x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])  # Shape: (2, 2) for num_channels=4
        output = embedder(x)

        # Check that output contains both sin and cos components
        assert output.shape == (2, 4)

        # For x = 0, sin should be 0 and cos should be 1
        sin_part = output[0, :2]
        cos_part = output[0, 2:]

        assert torch.allclose(sin_part, torch.zeros_like(sin_part), atol=1e-6)
        assert torch.allclose(cos_part, torch.ones_like(cos_part), atol=1e-6)

    def test_frequency_computation(self):
        """Test that frequencies are computed correctly."""
        num_channels = 32
        max_period = 10000
        embedder = SinusoidalEmbeddings(num_channels=num_channels, max_period=max_period)

        zdim = num_channels // 2
        expected_frequencies = torch.exp(-math.log(max_period) * torch.arange(0, zdim) / zdim)

        assert torch.allclose(embedder.frequencies, expected_frequencies)

    def test_different_max_periods(self):
        """Test embedder with different max_period values."""
        x = torch.randn(3, 16)  # Shape: (3, 16) for num_channels=32

        embedder_short = SinusoidalEmbeddings(num_channels=32, max_period=100)
        embedder_long = SinusoidalEmbeddings(num_channels=32, max_period=10000)

        output_short = embedder_short(x)
        output_long = embedder_long(x)

        assert output_short.shape == output_long.shape
        # Outputs should be different due to different frequency ranges
        assert not torch.allclose(output_short, output_long)

    def test_different_inputs(self):
        """Test embedder with different input values."""
        embedder = SinusoidalEmbeddings(num_channels=32)
        zdim = 16  # num_channels // 2

        # Test with various input shapes and values
        inputs = [
            torch.randn(1, zdim),
            torch.randn(3, zdim),
            torch.randn(10, zdim),
            torch.randn(5, zdim),
        ]

        for x in inputs:
            output = embedder(x)
            assert output.shape == (x.shape[0], 32)
            assert not torch.isnan(output).any()
            assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test that gradients flow through the embedding layer."""
        embedder = SinusoidalEmbeddings(num_channels=32)
        x = torch.randn(4, 16, requires_grad=True)  # Shape: (4, 16) for num_channels=32

        output = embedder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        embedder1 = SinusoidalEmbeddings(num_channels=32, max_period=10000)
        embedder2 = SinusoidalEmbeddings(num_channels=32, max_period=10000)

        x = torch.randn(5, 16)  # Shape: (5, 16) for num_channels=32
        output1 = embedder1(x)
        output2 = embedder2(x)

        assert torch.allclose(output1, output2)

    def test_periodic_properties(self):
        """Test that embeddings have appropriate periodic properties."""
        embedder = SinusoidalEmbeddings(num_channels=32, max_period=10000)

        # Test that sin/cos components are bounded between -1 and 1
        x = torch.randn(100, 16) * 1000  # Large range of inputs
        output = embedder(x)

        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_frequency_ordering(self):
        """Test that frequencies are ordered from high to low."""
        embedder = SinusoidalEmbeddings(num_channels=32, max_period=10000)

        # Frequencies should be in descending order
        frequencies = embedder.frequencies
        assert torch.all(frequencies[:-1] >= frequencies[1:])


class TestNoiseEmbeddingsComparison:
    """Test suite comparing the two noise embedding methods."""

    def test_output_range_comparison(self):
        """Compare output ranges of both embedding methods."""
        x = torch.randn(100, 16) * 10  # Shape: (100, 16) for num_channels=32

        random_embedder = RandomFourierEmbeddings(num_channels=32, scale=1)
        sinusoidal_embedder = SinusoidalEmbeddings(num_channels=32, max_period=10000)

        random_output = random_embedder(x)
        sinusoidal_output = sinusoidal_embedder(x)

        # Both should be bounded
        assert torch.all(torch.abs(random_output) <= 1.0)
        assert torch.all(torch.abs(sinusoidal_output) <= 1.0)

    def test_different_behaviors(self):
        """Test that the two methods produce different outputs."""
        torch.manual_seed(42)
        x = torch.randn(10, 16)  # Shape: (10, 16) for num_channels=32

        random_embedder = RandomFourierEmbeddings(num_channels=32)
        sinusoidal_embedder = SinusoidalEmbeddings(num_channels=32)

        random_output = random_embedder(x)
        sinusoidal_output = sinusoidal_embedder(x)

        # Outputs should be different
        assert not torch.allclose(random_output, sinusoidal_output)

    @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    def test_device_compatibility(self, device):
        """Test that both embedders work on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = torch.randn(5, 16).to(device)  # Shape: (5, 16) for num_channels=32

        random_embedder = RandomFourierEmbeddings(num_channels=32).to(device)
        sinusoidal_embedder = SinusoidalEmbeddings(num_channels=32).to(device)

        random_output = random_embedder(x)
        sinusoidal_output = sinusoidal_embedder(x)

        assert random_output.device.type == device
        assert sinusoidal_output.device.type == device
