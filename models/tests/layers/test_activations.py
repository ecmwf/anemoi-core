# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch
import torch.nn as nn
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from anemoi.models.layers.activations import GEGLU
from anemoi.models.layers.activations import GLU
from anemoi.models.layers.activations import ReGLU
from anemoi.models.layers.activations import Sine
from anemoi.models.layers.activations import SwiGLU


@st.composite
def tensor_strategy(draw, min_dims=3, max_dims=5, min_size=1, max_size=10):
    """Generate random tensors with controlled precision and size."""
    shape = draw(st.lists(st.integers(min_size, max_size), min_size=min_dims, max_size=max_dims))
    # Use width=32 to match float32 precision and avoid representation errors
    # Use allow_subnormal=False to prevent subnormal float issues
    array = draw(
        arrays(
            np.float32,
            shape,
            elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False, allow_subnormal=False, width=32),
        )
    )
    return torch.tensor(array, dtype=torch.float32)


class TestGLU:

    GLUS = [GLU, SwiGLU, ReGLU, GEGLU]

    common_strategy = given(x=tensor_strategy(), activation=st.sampled_from(GLUS), bias=st.booleans())

    @common_strategy
    def test_glu_property_shape_preservation(self, activation, x, bias):
        glu = activation(bias=bias)
        output = glu(x)
        assert output.shape == x.shape

    @common_strategy
    def test_glu_nan_free(self, activation, x, bias):
        glu = activation(bias=bias)
        output = glu(x)
        assert not torch.isnan(output).any()

    @common_strategy
    def test_glu_dim_init(self, activation, x, bias):
        glu = activation(bias=bias)

        assert glu.dim is None
        assert glu.projection is True
        assert glu.W is None
        assert glu.V is None

        _ = glu(x)

        assert glu.dim == x.shape[-1]
        assert isinstance(glu.W, nn.Linear)
        assert isinstance(glu.V, nn.Linear)
        assert isinstance(glu.projection, nn.Linear)

    @common_strategy
    @settings(max_examples=10, deadline=None)
    def test_glu_backward_pass(self, activation, x, bias):
        glu = activation(bias=bias)
        output = glu(x)
        output.sum().backward()

        assert x.grad is None
        assert not torch.isnan(glu.W.weight.grad).any()
        assert not torch.isnan(glu.V.weight.grad).any()
        if bias:
            assert not torch.isnan(glu.W.bias.grad).any()
            assert not torch.isnan(glu.V.bias.grad).any()

    @given(
        x=tensor_strategy(min_dims=3, max_dims=3),
        variation=st.sampled_from([nn.ReLU(), nn.Tanh(), nn.LeakyReLU()]),
        projection=st.booleans(),
        bias=st.booleans(),
    )
    def test_glu_custom_activations_property(self, x, variation, projection, bias):
        glu = GLU(variation=variation, projection=projection, bias=bias)
        output = glu(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_glu_computation(self):
        # Fixed test with deterministic output
        x = torch.ones(1, 1, 2)
        glu = GLU(projection=False, bias=True)

        # Set weights and biases manually
        glu._post_init(x)
        with torch.no_grad():
            glu.W.weight.fill_(0.5)
            glu.W.bias.fill_(0.1)
            glu.V.weight.fill_(0.3)
            glu.V.bias.fill_(0.2)

        output = glu(x)
        expected = torch.sigmoid(torch.tensor(1.1)) * 0.8
        assert torch.isclose(output[0, 0, 0], expected, atol=1e-5)


class TestSine:

    @given(w=st.floats(0.1, 10.0), phi=st.floats(-np.pi, np.pi), x=tensor_strategy(min_dims=1, max_dims=4))
    def test_sine_properties(self, w, phi, x):
        sine = Sine(w=w, phi=phi)
        output = sine(x)

        # Check output shape matches input
        assert output.shape == x.shape

        # Ensure all values are between -1 and 1 (sine range)
        assert (output >= -1.0).all() and (output <= 1.0).all()

        # For specific inputs, check if periodicity is preserved
        if x.numel() > 0:
            # Pick first element to test periodicity
            x_val = x.flatten()[0].item()
            period = 2 * np.pi / w

            # Create two inputs separated by exactly one period
            x1 = torch.tensor([x_val], dtype=torch.float32)
            x2 = torch.tensor([x_val + period], dtype=torch.float32)

            # Outputs should be almost equal
            out1 = sine(x1)
            out2 = sine(x2)
            assert torch.isclose(out1, out2, atol=1e-5)

    @pytest.mark.parametrize(
        "w,phi,x,expected",
        [
            (2.0, 0.0, torch.tensor(0.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)),
            (
                2.0,
                0.0,
                torch.tensor(torch.pi / 4, dtype=torch.float32),
                torch.tensor(torch.sin(torch.tensor(torch.pi / 2)), dtype=torch.float32),
            ),
            (1.0, torch.pi / 2, torch.tensor(0.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)),
        ],
    )
    def test_sine_parameterized(self, w, phi, x, expected):
        sine = Sine(w=w, phi=phi)
        output = sine(x)
        assert torch.isclose(output, expected, atol=1e-6)
