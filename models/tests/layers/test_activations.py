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
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from anemoi.models.layers.activations import Sine


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
                torch.tensor(torch.sin(torch.tensor(torch.pi / 2)).item(), dtype=torch.float32),
            ),
            (1.0, torch.pi / 2, torch.tensor(0.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)),
        ],
    )
    def test_sine_parameterized(self, w, phi, x, expected):
        sine = Sine(w=w, phi=phi)
        output = sine(x)
        assert torch.isclose(output, expected, atol=1e-6)
