# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Cartesian spectral transforms of inputs without variables."""

import pytest
import torch

from anemoi.models.layers.spectral_transforms import DCT2D
from anemoi.models.layers.spectral_transforms import FFT2D

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")),
]
TRANSFORMS = {
    "fft2d": lambda: FFT2D(x_dim=8, y_dim=6),
    "fft2d, filtered": lambda: FFT2D(x_dim=8, y_dim=6, apply_filter=True),
    "fft2d, padded patches": lambda: FFT2D(
        x_dim=8, y_dim=6, patch_size=(4, 3), patch_stride=(3, 3), patch_padding=True, apply_filter=True
    ),
    "dct2d": lambda: DCT2D(x_dim=8, y_dim=6),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", TRANSFORMS)
def test_transforms_of_no_variables_are_empty_and_pass_gradients(device: str, name: str) -> None:
    """Without variables, the coefficients have the shape and dtype they have with variables, but are empty.

    The backward pass still reaches the input.
    """
    if name == "dct2d":
        pytest.importorskip("torch_dct")
    transform = TRANSFORMS[name]()
    with_variables = transform(torch.randn(1, 2, 3, 48, 1, device=device))
    x = torch.zeros(1, 2, 3, 48, 0, device=device, requires_grad=True)
    coefficients = transform(x)
    assert coefficients.shape == (*with_variables.shape[:-1], 0)
    assert coefficients.dtype == with_variables.dtype
    coefficients.abs().sum().backward()
    assert x.grad.shape == x.shape
