# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.layers.spectral_transforms import DCT2D
from anemoi.models.layers.spectral_transforms import FFT2D
from anemoi.models.layers.spectral_transforms import InverseDCT2D
from anemoi.models.layers.spectral_transforms import InverseFFT2D


@pytest.mark.parametrize(("forward_cls", "inverse_cls"), [(FFT2D, InverseFFT2D), (DCT2D, InverseDCT2D)])
def test_planar_inverse_restores_the_field(forward_cls, inverse_cls):
    x_dim, y_dim = 6, 4
    data = torch.randn(2, 1, 3, y_dim * x_dim, 5, dtype=torch.float64)

    coeffs = forward_cls(x_dim=x_dim, y_dim=y_dim)(data)  # [..., y, x, variables]
    restored = inverse_cls(x_dim=x_dim, y_dim=y_dim)(coeffs.movedim(-1, -3)).transpose(-1, -2)

    torch.testing.assert_close(restored, data)
