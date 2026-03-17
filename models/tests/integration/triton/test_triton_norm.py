# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch

from anemoi.models.triton.utils import is_triton_available

if is_triton_available():
    from anemoi.models.triton.norm import LayerNorm
    from anemoi.models.triton.norm import RMSNorm


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


@pytest.fixture(autouse=True)
def setup_torch(device):
    """Set up torch defaults for all tests."""
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    yield


@pytest.mark.slow
@pytest.mark.parametrize(
    "b,h,d,norm_type",
    [(B, H, C, norm_type) for B in (1, 4) for H in (2, 6) for C in (4, 6) for norm_type in ["rmsNorm", "layerNorm"]],
)
def test_norm_forward(b: int, h: int, d: int, norm_type: str):
    """Test forward pass of RMS Norm."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn((b, h, d))
    if norm_type == "rmsNorm":
        norm = RMSNorm()
        norm_ref = torch.nn.RMSNorm(d, elementwise_affine=False)
    elif norm_type == "layerNorm":
        norm = LayerNorm()
        norm_ref = torch.nn.LayerNorm(d, elementwise_affine=False, bias=False)

    x_triton = norm(x)

    x_ref = norm_ref(x)

    tolerance = 1e-4
    torch.testing.assert_close(x_triton, x_ref, atol=tolerance, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "b, h,d,norm_type",
    [(B, H, C, norm_type) for B in (1, 4) for H in (2, 6) for C in (4, 6) for norm_type in ["rmsNorm", "layerNorm"]],
)
def test_norm_backward(b: int, h: int, d: int, norm_type: str):
    """Test backward pass of RMS Norm."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn((b, h, d), requires_grad=True)

    if norm_type == "rmsNorm":
        norm = RMSNorm()
        norm_ref = torch.nn.RMSNorm(d, elementwise_affine=False)
    elif norm_type == "layerNorm":
        norm = LayerNorm()
        norm_ref = torch.nn.LayerNorm(d, elementwise_affine=False, bias=False)

    x_triton = norm(x)
    loss_triton = x_triton.pow(2).sum()
    loss_triton.backward()
    grad_x_triton = x.grad.clone()

    x.grad.zero_()

    x_ref = norm_ref(x)
    loss_ref = x_ref.pow(2).sum()
    loss_ref.backward()
    grad_x_ref = x.grad.clone()

    tolerance = 1e-4
    torch.testing.assert_close(x_triton, x_ref, atol=tolerance, rtol=0)
    torch.testing.assert_close(grad_x_triton, grad_x_ref, atol=tolerance, rtol=0)  # dx
