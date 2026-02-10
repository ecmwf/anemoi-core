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
    from anemoi.models.triton.norm import RMSNorm


@pytest.fixture(autouse=True)
def setup_torch():
    """Set up torch defaults for all tests."""
    device = "cuda"
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    yield


@pytest.mark.slow
@pytest.mark.parametrize(
    "b,h,d,elementwise_affine",
    [
        (B,H, C, affine)
        for B in (1, 4)
        for H in (2, 6)
        for C in (4, 6)
        for affine in (False, True)
    ],
)
def test_rms_norm_forward(b: int, h: int, d: int, elementwise_affine: bool):
    """Test forward pass of RMS Norm."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn((b, h, d))

    rms_norm = RMSNorm(d, elementwise_affine=elementwise_affine).cuda()
    rms_norm_ref = torch.nn.RMSNorm(d, elementwise_affine=elementwise_affine)

    # change weights from inital value of [1] for correctness testing
    if elementwise_affine:
        with torch.no_grad():
            rms_norm.weight[:] = torch.arange(d, device=x.device) * 0.1 + 1.0
            rms_norm_ref.weight[:] = torch.arange(d, device=x.device) * 0.1 + 1.0

    x_triton = rms_norm(x)

    x_ref = rms_norm_ref(x)

    tolerance = 1e-4
    torch.testing.assert_close(x_triton, x_ref, atol=tolerance, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "b, h,d,elementwise_affine",
    [
        (B, H, C, affine)
        for B in (1, 4)
        for H in (2, 6)
        for C in (4, 6)
        for affine in (False, True)
    ],
)
def test_rms_norm_backward(b: int, h: int, d: int, elementwise_affine: bool):
    """Test backward pass of RMS Norm."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn((b, h, d), requires_grad=True)

    rms_norm = RMSNorm(d, elementwise_affine=elementwise_affine).cuda()
    rms_norm_ref = torch.nn.RMSNorm(d, elementwise_affine=elementwise_affine)

    # change weights from inital value of [1] for correctness testing
    if elementwise_affine:
        with torch.no_grad():
            rms_norm.weight[:] = torch.arange(d, device=x.device) * 0.1 + 1.0
            rms_norm_ref.weight[:] = torch.arange(d, device=x.device) * 0.1 + 1.0

    x_triton = rms_norm(x)
    loss_triton = x_triton.pow(2).sum()
    loss_triton.backward()
    grad_x_triton = x.grad.clone()
    
    if elementwise_affine:
        grad_w_triton = rms_norm.weight.grad.clone()

    x.grad.zero_()

    x_ref = rms_norm_ref(x)
    loss_ref = x_ref.pow(2).sum()
    loss_ref.backward()
    grad_x_ref = x.grad.clone()
    
    if elementwise_affine:
        grad_w_ref = rms_norm_ref.weight.grad.clone()

    tolerance = 1e-4
    torch.testing.assert_close(x_triton, x_ref, atol=tolerance, rtol=0)
    torch.testing.assert_close(grad_x_triton, grad_x_ref, atol=tolerance, rtol=0)  # dx
    if elementwise_affine:
        torch.testing.assert_close(grad_w_triton, grad_w_ref, atol=tolerance, rtol=0)  # dw
