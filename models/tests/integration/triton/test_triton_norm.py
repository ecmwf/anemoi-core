# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Tuple

import pytest
import torch

from anemoi.models.triton.utils import edge_index_to_csc
from anemoi.models.triton.utils import is_triton_available

if is_triton_available():
    from anemoi.models.triton.norm import RMSNormFunction


@pytest.fixture(autouse=True)
def setup_torch():
    """Set up torch defaults for all tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    yield


@pytest.mark.slow
@pytest.mark.parametrize(
    "h,d,elementwise_affine",
    [
        (2, 4, False),
        #(2, 6, False),
        #(6, 4, False),
        #(6, 6, False),
        (2, 4, True),
    ],
)
def test_rms_norm_forward(h: int, d: int, elementwise_affine: bool):
    """Test forward pass of RMS Norm."""
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn((h, d))
    x_ref = torch.clone(x)
    
    w=None
    if elementwise_affine:
        w = torch.ones(h, d, device=x.device, dtype=torch.float32)
        w[:] =torch.arange(d, device=x.device) * 0.1 + 1.0  #change weights from [1,]
    x_triton = RMSNormFunction.apply(x, w) 
    
    rms_norm_ref = torch.nn.RMSNorm(d, elementwise_affine=elementwise_affine)
    #change weights from [1,] for correctness testing
    if elementwise_affine:
        with torch.no_grad():
            rms_norm_ref.weight[:] = torch.arange(d, device=x.device) * 0.1 + 1.0
    x_ref = rms_norm_ref(x_ref)
    
    tolerance = 1e-4
    torch.testing.assert_close(x_triton, x_ref, atol=tolerance, rtol=0)

@pytest.mark.slow
@pytest.mark.parametrize(
    "h,d,elementwise_affine",
    [
        (2, 4, False),
        #(2, 6),
        #(6, 4),
        #(6, 6),
        (2, 4, True,)
    ],
)
def test_rms_norm_backward(h: int, d: int, elementwise_affine: bool):
    """Test backward pass of RMS Norm."""
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn((h, d), requires_grad=True)
    #x_ref = torch.clone(x, requires_grad=True)
    #x_ref.retain_grad()
    
    w=None
    if elementwise_affine:
        w = torch.ones(h, d, device=x.device, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            w[:] =torch.arange(d, device=x.device) * 0.1 + 1.0 
    x_triton = RMSNormFunction.apply(x, w) 
    
    loss_triton = x_triton.pow(2).sum()
    loss_triton.backward()
    grad_x_triton = x.grad.clone()
    if elementwise_affine:
        grad_w_triton = w.grad.clone()
        
    x.grad.zero_()
    
    rms_norm_ref = torch.nn.RMSNorm(d,elementwise_affine=elementwise_affine)
    
    #change weights from [1,] for correctness testing
    if elementwise_affine:
        with torch.no_grad():
            rms_norm_ref.weight[:] = torch.arange(d, device=x.device) * 0.1 + 1.0
    x_ref = rms_norm_ref(x)
    loss_ref = x_ref.pow(2).sum()
    loss_ref.backward()
    grad_x_ref = x.grad.clone()
    if elementwise_affine:
        grad_w_ref = rms_norm_ref.weight.grad.clone()
    
    tolerance = 1e-4
    torch.testing.assert_close(x_triton, x_ref, atol=tolerance, rtol=0)
    torch.testing.assert_close(grad_x_triton, grad_x_ref, atol=tolerance, rtol=0) # dx 
    if elementwise_affine:
        torch.testing.assert_close(grad_w_triton, grad_w_ref, atol=tolerance, rtol=0) # dw