# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# check if triton is installed
# If pytorch is installed on CPU then torch is not available
try:
    import triton
    import triton.language as tl
except ImportError:
    raise ValueError(
        "Error. The 'triton' backend was selected for the GraphTransformer but Triton is not installed. To use this backend please install Triton. Otherwise, select a different backend for the GraphTransformer in the models config."
    )


@triton.jit
def _rms_norm_fwd(x, C):
    """Normalises x in-place along the head dimension
    inputs:
        x: (H_pad,C_pad)
    returns:
        x_norm: (H_pad,C_pad)
    """
    eps: tl.constexpr = 0.0  # small numerical stability factor
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)
    x = x * inv_rms
    return x


@triton.jit
def _rms_norm_bwd(x, grad_out, C):
    """computes grad_x given x and grad_out
    x is the original unnormalised input

    inputs:
        x: (H_pad,C_pad)
        grad_out: (H_pad,C_pad)
        C: int
    returns:
        grad_x: (H_pad,C_pad)
    """
    eps: tl.constexpr = 0.0  # small numerical stability factor

    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1, keep_dims=True) / C + eps)

    # first term
    grad_x = grad_out * inv_rms
    # second term
    dot = tl.sum(grad_out * x, axis=1, keep_dims=True)
    grad_x = grad_x - (x * inv_rms * inv_rms * inv_rms) * (dot / C)

    return grad_x
