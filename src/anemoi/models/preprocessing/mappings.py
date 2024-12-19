# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch


def noop(x):
    """No operation."""
    return x


def cos_converter(x):
    """Convert angle in degree to cos."""
    return torch.cos(x / 180 * torch.pi)


def sin_converter(x):
    """Convert angle in degree to sin."""
    return torch.sin(x / 180 * torch.pi)


def atan2_converter(x):
    """Convert cos and sin to angle in degree.

    Input:
    x[..., 0]: cos
    x[..., 1]: sin
    """
    return torch.remainder(torch.atan2(x[..., 1], x[..., 0]) * 180 / torch.pi, 360)


def log1p_converter(x):
    """Convert positive var in to log(1+var)."""
    return torch.log1p(x)


def boxcox_converter(x, lambd=0.5):
    """Convert positive var in to boxcox(var)."""
    pos_lam = (torch.pow(x, lambd) - 1) / lambd
    null_lam = torch.log(x)
    if lambd == 0:
        return null_lam
    else:
        return pos_lam


def sqrt_converter(x):
    """Convert positive var in to sqrt(var)."""
    return torch.sqrt(x)


def expm1_converter(x):
    """Convert back log(1+var) to var."""
    return torch.expm1(x)


def square_converter(x):
    """Convert back sqrt(var) to var."""
    return x**2


def inverse_boxcox_converter(x, lambd=0.5):
    """Convert back boxcox(var) to var."""
    pos_lam = torch.pow(x * lambd + 1, 1 / lambd)
    null_lam = torch.exp(x)
    if lambd == 0:
        return null_lam
    else:
        return pos_lam
