# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import torch


def noop(x):
    """No operation."""
    return x


# box-cox transform (generalising powerlaw, identity and log relationship)
def boxcox_converter(x, lambd=0.5):
    """Convert positive var into boxcox(var)."""
    if lambd == 0:
        return torch.log(x)
    return (torch.pow(x, lambd) - 1) / lambd


def inverse_boxcox_converter(x, lambd=0.5):
    """Convert back boxcox(var) to var."""
    if lambd == 0:
        return torch.exp(x)
    return torch.pow(torch.relu(x * lambd + 1, 1 / lambd))


# atanh with atomic boundary clamping
def _check_atanh_params(k, rho, a):
    if not (0 < rho < 1):
        raise ValueError(f"rho must satisfy 0 < rho < 1, got {rho}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if a <= 0:
        raise ValueError(f"a must be > 0, got {a}")
    if a > k:
        raise ValueError(f"a must satisfy a <= k so the interior fits inside the endpoint buckets, got a={a}, k={k}")


def atanh_converter(x, rho=0.9, a=0.75, k=1.0):
    """Encode x in [0, 1] to a single scalar with explicit endpoint buckets.

    Mapping:
        x == 0   -> -k
        0 < x < 1 -> a * atanh(rho * (2x - 1)) / atanh(rho)
        x == 1   -> +k

    Notes
    -----
    - Choose a < k to create a real gap/endpoint basin.
    - If a == k, this reduces to a bounded smooth transform with no gap.
    """
    denominator = math.atanh(rho)
    interior = a * torch.atanh(rho * (2.0 * x - 1.0)) / denominator
    return torch.where(x <= 0.0, -k, torch.where(x >= 1.0, k, interior))


def inverse_atanh_converter(y, rho=0.9, a=0.75):
    interior = 0.5 * (1.0 + torch.tanh((y / a) * math.atanh(rho)) / rho)
    x = torch.where(y <= -a, torch.zeros_like(y), torch.where(y >= a, torch.ones_like(y), interior))
    return torch.clamp(x, 0.0, 1.0)


# sqrt/ sqr
def sqrt_converter(x):
    """Convert positive var in to sqrt(var)."""
    return torch.sqrt(x)


def square_converter(x):
    """Convert back sqrt(var) to var."""
    return x**2


## log1p and back
def log1p_converter(x):
    """Convert positive var in to log(1+var)."""
    return torch.log1p(x)


def expm1_converter(x):
    """Convert back log(1+var) to var."""
    return torch.expm1(x)
