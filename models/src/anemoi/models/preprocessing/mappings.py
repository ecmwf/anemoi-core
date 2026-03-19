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


def affine_transform(x, scale=1.0, shift=0.0):
    """Applies a scale and shift to the input tensor."""
    assert scale != 0, "scale must be non-zero for a reversible affine transform"
    return x.mul_(scale).add_(shift)


def inverse_affine_transform(x, scale=1.0, shift=0.0):
    assert scale != 0, "scale must be non-zero for a reversible affine transform"
    return x.sub_(shift).div_(scale)


# --------------------------------------------------------
# Displace boundary atoms
# --------------------------------------------------------
def displace_boundary_atoms(x, lower_atom=None, upper_atom=None, lower_target=None, upper_target=None, eps=0.0):
    """Displaces exact boundary values to target values (outside of the original range) to give model flexibility to model them as imprecise peaks, instead of delta functions. Reverse transform clamps the imprecise predicted values back to the original range to the original boundary values. Can be used on lower bound, upper bound, or both.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    lower_atom : float, optional
        Lower boundary atom
    upper_atom : float, optional
        Upper boundary atom
    lower_target : float, optional
        Target value for lower boundary atom
    upper_target : float, optional
        Target value for upper boundary atom
    eps : float, optional
        Epsilon value around the atoms for numerical stability. Default is 0.0.

    """

    if lower_atom is not None:
        assert lower_target is not None, "To displace lower boundary atom, lower_target must be specified"
        assert lower_target < lower_atom, "lower_target must be less than lower_atom"
        x.masked_fill_(x <= lower_atom + eps, lower_target)
    if upper_atom is not None:
        assert upper_target is not None, "To displace upper boundary atom, upper_target must be specified"
        assert upper_target > upper_atom, "upper_target must be greater than upper_atom"
        x.masked_fill_(x >= upper_atom - eps, upper_target)
    return x


def inverse_displace_boundary_atoms(x, lower_atom=None, upper_atom=None, lower_target=None, upper_target=None):
    """Clamps the values back to the original range, to the original boundary values. Can be used on lower bound, upper bound, or both."""

    return x.clamp(lower_atom, upper_atom)


# --------------------------------------------------------
# boxcox transform
# (generalising powerlaw, linear, and log relationship)
# --------------------------------------------------------
def boxcox_converter(x, lambd=0.5, clip_negative=False):
    """Convert positive var in to boxcox(var) = (x^lambd - 1) / lambd

    Special cases:
    - lambd == 0 -> log(x)
    - lambd == 1 -> x-1

    Notes
    -----
    - Choose lambd < 1 to create a real gap/endpoint basin.
    - If lambd == 1, this reduces to a bounded smooth transform with no gap.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    lambd : float, optional
        Lambda parameter for the boxcox transform. Default is 0.5.
    clip_negative : bool, optional
        Whether to clip negative values to 0. Default is False.
    """

    # Check domain of input
    if lambd == 0:
        assert x.gt(0.0).all(), "input x must be strictly positive for parameter lambd == 0"
    else:
        if clip_negative:
            x = torch.clamp_(x, min=0.0)
        else:
            assert x.ge(
                0.0
            ).all(), (
                "input x must me greater or equal to 0, or use with clip_negative=True to clip negative values to 0"
            )

    # Apply transformation
    if lambd == 0:
        return torch.log_(x)
    return (torch.pow_(x, lambd).sub_(1.0)).div_(lambd)


def inverse_boxcox_converter(x, lambd=0.5, clip_negative=None):
    """Convert back boxcox(var) to var."""
    if lambd == 0:
        return torch.exp_(x)
    return torch.pow_(torch.clamp_(x.mul_(lambd).add_(1.0), min=0.0), 1 / lambd)


# --------------------------------------------------------
# power quantile transform / boxcox rescaled
# --------------------------------------------------------
def power_transform(x, lambd=0.33, clip_negative=False, tangent_linear_above_one=False):
    """Apply a power transform
    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    lambd : float
        Exponent for the power transform. Default is 0.33.
    clip_negative : bool, optional
        Whether to clip negative values to 0. Default is False.
    tangent_linear_above_one : bool, optional
        Whether to use a tangent-linear extension above 1 instead of the power transform. Useful for max-scaled variables where we still might want to predict values above max without clamping them to max and without blowing them up with the power-transform. Default is False.
    """
    assert lambd > 0, f"For power transform, parameter lambd {lambd} must satisfy lambd > 0."

    if clip_negative:
        x = torch.clamp_(x, min=0.0)
    else:
        assert x.ge(
            0.0
        ).all(), (
            "Power transform input x must satisfy x >= 0, or use with clip_negative=True to clip negative values to 0."
        )
    if tangent_linear_above_one:
        lin_branch = x.clone().mul_(lambd).add_(1.0 - lambd)
        pow_branch = torch.pow_(x, lambd)
        return torch.where(x > 1.0, lin_branch, pow_branch)
    return torch.pow_(x, lambd)


def inverse_power_transform(x, lambd=0.33, tangent_linear_above_one=False):
    """Inverse power transform with optional inverse tangent-linear branch above 1."""
    assert lambd > 0, f"For inverse power transform, parameter lambd {lambd} must satisfy lambd > 0."
    if tangent_linear_above_one:
        lin_branch = x.clone().sub_(1.0 - lambd).div_(lambd)
        pow_branch = torch.pow_(torch.clamp_(x, min=0.0), 1 / lambd)
        return torch.where(x > 1.0, lin_branch, pow_branch)
    return torch.pow_(torch.clamp_(x, min=0.0), 1 / lambd)


# --------------------------------------------------------
# atanh transform
# --------------------------------------------------------
def atanh_converter(x, rho=2.0):
    """Encode x in [0, 1] to a single scalar value in [-1, 1]

    Mapping:
        x == 0   -> -1
        0 < x < 1 -> atanh(tanh(rho) * (2x - 1)) / rho
        x == 1   -> +1

        (x == 0.5 -> 0)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    rho : float, optional
        Rho parameter for the atanh transform. Default is 0.9. Controls the steepness of the transform at the boundaries.
    """
    if rho == 0:
        return x
    if not (0 <= rho):
        raise ValueError(f"rho must satisfy 0 < rho , got {rho}")

    return torch.atanh_((x.mul_(2.0).sub_(1.0)).mul_(torch.tanh(rho))).div_(rho)


def inverse_atanh_converter(y, rho=2.0):
    if rho == 0:
        return y.clamp_(0.0, 1.0)
    return torch.clamp_(torch.tanh_(y.mul_(rho)).div_(math.tanh(rho)).add_(1.0).mul_(0.5), min=0.0, max=1.0)


# --------------------------------------------------------
# asinh transform
# --------------------------------------------------------


def asinh_converter(x, c=1.0):
    """Apply an asinh transform"""
    return torch.asinh_(x.mul_(c))


def inverse_asinh_converter(x, c=1.0):
    """Inverse asinh transform"""
    return torch.sinh_(x).div_(c)


# --------------------------------------------------------
# log1p transform
# --------------------------------------------------------
def log1p_converter(x):
    """Convert positive var in to log(1+var)."""
    return torch.log1p(x)


def expm1_converter(x):
    """Convert back log(1+var) to var."""
    return torch.expm1(x)


# --------------------------------------------------------
# sqrt transform
# --------------------------------------------------------
def sqrt_converter(x):
    """Apply a sqrt transform"""
    return power_transform(x, lambd=0.5)


def inverse_sqrt_converter(x):
    """Inverse sqrt transform"""
    return inverse_power_transform(x, lambd=0.5)
