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
    assert scale != 0, "scale must be non-zero for a reversible affine transform"
    return x.mul_(scale).add_(shift)



def inverse_affine_transform(x, scale=1.0, shift=0.0):
    assert scale != 0, "scale must be non-zero for a reversible affine transform"
    return x.sub_(shift).div_(scale)





def displace_boundary_atoms(x,low_atom=None, high_atom=None, low_target=None, high_target=None, eps=0.0):
    """
    Displaces exact boundary values to target values (outside of the original range) to give model flexibility to model them as imprecise peaks, instead of delta functions. Reverse transform clamps the values back to the original range, to the original boundary values. Can be used on lower bound, upper bound, or both.
    """
    
    if low_atom is not None:
        assert low_target is not None, "To displace lower boundary atom, low_target must be specified"
        assert low_target < low_atom, "low_target must be less than low_atom"
        x.masked_fill_(x <= low_atom + eps, low_target)
    if high_atom is not None:
        assert high_target is not None, "To displace upper boundary atom, high_target must be specified"
        assert high_target > high_atom, "high_target must be greater than high_atom"
        x.masked_fill_(x >= high_atom - eps, high_target)
    return x

def displace_boundary_atoms_inverse(x, low_atom=None, high_atom=None, low_target=None, high_target=None):
    """
    Clamps the values back to the original range, to the original boundary values. Can be used on lower bound, upper bound, or both.
    """

    return x.clamp(low_atom, high_atom)




###### boxcox transform (generalising powerlaw, identity and log relationship)
def boxcox_converter(x, lambd=0.5, clip_negative=False):
    """Convert positive var in to boxcox(var)."""

    # Check domain of input
    if lambd == 0:
            assert x.gt(0.0).all(), f"input x must be strictly positive for parameter lambd == 0"
    else:
        if clip_negative:
            x = torch.clamp(x, min=0.0)
        else:
            assert x.ge(0.0).all(), f"input x must me greater or equal to 0, or use with clip_negative=True to clip negative values to 0"

    # Apply transformation
    if lambd == 0:
        return torch.log(x)
    return (torch.pow(x, lambd) - 1) / lambd

def inverse_boxcox_converter(x, lambd=0.5):
    """Convert back boxcox(var) to var."""
    if lambd == 0:
        return torch.exp(x)
    return torch.pow(torch.relu(x * lambd + 1, 1 / lambd))


###### 
# power quantile transform / boxcox rescaled
#

def power_transform(x, lambd=0.33):
    """Applies 
    """
    assert lambd > 0, f"For power transform, parameter lambd {lambd} must satisfy lambd > 0."
    assert x.ge(0.0).all(), f"Power trasnform input x must satisfy x >= 0."
    return torch.pow_(x, lambd)


def inverse_power_transform(x, lambd=0.33):
    """ 
      inverse of power_transform , with a clamping of low values to zero
    """
    assert lambd > 0, f"For inverse power transform, parameter lambd {lambd} must satisfy lambd > 0."
    return torch.pow_(torch.clamp_(x, min=0.0), 1 / lambd)




def atanh_converter(x, rho=0.9, a=1):
    """Encode x in [0, 1] to a single scalar with explicit endpoint buckets.

    Mapping:
        x == 0   -> -1
        0 < x < 1 -> a * atanh(rho * (2x - 1)) / atanh(rho)
        x == 1   -> +1
        
        (x == 0.5 -> 0)

    Notes
    -----
    - Choose a < 1 to create a real gap/endpoint basin.
    - If a == 1, this reduces to a bounded smooth transform with no gap.
    """
    if rho==0:
        return x
    if not (0 < rho < 1):
        raise ValueError(f"rho must satisfy 0 < rho < 1, got {rho}")
    if a <= 0:
        raise ValueError(f"a must be > 0, got {a}")
    
    return a * torch.atanh(rho * (2.0 * x - 1.0)) / math.atanh(rho)



def inverse_atanh_converter(y, rho=0.9, a=1):
    if rho==0:
        return y.clamp_(0.0, 1.0)
    return torch.clamp_(0.5 * (1.0 + torch.tanh((y / a) * math.atanh(rho)) / rho), 0.0, 1.0)


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


