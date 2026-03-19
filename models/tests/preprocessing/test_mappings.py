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

from anemoi.models.preprocessing.mappings import affine_transform
from anemoi.models.preprocessing.mappings import asinh_converter
from anemoi.models.preprocessing.mappings import atanh_converter
from anemoi.models.preprocessing.mappings import boxcox_converter
from anemoi.models.preprocessing.mappings import displace_boundary_atoms
from anemoi.models.preprocessing.mappings import displace_boundary_atoms_inverse
from anemoi.models.preprocessing.mappings import inverse_affine_transform
from anemoi.models.preprocessing.mappings import inverse_asinh_converter
from anemoi.models.preprocessing.mappings import inverse_atanh_converter
from anemoi.models.preprocessing.mappings import inverse_boxcox_converter
from anemoi.models.preprocessing.mappings import inverse_power_transform
from anemoi.models.preprocessing.mappings import power_transform


@pytest.mark.parametrize("lambd", [0.0, 0.25, 0.75, 1.5])
def test_boxcox_roundtrip(lambd: float) -> None:
    x = torch.tensor([0.05, 0.1, 0.5, 1.0, 2.0, 8.0], dtype=torch.float32)
    y = boxcox_converter(x.clone(), lambd=lambd)
    x_back = inverse_boxcox_converter(y.clone(), lambd=lambd)
    assert torch.allclose(x_back, x, atol=1e-5, rtol=1e-5)


def test_boxcox_clip_negative() -> None:
    x = torch.tensor([-1.0, 0.0, 0.1, 1.0], dtype=torch.float32)
    y = boxcox_converter(x.clone(), lambd=0.5, clip_negative=True)
    assert torch.isfinite(y).all()


def test_boxcox_without_clip_negative_raises() -> None:
    x = torch.tensor([-1.0, 0.1, 1.0], dtype=torch.float32)
    with pytest.raises(AssertionError):
        boxcox_converter(x, lambd=0.5, clip_negative=False)


@pytest.mark.parametrize("lambd", [0.2, 0.5, 1.1, 2.0])
def test_power_roundtrip(lambd: float) -> None:
    x = torch.tensor([0.0, 0.01, 0.1, 1.0, 3.0], dtype=torch.float32)
    y = power_transform(x.clone(), lambd=lambd)
    x_back = inverse_power_transform(y.clone(), lambd=lambd)
    assert torch.allclose(x_back, x, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("lambd", [0.2, 0.5, 1.1, 2.0])
def test_power_roundtrip_tangent_linear_above_one(lambd: float) -> None:
    x = torch.tensor([0.0, 0.01, 0.1, 1.0, 1.5, 3.0], dtype=torch.float32)
    y = power_transform(x.clone(), lambd=lambd, tangent_linear_above_one=True)
    x_back = inverse_power_transform(y.clone(), lambd=lambd, tangent_linear_above_one=True)
    assert torch.allclose(x_back, x, atol=1e-6, rtol=1e-5)


def test_power_negative_input_raises() -> None:
    with pytest.raises(AssertionError):
        power_transform(torch.tensor([-0.1, 0.2], dtype=torch.float32), lambd=0.5)


@pytest.mark.parametrize("rho", [0.25, 1.0, 2.0, 4.0])
def test_atanh_roundtrip(rho: float) -> None:
    x = torch.linspace(0.0, 1.0, steps=21, dtype=torch.float32)
    y = atanh_converter(x.clone(), rho=rho)
    x_back = inverse_atanh_converter(y.clone(), rho=rho)
    assert torch.allclose(x_back, x, atol=1e-5, rtol=1e-5)


def test_atanh_negative_rho_raises() -> None:
    with pytest.raises(ValueError):
        atanh_converter(torch.tensor([0.2, 0.7], dtype=torch.float32), rho=-0.1)


def test_asinh_roundtrip() -> None:
    x = torch.tensor([0.0, 0.01, 0.1, 1.0, 3.0, 100.0], dtype=torch.float32)
    y = asinh_converter(x.clone(), c=1.0)
    x_back = inverse_asinh_converter(y.clone(), c=1.0)
    assert torch.allclose(x_back, x, atol=1e-6, rtol=1e-5)


def test_affine_roundtrip() -> None:
    x = torch.tensor([-2.0, -0.5, 0.0, 1.5], dtype=torch.float32)
    y = affine_transform(x.clone(), scale=2.5, shift=-1.0)
    x_back = inverse_affine_transform(y.clone(), scale=2.5, shift=-1.0)
    assert torch.allclose(x_back, x, atol=1e-7, rtol=1e-7)


def test_affine_scale_zero_raises() -> None:
    with pytest.raises(AssertionError):
        affine_transform(torch.tensor([1.0], dtype=torch.float32), scale=0.0)
    with pytest.raises(AssertionError):
        inverse_affine_transform(torch.tensor([1.0], dtype=torch.float32), scale=0.0)


def test_displace_boundary_atoms_and_inverse() -> None:
    x = torch.tensor([0.0, 1e-5, 0.25, 0.75, 1.0], dtype=torch.float32)
    displaced = displace_boundary_atoms(
        x.clone(),
        lower_atom=0.0,
        lower_target=-1.0,
        upper_atom=1.0,
        upper_target=2.0,
        eps=1e-4,
    )
    expected_displaced = torch.tensor([-1.0, -1.0, 0.25, 0.75, 2.0], dtype=torch.float32)
    assert torch.allclose(displaced, expected_displaced)

    restored = displace_boundary_atoms_inverse(
        displaced.clone(),
        lower_atom=0.0,
        lower_target=-1.0,
        upper_atom=1.0,
        upper_target=2.0,
    )
    expected_restored = torch.tensor([0.0, 0.0, 0.25, 0.75, 1.0], dtype=torch.float32)
    assert torch.allclose(restored, expected_restored)


def test_displace_boundary_atoms_validates_targets() -> None:
    x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    with pytest.raises(AssertionError):
        displace_boundary_atoms(x.clone(), lower_atom=0.0, lower_target=0.1)
    with pytest.raises(AssertionError):
        displace_boundary_atoms(x.clone(), upper_atom=1.0, upper_target=0.9)
