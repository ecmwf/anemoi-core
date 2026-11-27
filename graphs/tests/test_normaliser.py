# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.graphs.normalise import NormaliserMixin


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-range", "unit-std", "log1p"])
def test_normaliser(norm: str):
    """Test NormaliserMixin normalise method."""

    class Normaliser(NormaliserMixin):
        def __init__(self, norm):
            self.norm = norm
            self.norm_by_group = False

        def __call__(self, data):
            return self.normalise(data)

    normaliser = Normaliser(norm=norm)
    data = torch.rand(10, 5)
    normalised_data = normaliser(data)
    assert isinstance(normalised_data, torch.Tensor)
    assert normalised_data.shape == data.shape


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-range", "unit-std", "log1p"])
def test_grouped_normaliser(norm: str):
    """Test NormaliserMixin normalise method."""

    class Normaliser(NormaliserMixin):
        def __init__(self, norm):
            self.norm = norm
            self.norm_by_group = True

        def __call__(self, data, index, num_groups):
            return self.normalise(data, index, num_groups)

    normaliser = Normaliser(norm=norm)
    data = torch.rand(10, 5)
    index = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    num_groups = 5
    normalised_data = normaliser(data, index, num_groups)
    assert isinstance(normalised_data, torch.Tensor)
    assert normalised_data.shape == data.shape


@pytest.mark.parametrize("norm_by_group", [False, True])
def test_unit_std_single_value_is_finite(norm_by_group: bool):
    """The sample std. dev. of a single value is NaN; unit-std must skip normalisation instead."""

    class Normaliser(NormaliserMixin):
        def __init__(self):
            self.norm = "unit-std"
            self.norm_by_group = norm_by_group

    normaliser = Normaliser()
    if norm_by_group:
        # group 1 holds a single value
        data = torch.tensor([[0.2], [0.5], [0.7]])
        normalised_data = normaliser.normalise(data, torch.tensor([0, 0, 1]), 2)
    else:
        data = torch.tensor([[0.7]])
        normalised_data = normaliser.normalise(data)

    assert torch.isfinite(normalised_data).all()


@pytest.mark.parametrize("norm", ["l3", "invalid"])
def test_normaliser_wrong_norm(norm: str):
    """Test NormaliserMixin normalise method."""

    class Normaliser(NormaliserMixin):
        def __init__(self, norm: str):
            self.norm = norm
            self.norm_by_group = False

        def __call__(self, data):
            return self.normalise(data)

    with pytest.raises(AssertionError):
        normaliser = Normaliser(norm=norm)
        data = torch.rand(10, 5)
        normaliser(data)


def test_normaliser_wrong_inheritance():
    """Test NormaliserMixin normalise method."""

    class Normaliser(NormaliserMixin):
        def __init__(self, attr):
            self.attr = attr

        def __call__(self, data):
            return self.normalise(data)

    with pytest.raises(AttributeError):
        normaliser = Normaliser(attr="attr_name")
        data = torch.rand(10, 5)
        normaliser(data)
