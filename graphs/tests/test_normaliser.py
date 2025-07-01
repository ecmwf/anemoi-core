# (C) Copyright 2024 Anemoi contributors.
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


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-range", "unit-std"])
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


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-range", "unit-std"])
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
