# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.training.losses.utils import ScaleTensor


def test_scale_contains() -> None:
    scale = ScaleTensor(test=(0, 2))
    assert "test" in scale


def test_scale_contains_indexing() -> None:
    scale = ScaleTensor(test=(0, 2))
    assert 0 in scale


def test_scale_tuple_contains_indexing() -> None:
    scale = ScaleTensor(test=((0, 1), 2))
    assert (0, 1) in scale


def test_scale_tuple_not_contains_indexing() -> None:
    scale = ScaleTensor(test=(0, 2))
    assert (0, 1) not in scale


def test_scale_contains_subset_indexing() -> None:
    scale = ScaleTensor(test=(0, 2), wow=(0, 2))
    assert "test" in scale
    scale = scale.subset("wow")
    assert "wow" in scale
    assert "test" not in scale


def test_scale_contains_subset_by_dim_indexing() -> None:
    scale = ScaleTensor(test=(0, 2), wow=(1, 2))
    assert "test" in scale
    scale = scale.subset_by_dim(1)
    assert "wow" in scale
    assert "test" not in scale


@pytest.mark.parametrize(
    ("scalars", "input_tensor", "output"),
    [
        ([[0, torch.Tensor([2])]], torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 6.0])),
        ([[0, torch.Tensor([0.5])]], torch.tensor([10.0, 20.0, 30.0]), torch.tensor([5.0, 10.0, 15.0])),
        ([[-1, torch.Tensor([0.5])]], torch.tensor([10.0, 20.0, 30.0]), torch.tensor([5.0, 10.0, 15.0])),
        ([[0, torch.Tensor([0])]], torch.tensor([5.0, 10.0, 15.0]), torch.tensor([0.0, 0.0, 0.0])),
        (
            [[0, torch.Tensor([0.5])], [-1, torch.Tensor([3])]],
            torch.tensor([10.0, 20.0, 30.0]),
            torch.tensor([15.0, 30.0, 45.0]),
        ),
        (
            [[0, torch.Tensor([0.5])], [0, torch.Tensor([3])]],
            torch.tensor([10.0, 20.0, 30.0]),
            torch.tensor([15.0, 30.0, 45.0]),
        ),
    ],
)
def test_scale_tensor_one_dim(
    scalars: list[list[int, torch.Tensor]],
    input_tensor: torch.Tensor,
    output: torch.Tensor,
) -> None:

    scale = ScaleTensor()
    for scalar in scalars:
        scale.add_scalar(*scalar)

    torch.testing.assert_close(scale.scale(input_tensor), output)


def test_invalid_dim_sizes() -> None:
    scalar = ScaleTensor()
    scalar.add_scalar(0, torch.ones((5,)))

    with pytest.raises(ValueError, match=r"Validating tensor 'invalid' raised an error."):
        scalar.add_scalar(0, torch.ones((15,)), name="invalid")


def test_invalid_dim_sizes_negative_indexing() -> None:
    scalar = ScaleTensor()
    scalar.add_scalar(0, torch.ones((5,)))
    scalar.add_scalar(-1, torch.ones((15,)), name="invalid")

    with pytest.raises(ValueError, match=r"Validating tensor 'invalid' raised an error."):
        scalar.resolve(1)


def test_valid_dim_sizes_negative_indexing() -> None:
    scalar = ScaleTensor()
    scalar.add_scalar(0, torch.ones((5,)))
    scalar.add_scalar(-1, torch.ones((15,)))

    scalar.resolve(2)


@pytest.mark.parametrize(
    ("scalars", "input_tensor", "output"),
    [
        ([[0, 2.0]], torch.ones([4, 6]), torch.ones([4, 6]) * 2),
        ([[0, [[1.0, 1.0], [1.0, 2.0]]]], torch.ones((2, 2)), [[1, 1], [1, 2]]),
        ([[(0, 1), [[1.0, 1.0], [1.0, 2.0]]]], torch.ones((2, 2)), [[1, 1], [1, 2]]),
        ([[(1, 0), [[1.0, 1.0], [1.0, 2.0]]]], torch.ones((2, 2)), [[1, 1], [1, 2]]),
        ([[(0, 1), [[1.0, 2.0], [1.0, 1.0]]]], torch.ones((2, 2)), [[1, 2], [1, 1]]),
        ([[(1, 0), [[1.0, 2.0], [1.0, 1.0]]]], torch.ones((2, 2)), [[1, 1], [2, 1]]),
    ],
)
def test_scale_tensor_two_dim(
    scalars: list[list[int, torch.Tensor]],
    input_tensor: torch.Tensor,
    output: torch.Tensor,
) -> None:

    scale = ScaleTensor()
    for scalar in scalars:
        scale.add_scalar(*scalar)

    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output, dtype=torch.float32)

    torch.testing.assert_close(scale.scale(input_tensor), output)
