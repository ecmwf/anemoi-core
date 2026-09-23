# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.training.utils.compile import subset_tensor


def test_subset_tensor_none_returns_input() -> None:
    x = torch.arange(24).reshape(2, 3, 4)

    out, subset_index, subset_dim = subset_tensor(x, None)

    assert out is x
    assert subset_index is None
    assert subset_dim is None


def test_subset_tensor_ellipsis_only_returns_input() -> None:
    x = torch.arange(24).reshape(2, 3, 4)

    out, subset_index, subset_dim = subset_tensor(x, (Ellipsis,))

    assert out is x
    assert subset_index is None
    assert subset_dim is None


def test_subset_tensor_tuple_indices_select_first_dimension() -> None:
    x = torch.arange(24).reshape(3, 2, 4)

    out, subset_index, subset_dim = subset_tensor(x, (2, 0))

    expected_index = torch.tensor([2, 0], dtype=torch.long)

    torch.testing.assert_close(out, torch.index_select(x, dim=0, index=expected_index))
    torch.testing.assert_close(subset_index, expected_index)
    assert subset_dim == 0


def test_subset_tensor_list_indices_select_first_dimension() -> None:
    x = torch.arange(24).reshape(3, 2, 4)

    out, subset_index, subset_dim = subset_tensor(x, [2, 0])
    expected_index = torch.tensor([2, 0], dtype=torch.long)

    torch.testing.assert_close(out, torch.index_select(x, dim=0, index=expected_index))
    torch.testing.assert_close(subset_index, expected_index)
    assert subset_dim == 0


def test_subset_tensor_ellipsis_selects_last_dimension() -> None:
    x = torch.arange(24).reshape(2, 3, 4)

    out, subset_index, subset_dim = subset_tensor(x, (Ellipsis, [3, 1]))
    expected_index = torch.tensor([3, 1], dtype=torch.long)

    torch.testing.assert_close(out, torch.index_select(x, dim=-1, index=expected_index))
    torch.testing.assert_close(subset_index, expected_index)
    assert subset_dim == -1


def test_subset_tensor_tensor_indices_are_moved_to_input_device_and_long_dtype() -> None:
    x = torch.arange(24).reshape(2, 3, 4)
    input_index = torch.tensor([3, 0], dtype=torch.int32)

    out, subset_index, subset_dim = subset_tensor(x, (Ellipsis, input_index))
    expected_index = input_index.to(device=x.device, dtype=torch.long)

    torch.testing.assert_close(out, torch.index_select(x, dim=-1, index=expected_index))
    assert subset_index is not None
    assert subset_index.device == x.device
    assert subset_index.dtype == torch.long
    torch.testing.assert_close(subset_index, expected_index)
    assert subset_dim == -1
