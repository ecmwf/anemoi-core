# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.training.utils.masks import Boolean1DMask


def test_apply_boolean1dmask() -> None:
    """Test Boolean1DMask(mask).apply()."""
    n=10
    m=4
    mask = torch.tensor([i < m for i in range(n)])

    x = torch.tensor([-1 for i in range(n)], dtype=torch.float32)
    a = torch.rand(2, 3)
    b = torch.rand(3, 2)
    x = torch.tensordot(a, x, dims=0)
    x = torch.tensordot(x, b, dims=0)

    # test case where fill_value is torch.Tensor
    fill_value = torch.arange(n, dtype=torch.float32)
    c = torch.rand(2, 3)
    d = torch.rand(3, 2)
    fill_value = torch.tensordot(c, fill_value, dims=0)
    fill_value = torch.tensordot(fill_value, d, dims=0)
    y = Boolean1DMask(mask).apply(x, dim=2, fill_value=fill_value, dim_sel=2)
    expected_y0 = torch.tensor([-1 for i in range(m)], dtype=torch.float32)
    expected_y0 = torch.tensordot(a, expected_y0, dims=0)
    expected_y0 = torch.tensordot(expected_y0, b, dims=0)
    expected_y1 = torch.arange(start=m, end=n, dtype=torch.float32)
    expected_y1 = torch.tensordot(c, expected_y1, dims=0)
    expected_y1 = torch.tensordot(expected_y1, d, dims=0)
    expected_y = torch.cat((expected_y0, expected_y1), dim=2)
    assert torch.equal(y, expected_y)

    # test case where fill_value is float
    fill_value = torch.rand(1).item()
    y = Boolean1DMask(mask).apply(x, dim=2, fill_value=fill_value)
    expected_y0 = torch.tensor([-1 for i in range(m)], dtype=torch.float32)
    expected_y0 = torch.tensordot(a, expected_y0, dims=0)
    expected_y0 = torch.tensordot(expected_y0, b, dims=0)
    expected_y1 = torch.tensor([fill_value for i in range(m, n)], dtype=torch.float32)
    c = torch.full((2, 3), 1, dtype=torch.float32)
    d = torch.full((3, 2), 1, dtype=torch.float32)
    expected_y1 = torch.tensordot(c, expected_y1, dims=0)
    expected_y1 = torch.tensordot(expected_y1, d, dims=0)
    expected_y = torch.cat((expected_y0, expected_y1), dim=2)
    assert torch.equal(y, expected_y)
