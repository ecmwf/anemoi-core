# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from anemoi.training.losses.graph_score_base import csr_matmul
from anemoi.training.losses.graph_score_base import safe_sqrt


def test_csr_matmul_matches_dense_for_arbitrary_leading_dimensions() -> None:
    dense_matrix = torch.tensor(
        [
            [0.25, 0.75, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.6],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    matrix = dense_matrix.to_sparse_csr()
    values = torch.randn(2, 3, 5, 4, 7, dtype=torch.float64, requires_grad=True)
    reference_values = values.detach().clone().requires_grad_()

    actual = csr_matmul(matrix, values)
    expected = torch.einsum("ij,...jv->...iv", dense_matrix, reference_values)
    actual.square().sum().backward()
    expected.square().sum().backward()

    torch.testing.assert_close(actual, expected)
    assert values.grad is not None
    assert reference_values.grad is not None
    torch.testing.assert_close(values.grad, reference_values.grad)


def test_safe_sqrt_has_a_finite_zero_derivative() -> None:
    squared_norm = torch.tensor([0.0, 4.0], requires_grad=True)

    result = safe_sqrt(squared_norm)
    result.sum().backward()

    torch.testing.assert_close(result, torch.tensor([0.0, 2.0]))
    assert squared_norm.grad is not None
    torch.testing.assert_close(squared_norm.grad, torch.tensor([0.0, 0.25]))
