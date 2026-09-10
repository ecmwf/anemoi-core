# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch

from anemoi.models.transport.data_helpers import zip_map_batch_scalar_data

if TYPE_CHECKING:
    from anemoi.models.transport.data_helpers import Data


def test_zip_map_batch_scalar_preserves_sample_and_member_alignment() -> None:
    left = [torch.ones(2, 2, 1), torch.full((2, 3, 1), 4.0)]
    right = [torch.full((2, 2, 1), 2.0), torch.full((2, 3, 1), 10.0)]
    scalar = torch.tensor([2.0, 3.0, 5.0, 7.0]).reshape(2, 1, 2, 1, 1)

    result = zip_map_batch_scalar_data(left, right, scalar=scalar, fn=lambda a, b, c: a + b * c)

    torch.testing.assert_close(result[0], torch.tensor([[[5.0], [5.0]], [[7.0], [7.0]]]))
    torch.testing.assert_close(result[1], torch.tensor([[[54.0], [54.0], [54.0]], [[74.0], [74.0], [74.0]]]))


@pytest.mark.parametrize(
    ("data", "other", "batch_size", "error", "message"),
    [
        ([torch.zeros(1, 1)], torch.zeros(1, 1), 1, TypeError, "Cannot combine dense and sparse"),
        ([torch.zeros(1, 1)], [torch.zeros(1, 1), torch.zeros(1, 1)], 1, ValueError, "same length"),
        ([], [torch.zeros(1, 1)], 0, ValueError, "same length"),
        ([torch.zeros(1, 1)], [torch.zeros(1, 1)], 2, ValueError, "Condition batch size"),
    ],
)
def test_zip_map_batch_scalar_validates_structure_before_computing(
    data: Data, other: Data, batch_size: int, error: type[Exception], message: str
) -> None:
    operation = Mock()
    with pytest.raises(error, match=message):
        zip_map_batch_scalar_data(data, other, scalar=torch.ones(batch_size, 1, 1, 1, 1), fn=operation)
    operation.assert_not_called()
