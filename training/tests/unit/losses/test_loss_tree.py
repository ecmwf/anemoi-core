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

from anemoi.training.losses.loss_tree import LossTree
from anemoi.training.losses.loss_tree import loss_components
from anemoi.training.losses.loss_tree import loss_per_variable
from anemoi.training.losses.loss_tree import sum_loss
from anemoi.training.losses.loss_tree import sum_loss_per_variable


def test_loss_tree_requires_value_or_children() -> None:
    with pytest.raises(ValueError, match="must contain a value or at least one child"):
        LossTree(name="empty")

    with pytest.raises(ValueError, match="cannot contain both a value and children"):
        LossTree(
            name="ambiguous",
            value=torch.tensor(1.0),
            children=(LossTree(name="child", value=torch.tensor(2.0)),),
        )


def test_sum_loss_accepts_plain_tensors() -> None:
    loss = torch.tensor([1.0, 2.0, 3.0])

    torch.testing.assert_close(sum_loss(loss), torch.tensor(6.0))


def test_sum_loss_reduces_weighted_tree_once() -> None:
    scalar_score = torch.tensor(3.0, requires_grad=True)
    multiscale_scores = torch.tensor([4.0, 8.0], requires_grad=True)
    loss = LossTree(
        name="combined",
        weight=2.0,
        children=(
            LossTree(name="scalar", weight=0.5, value=scalar_score),
            LossTree(name="multiscale", weight=0.25, value=multiscale_scores),
        ),
    )

    components = loss_components(loss)
    total = sum_loss(loss)

    assert set(components) == {"scalar", "multiscale"}
    torch.testing.assert_close(components["scalar"], torch.tensor(3.0))
    torch.testing.assert_close(components["multiscale"], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(total, torch.tensor(9.0))

    total.backward()
    torch.testing.assert_close(scalar_score.grad, torch.tensor(1.0))
    torch.testing.assert_close(multiscale_scores.grad, torch.tensor([0.5, 0.5]))


def test_sum_loss_per_variable_only_reduces_leading_dimensions() -> None:
    direct_scores = torch.tensor([1.0, 2.0])
    multiscale_scores = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    loss = LossTree(
        name="combined",
        children=(
            LossTree(name="direct", weight=2.0, value=direct_scores),
            LossTree(name="multiscale", weight=0.5, value=multiscale_scores),
        ),
    )

    result = sum_loss_per_variable(loss, num_variables=2)

    torch.testing.assert_close(result, torch.tensor([6.0, 9.0]))


def test_loss_per_variable_keeps_named_components_separate() -> None:
    loss = LossTree(
        name="combined",
        children=(
            LossTree(name="direct", weight=2.0, value=torch.tensor([1.0, 2.0])),
            LossTree(
                name="multiscale",
                weight=0.5,
                value=torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
            ),
        ),
    )

    components = loss_per_variable(loss)

    assert set(components) == {"direct", "multiscale"}
    torch.testing.assert_close(components["direct"], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(components["multiscale"], torch.tensor([4.0, 5.0]))


def test_sum_loss_per_variable_ignores_scalar_components(caplog: pytest.LogCaptureFixture) -> None:
    loss = LossTree(
        name="combined",
        children=(
            LossTree(name="global_scalar_score", value=torch.tensor(10.0)),
            LossTree(name="per_variable", value=torch.tensor([1.0, 2.0, 3.0])),
        ),
    )

    result = sum_loss_per_variable(loss, num_variables=3)

    torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0]))
    assert "global_scalar_score" in caplog.text


def test_sum_loss_per_variable_accepts_scalar_for_single_variable() -> None:
    result = sum_loss_per_variable(torch.tensor(2.0), num_variables=1)

    torch.testing.assert_close(result, torch.tensor([2.0]))


def test_sum_loss_per_variable_returns_none_without_per_variable_values() -> None:
    result = sum_loss_per_variable(torch.tensor(2.0), num_variables=3)

    assert result is None
