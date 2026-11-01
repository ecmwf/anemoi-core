# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CRPS with antithetic member pairs (noise centring): partner pairs are left out of the spread term."""

import pytest
import torch

from anemoi.training.losses import CRPS
from anemoi.training.losses.kcrps import antithetic_cross_pairs


def _expected_antithetic_crps(preds: torch.Tensor, targets: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mean distance to the target, minus (1 - eps) / 2 times the mean distance over the non-partner pairs."""
    ens_size = preds.shape[-1]
    mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)
    pairs = [(i, j) for i in range(ens_size) for j in range(ens_size) if i != j and i // 2 != j // 2]
    spread = sum(torch.abs(preds[..., i] - preds[..., j]) for i, j in pairs) / len(pairs)
    return mae - 0.5 * (1.0 - (1.0 - alpha) / ens_size) * spread


def test_cross_pairs_of_four_members() -> None:
    cross = antithetic_cross_pairs(4)
    expected = torch.tensor(
        [
            [False, False, True, True],
            [False, False, True, True],
            [True, True, False, False],
            [True, True, False, False],
        ],
    )
    assert torch.equal(cross, expected)


def test_unpaired_last_member_pairs_with_all_others() -> None:
    cross = antithetic_cross_pairs(5)
    assert cross[4, :4].all()
    assert cross[:4, 4].all()
    assert not cross[4, 4]


def test_two_antithetic_members_have_no_spread_pair() -> None:
    with pytest.raises(ValueError, match="no independent pair"):
        antithetic_cross_pairs(2)


def test_antithetic_pairs_off_by_default() -> None:
    assert CRPS().antithetic_pairs is False


@pytest.mark.parametrize("ens_size", [3, 4, 6])
@pytest.mark.parametrize("alpha", [0.0, 0.95, 1.0])
def test_backends_match_expected_formula(ens_size: int, alpha: float) -> None:
    generator = torch.Generator().manual_seed(ens_size)
    preds = torch.randn(2, 2, 3, 4, ens_size, dtype=torch.float64, generator=generator)
    targets = torch.randn(2, 2, 3, 4, dtype=torch.float64, generator=generator)

    expected = _expected_antithetic_crps(preds, targets, alpha)
    for backend in ("naive", "stable"):
        crps = CRPS(alpha=alpha, backend=backend, antithetic_pairs=True)._kernel_crps(preds, targets)
        torch.testing.assert_close(crps, expected)


def test_spread_term_ignores_partner_distances() -> None:
    """Moving members 0 and 1 apart symmetrically changes only their partner distance and their errors."""
    preds = torch.tensor([[0.0, 0.0, 1.0, -1.0]], dtype=torch.float64)
    wider = torch.tensor([[-0.5, 0.5, 1.0, -1.0]], dtype=torch.float64)
    target = torch.tensor([10.0], dtype=torch.float64)  # beyond every member, so the errors sum the same
    loss = CRPS(alpha=1.0, antithetic_pairs=True)
    # Cross distances: |0-1|, |0+1|, twice -> mean 1; then |-0.5-1|, |-0.5+1|, |0.5-1|, |0.5+1| -> mean 1.
    torch.testing.assert_close(loss._kernel_crps(preds, target), loss._kernel_crps(wider, target))


def test_mirrored_members_score_like_independent_ones() -> None:
    """With mirrored partners the all-pairs spread term is biased; leaving the partners out removes the bias."""
    generator = torch.Generator().manual_seed(0)
    n = 200_000
    # (batch, time, variable, grid, member), as _kernel_crps takes them
    targets = torch.randn(1, 1, 1, n, 1, dtype=torch.float64, generator=generator)
    draws = torch.randn(1, 1, 1, n, 4, dtype=torch.float64, generator=generator)
    mirrored = torch.stack([draws[..., 0], -draws[..., 0], draws[..., 1], -draws[..., 1]], dim=-1)

    independent = CRPS(alpha=1.0)._kernel_crps(draws, targets).mean()
    all_pairs = CRPS(alpha=1.0)._kernel_crps(mirrored, targets).mean()
    cross_pairs = CRPS(alpha=1.0, antithetic_pairs=True)._kernel_crps(mirrored, targets).mean()

    # Standard error of each mean is about 0.001.
    assert abs(cross_pairs - independent) < 0.005
    assert all_pairs < independent - 0.03
