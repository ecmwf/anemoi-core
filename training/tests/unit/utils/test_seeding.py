# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import random

import numpy as np
import pytest
import torch

from anemoi.training.utils import seeding
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import choose_base_seed
from anemoi.training.utils.seeding import derive_seed
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.seeding import seed_random_generators
from anemoi.training.utils.seeding import set_base_seed


def test_choose_base_seed_takes_checkpoint_seed_without_config_seed() -> None:
    assert choose_base_seed(config_seed=None, checkpoint_seed=1234) == 1234


def test_choose_base_seed_draws_new_seed_for_new_run() -> None:
    seeds = {choose_base_seed(config_seed=None, checkpoint_seed=None) for _ in range(5)}

    assert len(seeds) > 1
    assert all(0 <= seed <= 2**32 - 1 for seed in seeds)


@pytest.mark.parametrize("checkpoint_seed", [None, 1234])
def test_choose_base_seed_uses_config_seed(caplog: pytest.LogCaptureFixture, checkpoint_seed: int | None) -> None:
    with caplog.at_level(logging.WARNING):
        assert choose_base_seed(config_seed=1234, checkpoint_seed=checkpoint_seed) == 1234

    assert not caplog.records


def test_choose_base_seed_warns_when_config_seed_differs_from_checkpoint(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert choose_base_seed(config_seed=1234, checkpoint_seed=4321) == 1234

    assert "base seed 4321, but training.seed is 1234" in caplog.text


def test_get_base_seed_returns_set_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seeding, "_base_seed", None)
    set_base_seed(1234)

    assert get_base_seed() == 1234


def test_get_base_seed_fails_before_seed_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seeding, "_base_seed", None)

    with pytest.raises(RuntimeError, match="has not been set"):
        get_base_seed()


@pytest.mark.parametrize(
    ("base_seed", "keys", "expected_seed"),
    [
        (19525198, (SeedContext.TRAINER,), 2478688360),
        (19525198, (SeedContext.MODEL, 0), 1959788892),
        (19525198, (SeedContext.MODEL, 1), 2516803993),
        (19525198, (SeedContext.DATALOADER, 5), 3643427004),
    ],
)
def test_derive_seed_is_reproducible(base_seed: int, keys: tuple[int, ...], expected_seed: int) -> None:
    assert derive_seed(base_seed, *keys) == expected_seed


def test_derive_seed_distinguishes_contexts() -> None:
    base_seed = 19525198
    seeds = {
        derive_seed(base_seed, SeedContext.TRAINER),
        derive_seed(base_seed, SeedContext.MODEL, 0),
        derive_seed(base_seed, SeedContext.DATALOADER, 0),
        derive_seed(base_seed, SeedContext.TRAINING_BATCH, 0),
    }

    assert len(seeds) == 4


@pytest.mark.parametrize(
    ("base_seed", "keys"),
    [
        (0, (SeedContext.TRAINER,)),
        (42, (SeedContext.DATALOADER, 0)),
        (19525198, (SeedContext.MODEL, 219)),
        (19525198, (SeedContext.MODEL, 10_000)),
        (2**63, (SeedContext.DATALOADER, 7)),
    ],
)
def test_derive_seed_within_uint32_bounds(base_seed: int, keys: tuple[int, ...]) -> None:
    assert 0 <= derive_seed(base_seed, *keys) <= 2**32 - 1


def test_seed_random_generators_repeats_all_streams() -> None:
    def draw() -> tuple[float, float, float]:
        return random.random(), float(np.random.random()), float(torch.rand(1))  # noqa: S311, NPY002

    seed_random_generators(1234)
    first = draw()
    seed_random_generators(1234)

    assert draw() == first
