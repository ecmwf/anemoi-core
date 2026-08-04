# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed
from anemoi.training.utils.seeding import get_base_seed


@pytest.mark.parametrize("env_var", ["ANEMOI_BASE_SEED", "SLURM_JOB_ID", "CUSTOM_BASE_SEED"])
def test_get_base_seed_from_environment(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    monkeypatch.delenv("ANEMOI_BASE_SEED", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("CUSTOM_BASE_SEED", raising=False)
    monkeypatch.setenv(env_var, "1234")

    base_seed_env = "CUSTOM_BASE_SEED" if env_var == "CUSTOM_BASE_SEED" else None

    assert get_base_seed(base_seed_env=base_seed_env) == 1234


def test_get_base_seed_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANEMOI_BASE_SEED", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    assert get_base_seed() == 42


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
    }

    assert len(seeds) == 3


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
