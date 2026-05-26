# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

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

    assert get_base_seed() == 42000
