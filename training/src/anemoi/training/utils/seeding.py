# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import os
from enum import IntEnum

import numpy as np


class SeedContext(IntEnum):
    """Contexts in which random seeds are used."""

    TRAINER = 0
    MODEL = 1
    DATALOADER = 2


def derive_seed(base_seed: int, context: SeedContext, *keys: int) -> int:
    """Build a seed accepted by Lightning, NumPy, and PyTorch.

    Parameters
    ----------
    base_seed : int
        Base seed shared by all ranks.
    context : SeedContext
        Context in which the seed is used.
    *keys : int
        Additional keys used to derive independent seeds, by default none.

    Returns
    -------
    int
        Unsigned 32-bit seed in the range [0, 2**32 - 1].
        Returned as a Python integer for compatibility with random.seed().

    """
    seed_seq = np.random.SeedSequence(entropy=base_seed, spawn_key=(context, *keys))
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def get_base_seed(base_seed_env: str | None = None) -> int:
    """Gets the base seed from the environment variables.

    Option to manually set a seed via export ANEMOI_BASE_SEED=xxx in job script
    If no supported environment variable is set, falls back to 42.

    Parameters
    ----------
    base_seed_env : str, optional
        Environment variable to use for the base seed, by default None

    Returns
    -------
    int
        Base seed.

    """
    env_var_list = ["ANEMOI_BASE_SEED", "SLURM_JOB_ID"]
    if base_seed_env is not None:
        env_var_list = [base_seed_env, *env_var_list]

    base_seed = None
    for env_var in env_var_list:
        if env_var in os.environ:
            base_seed = int(os.environ.get(env_var))
            break

    if base_seed is None:
        base_seed = 42

    return base_seed
