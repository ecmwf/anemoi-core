# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import random
import secrets
from enum import IntEnum

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


class SeedContext(IntEnum):
    """Contexts in which random seeds are used."""

    TRAINER = 0
    MODEL = 1
    DATALOADER = 2
    TRAINING_BATCH = 3


# Base seed of the current run; see set_base_seed().
_base_seed: int | None = None


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


def choose_base_seed(config_seed: int | None, checkpoint_seed: int | None) -> int:
    """Pick the base seed of a training run.

    A seed set in the config is used as given. Without one, a run that continues
    from a checkpoint takes the checkpoint's seed, so it draws the same data order
    and random numbers as the interrupted run, and a new run draws a fresh seed.

    Parameters
    ----------
    config_seed : int | None
        Seed from ``training.seed`` in the config, or None.
    checkpoint_seed : int | None
        Base seed stored in the checkpoint the run continues from, or None for a new run.

    Returns
    -------
    int
        Base seed of the run.

    """
    if config_seed is None:
        if checkpoint_seed is not None:
            return checkpoint_seed
        return secrets.randbits(32)

    if checkpoint_seed is not None and config_seed != checkpoint_seed:
        LOGGER.warning(
            "The checkpoint was trained with base seed %d, but training.seed is %d. The run does not continue "
            "exactly: the rest of the current epoch repeats some samples and skips others, and every step draws "
            "different random numbers. Set training.seed to null to continue with the checkpoint's seed.",
            checkpoint_seed,
            config_seed,
        )
    return config_seed


def set_base_seed(base_seed: int) -> None:
    """Set the base seed that all random number generators of this run are derived from.

    Parameters
    ----------
    base_seed : int
        Base seed shared by all ranks.

    """
    global _base_seed
    _base_seed = base_seed


def get_base_seed() -> int:
    """Return the base seed set for this run with set_base_seed().

    Returns
    -------
    int
        Base seed shared by all ranks.

    """
    if _base_seed is None:
        msg = "The base seed has not been set. AnemoiTrainer sets it when a training run starts."
        raise RuntimeError(msg)
    return _base_seed


def seed_random_generators(seed: int) -> None:
    """Seed the Python, NumPy and PyTorch (CPU and all GPUs) random number generators of this process.

    Parameters
    ----------
    seed : int
        Seed in the range [0, 2**32 - 1], as returned by derive_seed().

    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
