# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""State-dict prefixes refreshed by ``training.update_ds_stats_on_ckpt_load``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

STATE_PROCESSORS = ("model.pre_processors.", "model.post_processors.")
TENDENCY_PROCESSORS = ("model.pre_processors_tendencies.", "model.post_processors_tendencies.")


def _prefixes(setting: Any, processors: tuple[str, ...]) -> tuple[str, ...]:
    """Prefixes for one flag: ``True`` -> all datasets, a list of names -> only those datasets."""
    if setting is None or setting is False:
        return ()
    if setting is True:
        return processors
    if isinstance(setting, str):
        setting = [setting]
    if isinstance(setting, Iterable):
        return tuple(f"{processor}{name}." for processor in processors for name in setting)
    return processors if bool(setting) else ()


def stats_refresh_prefixes(update_cfg: Any) -> tuple[str, ...]:
    """Return the checkpoint state-dict key prefixes whose statistics are rebuilt from the current dataset.

    ``update_cfg.states`` / ``update_cfg.tendencies`` are each ``True`` (every dataset),
    ``False`` (none) or a list of dataset names (only those datasets' processors, e.g. after
    swapping the source of one dataset while keeping the statistics of the others).
    """
    if update_cfg is None:
        return ()
    return _prefixes(getattr(update_cfg, "states", False), STATE_PROCESSORS) + _prefixes(
        getattr(update_cfg, "tendencies", False),
        TENDENCY_PROCESSORS,
    )
