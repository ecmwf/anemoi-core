# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import torch


def shard_sizes_from_pattern(shard_size_pattern: tuple[int, ...], world_size: int) -> list[int]:
    """Repeat a shard-size pattern until there is one size per rank."""
    return [shard_size_pattern[rank % len(shard_size_pattern)] for rank in range(world_size)]


def torch_version_less_than(major: int, minor: int) -> bool:
    """Return whether the installed torch version is older than ``major.minor``."""
    version_parts = torch.__version__.split("+", maxsplit=1)[0].split(".")
    return (int(version_parts[0]), int(version_parts[1])) < (major, minor)
