# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pure utility functions for state dict manipulation.

These are stateless helpers used by loading strategies. No model loading,
no async, no pipeline wiring — just pure functions on state dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch


@dataclass(frozen=True)
class MatchResult:
    """Result of comparing source and target state dict keys.

    Attributes
    ----------
    missing_in_source : set[str]
        Keys present in target but not in source
    unexpected_in_source : set[str]
        Keys present in source but not in target
    shape_mismatches : set[str]
        Keys present in both but with different tensor shapes
    """

    missing_in_source: frozenset[str] = field(default_factory=frozenset)
    unexpected_in_source: frozenset[str] = field(default_factory=frozenset)
    shape_mismatches: frozenset[str] = field(default_factory=frozenset)


def filter_state_dict(
    source: dict[str, Any],
    target: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Filter source state dict to only include keys compatible with target.

    Non-mutating: builds new dicts, never modifies the inputs.

    Parameters
    ----------
    source : dict
        Source state dictionary (e.g. from checkpoint)
    target : dict
        Target state dictionary (e.g. from model)

    Returns
    -------
    tuple[dict, dict]
        (filtered, skipped) where filtered contains compatible entries
        and skipped maps key to a reason string for incompatible entries
    """
    filtered: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    for key, value in source.items():
        if key not in target:
            skipped[key] = "Key not in target"
            continue

        if (
            isinstance(value, torch.Tensor)
            and isinstance(target[key], torch.Tensor)
            and value.shape != target[key].shape
        ):
            skipped[key] = f"Shape mismatch: {value.shape} vs {target[key].shape}"
            continue

        filtered[key] = value

    return filtered, skipped


def match_state_dict_keys(
    source_dict: dict[str, Any],
    target_dict: dict[str, Any],
) -> MatchResult:
    """Compare keys and shapes between source and target state dicts.

    Parameters
    ----------
    source_dict : dict
        Source state dictionary
    target_dict : dict
        Target state dictionary

    Returns
    -------
    MatchResult
        Comparison result with missing, unexpected, and mismatched keys
    """
    source_keys = set(source_dict.keys())
    target_keys = set(target_dict.keys())

    missing_in_source = target_keys - source_keys
    unexpected_in_source = source_keys - target_keys

    shape_mismatches: set[str] = set()
    for key in source_keys & target_keys:
        src_val = source_dict[key]
        tgt_val = target_dict[key]
        if isinstance(src_val, torch.Tensor) and isinstance(tgt_val, torch.Tensor) and src_val.shape != tgt_val.shape:
            shape_mismatches.add(key)

    return MatchResult(
        missing_in_source=frozenset(missing_in_source),
        unexpected_in_source=frozenset(unexpected_in_source),
        shape_mismatches=frozenset(shape_mismatches),
    )
