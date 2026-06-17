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

from typing import Any

import torch


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
