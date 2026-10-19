# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""State dict helpers for the loading strategies."""

from __future__ import annotations

from typing import Any

import torch


def shape_mismatches(target: dict[str, Any], source: dict[str, Any]) -> dict[str, str]:
    """Return the keys in ``source`` and ``target`` whose tensor shapes differ.

    Parameters
    ----------
    target : dict
        Model state dict.
    source : dict
        Checkpoint state dict.

    Returns
    -------
    dict[str, str]
        Key to reason (``"Shape mismatch: <source> vs <target>"``).
    """
    return {
        key: f"Shape mismatch: {value.shape} vs {target[key].shape}"
        for key, value in source.items()
        if isinstance(value, torch.Tensor)
        and isinstance(target.get(key), torch.Tensor)
        and value.shape != target[key].shape
    }
