# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import torch


def leaky_hardtanh(
    input: torch.Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    negative_slope: float = 0.01,
    positive_slope: float = 0.01,
) -> torch.Tensor:
    """Leaky version of hardtanh where regions outside [min_val, max_val] have small non-zero slopes.

    Args:
        input: Input tensor
        min_val: Minimum value for the hardtanh region
        max_val: Maximum value for the hardtanh region
        negative_slope: Slope for values below min_val
        positive_slope: Slope for values above max_val

    Returns:
        Tensor with leaky hardtanh applied
    """
    below_min = input < min_val
    above_max = input > max_val
    # Standard hardtanh behavior for the middle region
    result = torch.clamp(input, min_val, max_val)
    # Add leaky behavior for regions outside the clamped range
    result = torch.where(below_min, min_val + negative_slope * (input - min_val), result)
    result = torch.where(above_max, max_val + positive_slope * (input - max_val), result)
    return result
