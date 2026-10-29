# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The kernels that the sparse projection and the Gaussian smoothing can run on.

``"triton"`` runs GPU kernels; ``"torch"`` runs plain PyTorch operations on any device. Both
backends keep their weighted sums in the same precision, given by :func:`accumulation_dtype`.
"""

from typing import Literal
from typing import get_args

import torch

ProjectionBackend = Literal["triton", "torch"]
PROJECTION_BACKENDS = get_args(ProjectionBackend)


def accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """The dtype the weighted sums are kept in: float64 for float64 fields, float32 for all others."""
    return torch.float64 if dtype == torch.float64 else torch.float32
