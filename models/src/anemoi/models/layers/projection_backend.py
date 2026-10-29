# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Backends for the sparse projection and the Gaussian smoothing.

``"triton"`` runs our GPU kernels, ``"torch"`` plain PyTorch on any device.
"""

from typing import Literal
from typing import get_args

import torch

ProjectionBackend = Literal["triton", "torch"]
PROJECTION_BACKENDS = get_args(ProjectionBackend)


def accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype: float64 for float64 inputs, float32 otherwise."""
    return torch.float64 if dtype == torch.float64 else torch.float32
