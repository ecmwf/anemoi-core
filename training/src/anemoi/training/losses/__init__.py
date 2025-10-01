# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .combined import CombinedLoss
from .kcrps import AlmostFairKernelCRPS
from .kcrps import KernelCRPS
from .logcosh import LogCoshLoss
from .loss import get_loss_function
from .mse import MSELoss
from .torch import TorchLoss
from .weighted_mse import WeightedMSELoss

__all__ = [
    "AlmostFairKernelCRPS",
    "CombinedLoss",
    "KernelCRPS",
    "LogCoshLoss",
    "MSELoss",
    "TorchLoss",
    "WeightedMSELoss",
    "get_loss_function",
]
