# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .combined import CombinedLoss
from .energy import EnergyScore
from .energy import GroupedEnergyScore
from .huber import HuberLoss
from .kcrps import AlmostFairKernelCRPS
from .kcrps import KernelCRPS
from .logcosh import LogCoshLoss
from .loss import get_loss_function
from .mae import MAELoss
from .mse import MSELoss
from .multivariate_kcrps import GroupedMultivariateKernelCRPS
from .multivariate_kcrps import MultivariateKernelCRPS
from .rmse import RMSELoss
from .variogram import VariogramScore
from .weighted_mse import WeightedMSELoss

__all__ = [
    "AlmostFairKernelCRPS",
    "CombinedLoss",
    "EnergyScore",
    "GroupedEnergyScore",
    "GroupedMultivariateKernelCRPS",
    "HuberLoss",
    "KernelCRPS",
    "LogCoshLoss",
    "MAELoss",
    "MSELoss",
    "MultivariateKernelCRPS",
    "RMSELoss",
    "VariogramScore",
    "WeightedMSELoss",
    "get_loss_function",
]
