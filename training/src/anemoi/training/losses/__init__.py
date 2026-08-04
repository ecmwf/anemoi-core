# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .aggregate import TimeAggregateLossWrapper
from .combined import CombinedLoss
from .huber import HuberLoss
from .kcrps import CRPS
from .logcosh import LogCoshLoss
from .loss import get_loss_function
from .mae import MAELoss
from .mse import MSELoss
from .multiscale import MultiscaleLossWrapper
from .rmse import RMSELoss
from .spectral import FourierCorrelationLoss
from .spectral import LogFFT2Distance
from .spectral import LogSpectralDistance
from .spectral import PowerSpectrumLoss
from .spectral import SpectralAMSELoss
from .spectral import SpectralCRPSLoss
from .variable_mapper import LossVariableMapper
from .weighted_mse import WeightedMSELoss

__all__ = [
    "CRPS",
    "CombinedLoss",
    "FourierCorrelationLoss",
    "HuberLoss",
    "LogCoshLoss",
    "LogFFT2Distance",
    "LogSpectralDistance",
    "LossVariableMapper",
    "MAELoss",
    "MSELoss",
    "MultiscaleLossWrapper",
    "PowerSpectrumLoss",
    "RMSELoss",
    "SpectralAMSELoss",
    "SpectralCRPSLoss",
    "TimeAggregateLossWrapper",
    "WeightedMSELoss",
    "get_loss_function",
]
