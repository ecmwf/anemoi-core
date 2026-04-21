# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .combined import CombinedLoss
from .huber import HuberLoss
from .kcrps import AlmostFairKernelCRPS
from .kcrps import KernelCRPS
from .logcosh import LogCoshLoss
from .loss import get_loss_function
from .mae import MAELoss
from .mse import MSELoss
from .multiscale import MultiscaleLossWrapper
from .optical_flow import OpticalFlowConsistencyLoss
from .rmse import RMSELoss
from .rolling_accumulation import RollingAccumulationHuberLoss
from .spectral import FourierCorrelationLoss
from .spectral import LogFFT2Distance
from .spectral import LogSpectralDistance
from .spectral import SpectralCRPSLoss
from .spectral import SpectralL2Loss
from .ssim import SSIMLoss
from .ssim import MaskedLogSSIMLoss
from .weighted_mse import WeightedMSELoss
from .wet_area_loss import WeightedSoftWetAreaLoss
from .optical_flow import OpticalFlowConsistencyLoss
from .optical_flow import SoftWetMaskAdvectiveConsistencyLoss

__all__ = [
    "AlmostFairKernelCRPS",
    "CombinedLoss",
    "FourierCorrelationLoss",
    "HuberLoss",
    "KernelCRPS",
    "LogCoshLoss",
    "LogFFT2Distance",
    "LogSpectralDistance",
    "MAELoss",
    "MaskedLogSSIMLoss",
    "MSELoss",
    "MultiscaleLossWrapper",
    "OpticalFlowConsistencyLoss",
    "SoftWetMaskAdvectiveConsistencyLoss",
    "RMSELoss",
    "RollingAccumulationHuberLoss",
    "SpectralCRPSLoss",
    "SpectralL2Loss",
    "SSIMLoss",
    "WeightedMSELoss",
    "WeightedSoftWetAreaLoss",
    "get_loss_function",
]
