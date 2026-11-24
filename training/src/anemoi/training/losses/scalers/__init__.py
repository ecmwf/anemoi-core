# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .base_scaler import TensorDim
from .loss_weights_mask import NaNMaskScaler
from .node_attributes import GraphNodeAttributeScaler, ReweightedGraphNodeAttributeScaler
from .scalers import create_scalers
from .spectral import SpectralDimensionScaler
from .time_step import LeadTimeDecayScaler, TimeStepScaler, UniformTimeStepScaler
from .variable import GeneralVariableLossScaler
from .variable_level import (
    LinearVariableLevelScaler,
    NoVariableLevelScaler,
    PolynomialVariableLevelScaler,
    ReluVariableLevelScaler,
)
from .variable_masking import VariableMaskingLossScaler
from .variable_tendency import NoTendencyScaler, StdevTendencyScaler, VarTendencyScaler

__all__ = [
    "GeneralVariableLossScaler",
    "GraphNodeAttributeScaler",
    "LeadTimeDecayScaler",
    "LinearVariableLevelScaler",
    "NaNMaskScaler",
    "NoTendencyScaler",
    "NoVariableLevelScaler",
    "PolynomialVariableLevelScaler",
    "ReluVariableLevelScaler",
    "ReweightedGraphNodeAttributeScaler",
    "SpectralDimensionScaler",
    "StdevTendencyScaler",
    "TensorDim",
    "TimeStepScaler",
    "UniformTimeStepScaler",
    "VarTendencyScaler",
    "VariableMaskingLossScaler",
    "create_scalers",
]
