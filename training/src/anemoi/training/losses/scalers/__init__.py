# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .base_scaler import TensorDim
from .loss_weights_mask import NaNMaskScaler
from .node_attributes import GraphNodeAttributeScaler
from .node_attributes import ReweightedGraphNodeAttributeScaler
from .scalers import create_scalers
from .variable import GeneralVariableLossScaler
from .variable_level import LinearVariableLevelScaler
from .variable_level import NoVariableLevelScaler
from .variable_level import PolynomialVariableLevelScaler
from .variable_level import ReluVariableLevelScaler
from .variable_masking import VariableMaskingLossScaler
from .variable_tendency import NoTendencyScaler
from .variable_tendency import StdevTendencyScaler
from .variable_tendency import VarTendencyScaler

__all__ = [
    "GeneralVariableLossScaler",
    "GraphNodeAttributeScaler",
    "LinearVariableLevelScaler",
    "NaNMaskScaler",
    "NoTendencyScaler",
    "NoVariableLevelScaler",
    "PolynomialVariableLevelScaler",
    "ReluVariableLevelScaler",
    "ReweightedGraphNodeAttributeScaler",
    "StdevTendencyScaler",
    "TensorDim",
    "VarTendencyScaler",
    "VariableMaskingLossScaler",
    "create_scalers",
]
