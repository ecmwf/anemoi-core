# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel
from omegaconf import DictConfig
from anemoi.training.losses.scalers.variable import GeneralVariableLossScaler

if TYPE_CHECKING:
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

LOGGER = logging.getLogger(__name__)


class VariableMaskingLossScaler(GeneralVariableLossScaler):
    """Class for masking variables in the loss."""
    def __init__(self, variables:list[str], data_indices: IndexCollection, metadata_extractor: ExtractVariableGroupAndLevel, inverse:bool = False, norm: str | None = None, **kwargs) -> None:
        weights =  {var: 0.0 for var in variables} if not inverse else {var: 1.0 for var in variables}
        weights["default"] = 1.0 if not inverse else 0.0
        super().__init__(data_indices, DictConfig(weights), metadata_extractor, norm, **kwargs)

