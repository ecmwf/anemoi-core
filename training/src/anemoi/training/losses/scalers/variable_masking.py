# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers.variable import GeneralVariableLossScaler
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

LOGGER = logging.getLogger(__name__)


class VariableMaskingLossScaler(GeneralVariableLossScaler):
    """Class for masking variables out in the loss.

    Use `invert=True` to set this to only mask the specified
    variables in.
    """

    def __init__(
        self,
        variables: list[str],
        data_indices: IndexCollection,
        metadata_extractor: ExtractVariableGroupAndLevel,
        invert: bool = False,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        weights = dict.fromkeys(variables, 0.0) if not invert else dict.fromkeys(variables, 1.0)
        weights["default"] = 1.0 if not invert else 0.0
        super().__init__(data_indices, DictConfig(weights), metadata_extractor, norm, **kwargs)
