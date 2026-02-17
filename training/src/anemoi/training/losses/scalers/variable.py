# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel

LOGGER = logging.getLogger(__name__)


class BaseVariableLossScaler(BaseScaler):
    """Base class for all variable loss scalers."""

    scale_dims: TensorDim = TensorDim.VARIABLE

    def __init__(
        self,
        data_indices: IndexCollection,
        metadata_extractor: ExtractVariableGroupAndLevel,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise Scaler.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices.
        metadata_extractor : ExtractVariableGroupAndLevel
            Metadata extractor for variable groups and levels.
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(norm=norm)
        del kwargs
        self.data_indices = data_indices
        self.variable_metadata_extractor = metadata_extractor


class GeneralVariableLossScaler(BaseVariableLossScaler):
    """Scaling per variable defined in config file.

    Supports per-level and per-channel weight specification for variables.
    When a variable has levels or channels (e.g., z_500, z_500_ch0), you can specify:
    - Specific level/channel weights (e.g., z_500: 250, z_500_ch0: 300)
    - Base variable weight applied to all levels/channels (e.g., z: 150)
    - Default weight for all unspecified variables

    The lookup follows a fallback chain:
    variable_name (e.g., z_500) -> base_variable (e.g., z) -> default
    """

    def __init__(
        self,
        data_indices: IndexCollection,
        weights: DictConfig,
        metadata_extractor: ExtractVariableGroupAndLevel,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise GeneralVariableLossScaler.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices.
        weights : DictConfig
            Configuration for variable loss scaling. Can specify weights by:
            - Full variable name with level/channel (e.g., "z_500": 250, "z_500_ch0": 300)
            - Base variable name (e.g., "z": 150) applied to all levels/channels
            - "default": fallback weight for unspecified variables
        metadata_extractor : ExtractVariableGroupAndLevel
            Metadata extractor for variable groups and levels.
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        super().__init__(data_indices, metadata_extractor=metadata_extractor, norm=norm)
        self.weights = weights
        del kwargs

    def get_scaling_values(self, **_kwargs) -> torch.Tensor:
        """Get loss scaling.

        Retrieve the loss scaling for each variable from the config file.
        Supports per-level and per-channel specification (e.g., z_500, z_500_ch0) with fallback
        to base variable name (e.g., z) and then to default.

        The lookup follows a fallback chain:
        1. Check for exact variable name match (e.g., "z_500": 250)
        2. Fall back to base variable name (e.g., "z": 150)
        3. Fall back to default weight
        """
        variable_loss_scaling = torch.empty((len(self.data_indices.data.output.full),), dtype=torch.float32)

        for variable_name, idx in self.data_indices.model.output.name_to_index.items():
            _, variable_ref, _ = self.variable_metadata_extractor.get_group_and_level(variable_name)
            # Apply variable scaling by specific variable name (e.g. z_500, z_500_ch0),
            # falling back to base variable name (e.g. z), then default
            variable_loss_scaling[idx] = self.weights.get(
                variable_name,
                self.weights.get(variable_ref, self.weights.get("default", 1.0)),
            )

        return variable_loss_scaling
