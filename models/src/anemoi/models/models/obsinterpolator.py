# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging

from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelObsInterpolator(AnemoiModelEncProcDec):
    """Message passing interpolating graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        truncation_data: dict,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        config : DotDict
            Job configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        model_config = DotDict(model_config)
        self.known_future_variables = (
            []
            if model_config.training.get("known_future_variables", None) is None
            else OmegaConf.to_container(model_config.training.known_future_variables, resolve=True)
        )
        self.num_channels = model_config.model.num_channels
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            truncation_data=truncation_data,
        )

    def _calculate_input_dim(self):
        return (
            self.multi_step * self.num_input_channels # past context for observations
            + 2 * len(self.known_future_variables)  # 1 for specific interpolation time 1, for upper bound
            + 1 # for time fraction
            + self.node_attributes.attr_ndims[self._graph_name_data]
        )
