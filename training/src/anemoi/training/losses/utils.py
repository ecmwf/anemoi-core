# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from torch import nn

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses.scalers.base_scaler import ScalerDim

LOGGER = logging.getLogger(__name__)


def print_variable_scaling(loss: type[nn.Module], data_indices: IndexCollection) -> None:
    variable_scaling = loss.scaler.subset_by_dim(ScalerDim.VARIABLE.value)
    log_text = "Final Variable Scaling: "
    for idx, name in enumerate(data_indices.internal_model.output.name_to_index.keys()):
        log_text += f"{name}: {variable_scaling[idx]:.4g}, "
    LOGGER.debug(log_text)
