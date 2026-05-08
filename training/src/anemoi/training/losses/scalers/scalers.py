# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from hydra.utils import instantiate

from anemoi.training.losses.scaler_tensor import TENSOR_SPEC
from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


def create_scalers(scalers_config: DotDict, **kwargs) -> tuple[dict[str, TENSOR_SPEC], dict[str, BaseUpdatingScaler]]:
    scalers, updating_scalars = {}, {}
    for name, config in scalers_config.items():
        scaler_builder: BaseScaler = instantiate(config, **kwargs)

        if isinstance(scaler_builder, BaseUpdatingScaler):
            updating_scalars[name] = scaler_builder

        scalers[name] = scaler_builder.get_scaling()

    return scalers, updating_scalars
