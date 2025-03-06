# (C) Copyright 2025- Anemoi contributors.
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

from hydra.utils import instantiate

from anemoi.training.losses.scalers.base_scaler import BaseDelayedScaler

if TYPE_CHECKING:

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.losses.scalers.base_scaler import SCALER_DTYPE
    from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


def create_scalers(
    scalers_config: DotDict,
    data_indices: IndexCollection,
    **kwargs,
) -> tuple[dict[str, SCALER_DTYPE], dict[str, SCALER_DTYPE]]:
    scalers, delayed_scaler_builders = {}, {}
    for name, config in scalers_config.items():
        scaler_builder = instantiate(config, data_indices=data_indices, **kwargs)

        if isinstance(scaler_builder, BaseDelayedScaler):
            delayed_scaler_builders[name] = scaler_builder

        scalers[name] = scaler_builder.get_scaling()

    return scalers, delayed_scaler_builders
