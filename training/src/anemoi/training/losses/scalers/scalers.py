# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

from anemoi.training.losses.scaler_tensor import TENSOR_SPEC
from anemoi.training.losses.scalers.base_scaler import BaseScaler
from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler

LOGGER = logging.getLogger(__name__)


def _build_scaler(scaler: BaseScaler | Callable[..., BaseScaler], **kwargs: Any) -> BaseScaler:
    if isinstance(scaler, BaseScaler):
        return scaler
    if callable(scaler):
        built = scaler(**kwargs)
        if not isinstance(built, BaseScaler):
            error_msg = f"Scaler factory must return a BaseScaler, not {type(built)}"
            raise TypeError(error_msg)
        return built
    error_msg = f"Scaler must be a BaseScaler or a callable factory, not {type(scaler)}"
    raise TypeError(error_msg)


def create_scalers(
    scalers: Mapping[str, BaseScaler | Callable[..., BaseScaler]],
    **kwargs: Any,
) -> tuple[dict[str, TENSOR_SPEC], dict[str, BaseUpdatingScaler]]:
    scalers_out, updating_scalars = {}, {}
    for name, scaler in scalers.items():
        scaler_builder = _build_scaler(scaler, **kwargs)

        if isinstance(scaler_builder, BaseUpdatingScaler):
            updating_scalars[name] = scaler_builder

        scalers_out[name] = scaler_builder.get_scaling()

    return scalers_out, updating_scalars
