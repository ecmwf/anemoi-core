# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from anemoi.training.builders.components import build_component
from anemoi.training.losses.scalers.scalers import create_scalers

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anemoi.training.losses.scaler_tensor import TENSOR_SPEC
    from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler


def build_scalers_from_config(
    scalers_config: Mapping[str, Any],
    **kwargs: Any,
) -> tuple[dict[str, TENSOR_SPEC], dict[str, BaseUpdatingScaler]]:
    """Instantiate scalers from config and return scale tensors plus updating scalers."""
    scaler_builders = {name: build_component(config, **kwargs) for name, config in scalers_config.items()}
    return create_scalers(scaler_builders)
