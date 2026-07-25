# (C) Copyright 2026 Anemoi contributors.
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

from anemoi.utils.builder import build

if TYPE_CHECKING:
    from omegaconf import DictConfig


def instantiate_with_runtime_kwargs(instantiate_config: DictConfig, **runtime_kwargs: Any) -> Any:
    """Build an object from a config spec with kwargs only available at runtime.

    Deprecated thin wrapper around :func:`anemoi.utils.builder.build`, kept for backward
    compatibility. ``build`` already merges runtime kwargs into the target call.
    """
    return build(instantiate_config, **runtime_kwargs)
