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

from hydra.utils import instantiate

if TYPE_CHECKING:
    from omegaconf import DictConfig


def instantiate_with_runtime_kwargs(instantiate_config: DictConfig, **runtime_kwargs: Any) -> Any:
    """Instantiate a Hydra config with kwargs that are only available at runtime.

    Hydra first resolves the configured target into a partial factory. Runtime
    kwargs are then passed through regular Python so values such as the full
    config are not resolved or recursively instantiated by Hydra.
    """
    factory = instantiate(instantiate_config, _partial_=True)
    return factory(**runtime_kwargs)
