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

from anemoi.utils.parametrisation import DictParametrisation

if TYPE_CHECKING:
    from omegaconf import DictConfig

_PARAMETRISATION = DictParametrisation()


def instantiate_with_runtime_kwargs(instantiate_config: DictConfig, **runtime_kwargs: Any) -> Any:
    """Build an object from a spec with kwargs that are only available at runtime.

    The configured target is first resolved into a partial factory (no recursive
    construction), then the runtime kwargs are applied through regular Python so values
    such as the full config are not resolved or recursively built.
    """
    factory = _PARAMETRISATION.create_module(instantiate_config, _partial_=True)
    return factory(**runtime_kwargs)
