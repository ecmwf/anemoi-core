# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Hydra-backed :class:`~anemoi.utils.parametrisation.Parametrisation` for training.

`HydraParametrisation` builds specs with ``hydra.utils.instantiate`` (Hydra's full instantiate
semantics). Use it only where that is actually required; everywhere else prefer the Hydra-free
:class:`~anemoi.utils.parametrisation.DictParametrisation` (via ``Parametrisation.from_dict``).

It subclasses :class:`~anemoi.utils.parametrisation.DictParametrisationBase` (never an
instantiated class), and is itself a concrete leaf -- do not subclass it.
"""

from __future__ import annotations

from typing import Any

from anemoi.utils.parametrisation import DictParametrisationBase

__all__ = ["HydraParametrisation"]


class HydraParametrisation(DictParametrisationBase):
    """A dict-backed parametrisation whose ``create_module`` builds specs with Hydra."""

    def _build_spec(self, spec: Any, *args: Any, **kwargs: Any) -> Any:
        # Class / instance dispatch is handled by Parametrisation.create_module; here we only
        # build from a spec (dotted-path string / _target_ mapping) via Hydra. Import lazily so
        # importing this module does not pull in Hydra until a spec is actually built.
        from hydra.utils import instantiate

        return instantiate(spec, *args, **kwargs)
