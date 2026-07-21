# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Training-side, Hydra-backed :class:`~anemoi.utils.parametrisation.Parametrisation`.

Object construction in the training / config-driven world goes through Hydra (current
practice). :class:`HydraParametrisation` is the ``create_module`` hook ``refactor.md``
reserved for "bringing back Hydra": it delegates to ``hydra.utils.instantiate`` so behaviour
matches existing config instantiation exactly (interpolations, ``_convert_``, recursive
nested targets, ``_partial_``). It lives in ``anemoi.training`` because Hydra is a training
concern; ``anemoi.graphs`` / ``anemoi.models`` stay Hydra-free and use
:class:`~anemoi.utils.parametrisation.DictParametrisation`.

:class:`TrainingParametrisation` is the concrete parametrisation the trainer builds from the
OmegaConf config and hands to the model; at inference the same parametrisation is rebuilt
from the checkpoint JSON as a plain ``DictParametrisation``.
"""

from __future__ import annotations

from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.utils.parametrisation import MISSING
from anemoi.utils.parametrisation import Parametrisation
from anemoi.utils.parametrisation import ParametrisationError

__all__ = ["HydraParametrisation", "TrainingParametrisation"]


class HydraParametrisation(Parametrisation):
    """Parametrisation backed by Hydra/OmegaConf -- matches current practice.

    ``create_module`` delegates to :func:`hydra.utils.instantiate`; ``get`` reads values
    from the (OmegaConf) config, resolving interpolations on access.
    """

    def __init__(self, config: Any = None) -> None:
        self._config = config if config is not None else {}

    @classmethod
    def from_config(cls, config: Any) -> HydraParametrisation:
        """Build from an OmegaConf/dict config (kept as-is for interpolation-aware ``get``)."""
        return cls(config)

    def get(self, key: str, default: Any = MISSING) -> Any:
        node: Any = self._config
        for part in key.split("."):
            try:
                node = node[part]  # OmegaConf resolves interpolations on access
            except (KeyError, TypeError, AttributeError):
                if default is MISSING:
                    msg = f"Missing parameter {key!r}"
                    raise ParametrisationError(msg) from None
                return default
        return node

    def to_dict(self) -> dict:
        if OmegaConf.is_config(self._config):
            return OmegaConf.to_container(self._config, resolve=True)
        return dict(self._config)

    def _build_spec(self, spec: Any, *args: Any, **kwargs: Any) -> Any:
        # Class / instance dispatch is handled by Parametrisation.create_module; here we only
        # build from a spec (dotted-path string / _target_ mapping) via Hydra.
        return instantiate(spec, *args, **kwargs)


class TrainingParametrisation(HydraParametrisation):
    """Parametrisation built from the training (OmegaConf) config.

    Values (e.g. ``number_of_channels``) are read from the config; dataset-derived values can
    be layered on top before the model is built. Sub-modules are built with Hydra.
    """

    @classmethod
    def from_config(cls, config: DictConfig | dict, **overrides: Any) -> TrainingParametrisation:
        """Build from an OmegaConf/dict training config.

        The config is kept as-is (OmegaConf interpolations resolve on access). ``overrides``
        let callers inject dataset-derived values on top.
        """
        if overrides:
            config = OmegaConf.merge(config, overrides) if isinstance(config, DictConfig) else {**config, **overrides}
        return cls(config)
