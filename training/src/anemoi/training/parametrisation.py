# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Training-side :class:`~anemoi.utils.parametrisation.Parametrisation`.

During training the model is parametrised from the (Hydra/OmegaConf) training config plus
dataset-derived values. This subclass adapts that config into the abstract
:class:`Parametrisation` the models consume, and keeps a JSON-serialisable snapshot so the
same parametrisation can be rebuilt at inference time from the checkpoint.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.utils.parametrisation import DictParametrisation

__all__ = ["TrainingParametrisation"]


class TrainingParametrisation(DictParametrisation):
    """Parametrisation built from the training (OmegaConf) config.

    Values (e.g. ``number_of_channels``) are read from the resolved config; dataset-derived
    values can be layered on top before the model is built.
    """

    @classmethod
    def from_config(cls, config: DictConfig | dict, **overrides: Any) -> TrainingParametrisation:
        """Build from an OmegaConf/dict training config, resolving all interpolations.

        ``overrides`` let callers inject dataset-derived values (deep-merged on top).
        """
        data = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else dict(config)
        data.update(overrides)
        return cls(data)
