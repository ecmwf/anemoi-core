# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Training-side :class:`~anemoi.utils.parametrisation.Parametrisation`.

During training the model is parametrised from the (Hydra/OmegaConf) training config. This
subclass is Hydra-backed (:class:`HydraParametrisation`): ``create_module`` delegates to
``hydra.utils.instantiate``, so object construction matches current practice exactly, while
``get`` reads values from the config (interpolations resolved on access). At inference the
same parametrisation is rebuilt from the checkpoint JSON as a plain
:class:`~anemoi.utils.parametrisation.DictParametrisation`.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.utils.parametrisation import HydraParametrisation

__all__ = ["TrainingParametrisation"]


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
