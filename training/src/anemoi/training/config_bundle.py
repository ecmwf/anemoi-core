# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from omegaconf import DictConfig
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import ListConfig


_MODEL_CONFIG_EXCLUDE_KEYS = {
    "training_config",
    "data_config",
    "dataloader_config",
    "graph_config",
    "system_config",
    "config_bundle",
}


def _clone_config_section(section: Any) -> DictConfig | ListConfig:
    return OmegaConf.create(OmegaConf.to_container(section, resolve=True))


def _clean_model_section(model_config: DictConfig) -> DictConfig:
    clean_model_dict = {
        key: OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
        for key, value in model_config.items()
        if key not in _MODEL_CONFIG_EXCLUDE_KEYS
    }
    return OmegaConf.create(clean_model_dict)


@dataclass(frozen=True)
class ModelConfigBundle:
    """Parts of the config used to create the model."""

    training: DictConfig
    data: DictConfig
    model: DictConfig

    @classmethod
    def from_root_config(cls, root_config: DictConfig) -> ModelConfigBundle:
        return cls(
            training=_clone_config_section(root_config.training),
            data=_clone_config_section(root_config.data),
            model=_clean_model_section(root_config.model),
        )

    def to_dictconfig(self) -> DictConfig:
        return OmegaConf.create(
            {
                "training": OmegaConf.to_container(self.training, resolve=True),
                "data": OmegaConf.to_container(self.data, resolve=True),
                "model": OmegaConf.to_container(self.model, resolve=True),
            },
        )


@dataclass(frozen=True)
class TaskConfigBundle:
    """Parts of the config used by the training task."""

    training: DictConfig
    system: DictConfig
    dataloader: DictConfig
    graph: DictConfig
    model: DictConfig

    @classmethod
    def from_root_config(cls, root_config: DictConfig) -> TaskConfigBundle:
        return cls(
            training=_clone_config_section(root_config.training),
            system=_clone_config_section(root_config.system),
            dataloader=_clone_config_section(root_config.dataloader),
            graph=_clone_config_section(root_config.graph),
            model=_clean_model_section(root_config.model),
        )

    def to_dictconfig(self) -> DictConfig:
        return OmegaConf.create(
            {
                "training": OmegaConf.to_container(self.training, resolve=True),
                "system": OmegaConf.to_container(self.system, resolve=True),
                "dataloader": OmegaConf.to_container(self.dataloader, resolve=True),
                "graph": OmegaConf.to_container(self.graph, resolve=True),
                "model": OmegaConf.to_container(self.model, resolve=True),
            },
        )
