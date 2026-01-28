# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict


class ConfigBase(BaseModel, Mapping[str, Any]):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, validate_assignment=True)

    def __getattr__(self, item: str) -> Any:
        extra = self.__pydantic_extra__ or {}
        if item in extra:
            value = self._wrap_value(extra[item])
            extra[item] = value
            return value
        raise AttributeError(item)

    def __getitem__(self, key: str) -> Any:
        if key in self.model_fields:
            return self._wrap_value(getattr(self, key))
        extra = self.__pydantic_extra__ or {}
        if key in extra:
            return self._wrap_value(extra[key])
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw_items())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._raw_items())

    def _raw_items(self) -> dict[str, Any]:
        data: dict[str, Any] = {name: getattr(self, name) for name in self.model_fields}
        data.update(self.__pydantic_extra__ or {})
        return data

    def keys(self) -> list[str]:
        return list(self._raw_items().keys())

    def items(self) -> list[tuple[str, Any]]:
        return [(key, self[key]) for key in self._raw_items().keys()]

    def values(self) -> list[Any]:
        return [self[key] for key in self._raw_items().keys()]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    @classmethod
    def _wrap_value(cls, value: Any) -> Any:
        if isinstance(value, ConfigBase):
            return value
        if isinstance(value, Mapping):
            if all(isinstance(key, str) for key in value.keys()):
                return ConfigNode(**value)
            return {key: cls._wrap_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._wrap_value(item) for item in value]
        return value


class ConfigNode(ConfigBase):
    """Generic config node with attribute access."""


class GraphConfig(ConfigBase):
    data: str | None = None
    hidden: str | list[str] | None = None


class TrainingConfig(ConfigBase):
    multistep_input: int | None = None
    explicit_times: ConfigNode | None = None
    target_forcing: ConfigNode | None = None
    model_task: str | None = None


class ModelConfig(ConfigBase):
    model: ConfigNode
    num_channels: int | None = None
    cpu_offload: bool | None = None
    keep_batch_sharded: bool | None = None
    trainable_parameters: ConfigNode | None = None
    residual: ConfigNode | None = None
    output_mask: ConfigNode | None = None
    bounding: list[ConfigNode] | None = None
    encoder: ConfigNode | None = None
    decoder: ConfigNode | None = None
    processor: ConfigNode | None = None
    layer_kernels: ConfigNode | None = None
    latent_skip: bool | None = None
    noise_injector: ConfigNode | None = None
    condition_on_residual: bool | None = None


class DataConfig(ConfigBase):
    datasets: ConfigNode | None = None
    format: str | None = None
    timestep: str | None = None
    frequency: str | None = None


class Settings(ConfigBase):
    model: ModelConfig
    training: TrainingConfig
    graph: GraphConfig
    data: DataConfig | None = None


def to_container(value: Any) -> Any:
    if isinstance(value, ConfigBase):
        return {key: to_container(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {key: to_container(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_container(item) for item in value]
    return value


def get_path(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return current


__all__ = [
    "ConfigBase",
    "ConfigNode",
    "DataConfig",
    "GraphConfig",
    "ModelConfig",
    "Settings",
    "TrainingConfig",
    "get_path",
    "to_container",
]
