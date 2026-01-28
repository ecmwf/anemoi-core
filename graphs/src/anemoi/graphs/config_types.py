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


class NodeConfig(ConfigBase):
    node_builder: ConfigNode
    attributes: ConfigNode | None = None


class EdgeConfig(ConfigBase):
    source_name: str
    target_name: str
    edge_builders: list[ConfigNode]
    attributes: ConfigNode | None = None


class GraphConfig(ConfigBase):
    data: str | None = None
    hidden: str | list[str] | None = None
    overwrite: bool | None = None
    nodes: dict[str, NodeConfig] | None = None
    edges: list[EdgeConfig] | None = None
    post_processors: list[ConfigNode] | None = None


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
    "EdgeConfig",
    "GraphConfig",
    "NodeConfig",
    "get_path",
    "to_container",
]
