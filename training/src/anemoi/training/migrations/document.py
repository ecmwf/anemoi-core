# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

import yamlrocks
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf


class NodeBase(ABC):
    @property
    @abstractmethod
    def prefix(self) -> str: ...

    @property
    @abstractmethod
    def yaml(self) -> yamlrocks.YAMLRocksDocumentView: ...

    @property
    @abstractmethod
    def yaml_node(self) -> yamlrocks.YAMLRocksNode: ...

    @property
    @abstractmethod
    def cfg(self) -> Any: ...

    @property
    def value(self) -> Any:
        return self.yaml_node.value

    @property
    @abstractmethod
    def parent(self) -> NodeBase: ...

    @abstractmethod
    def _is_key_valid(self, key: Any) -> bool: ...

    def __contains__(self, key: Any) -> bool:
        return self._is_key_valid(key)

    def __getitem__(self, key: Any) -> NodeBase:
        if not self._is_key_valid(key):
            raise ValueError(f"key {key} not in Node.")

        if isinstance(self.cfg[key], ListConfig):
            cls = NodeList
        else:
            cls = Node

        return cls(
            self,
            self.yaml,
            self.cfg,
            key,
            self.prefix,
        )

    def __setitem__(self, key: Any, value: Any) -> None:
        self.yaml[key] = value
        self.cfg[key] = value

    def __delitem__(self, key: str) -> None:
        del self.yaml[key]
        del self.cfg[key]

    def set_comments(
        self,
        before: str | None = None,
        inline: str | None = None,
        after: str | None = None,
    ) -> None:
        if before is not None:
            if self.yaml_node.comment_before is not None:
                before = f"{before}\n{self.yaml_node.comment_before}"
            self.yaml_node.comment_before = before
        if inline is not None:
            if self.yaml_node.comment is not None:
                inline = f"{inline} {self.yaml_node.comment}"
            self.yaml_node.comment = inline
        if after is not None:
            if self.yaml_node.comment_after is not None:
                after = f"{self.yaml_node.comment_after}\n{after}"
            try:
                self.yaml_node.comment_after = after
            except ValueError:
                if self.yaml_node.comment is not None:
                    after = f"{self.yaml_node.comment}\n# {after}"
                self.yaml_node.comment = after


class Node(NodeBase):
    def __init__(
        self,
        parent: NodeBase,
        yr_parent: yamlrocks.YAMLRocksDocument | yamlrocks.YAMLRocksDocumentView,
        cfg_parent: DictConfig,
        key: Any,
        prefix: str | None = None,
    ):
        self._parent = parent
        self.yaml_parent = yr_parent
        self.cfg_parent = cfg_parent
        self.key = key
        self._prefix = prefix or ""

    @property
    def prefix(self) -> Sequence[str]:
        return f"{self._prefix}.{self.key}".removeprefix(".")

    @property
    def yaml(self) -> yamlrocks.YAMLRocksDocumentView:
        return self.yaml_parent[self.key]

    @property
    def yaml_node(self) -> yamlrocks.YAMLRocksNode:
        return self.yaml_parent.node[self.key]

    @property
    def cfg(self) -> Any:
        return self.cfg_parent[self.key]

    @property
    def parent(self) -> NodeBase:
        return self._parent

    def _is_key_valid(self, key: Any) -> bool:
        return key in self.cfg

    def __repr__(self) -> str:
        return f"NodeDict({self.prefix})"


class NodeList(Node):
    def _is_key_valid(self, key: Any) -> bool:
        return len(self.cfg) > int(key)

    def append(self, value: Any) -> None:
        self.cfg.append(value)
        new_val = self.yaml_node.value
        new_val.append(value)
        self.yaml_node.value = new_val

    def __getitem__(self, key: Any) -> NodeBase:
        return super().__getitem__(int(key))

    def __repr__(self) -> str:
        return f"NodeList({self.prefix})"


def parse_key(key: str) -> tuple[list[str], str]:
    parts = key.split(".")
    return parts[:-1], parts[-1]


class Document(NodeBase):
    def __init__(self, path: Path | str, prefix: str | None = None) -> None:
        self._path = Path(path)
        self._prefix = prefix or ""
        self._cfg = OmegaConf.load(self._path)

    @property
    def prefix(self) -> str:
        return self._prefix

    @cached_property
    def yaml(self) -> yamlrocks.YAMLRocksDocument:
        doc = yamlrocks.load(self._path, option=yamlrocks.OPT_ROUND_TRIP)
        assert isinstance(doc, yamlrocks.YAMLRocksDocument)
        return doc

    @property
    def yaml_node(self) -> yamlrocks.YAMLRocksNode:
        return self.yaml.node

    @property
    def cfg(self) -> Any:
        return self._cfg

    @property
    def parent(self) -> NodeBase:
        return self

    def _is_key_valid(self, key: Any) -> bool:
        return key in self.cfg

    def to_yaml(self) -> str:
        return self.yaml.to_yaml().decode()

    def select(self, parts: Sequence[str], create_missing: bool = False) -> NodeBase:
        node = self
        for part in parts:
            if part not in node and create_missing:
                node[part] = {}
            node = node[part]
        return node

    def drop_key(self, keys: str, remove_empty: bool = False) -> None:
        parents, key = parse_key(keys)
        parent_node = self.select(parents)

        if not remove_empty:
            del parent_node[key]
            return

        parts = keys.split(".")
        head_key_k = len(parts) - 1
        while isinstance(parent_node.cfg, (ListConfig, DictConfig)) and len(parent_node.value) == 1:
            parent_node = parent_node.parent
            head_key_k -= 1
        del parent_node[parts[head_key_k]]

    def add_key(self, keys: str, value: Any) -> None:
        parents, key = parse_key(keys)
        parent_node = self.select(parents, create_missing=True)
        parent_node[key] = value

    def rename_key(self, start: str, end: str, remove_empty: bool = False) -> None:
        parts = start.split(".")
        start_node = self.select(parts)
        value = start_node.value
        self.add_key(end, value)
        self.drop_key(start, remove_empty)

    def __repr__(self) -> str:
        return f"Document({self._path}, {self.prefix})"
