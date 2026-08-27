# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

import yamlrocks
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from anemoi.training.migrations.interpolations import InterpolationHandler

LOGGER = logging.getLogger(__name__)


class NodeBase(ABC):
    @property
    @abstractmethod
    def prefix(self) -> Sequence[str | int]: ...

    @property
    @abstractmethod
    def prefix_str(self) -> str: ...

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
    def parent(self) -> NodeContainer: ...

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
        parent: NodeContainer,
        yr_parent: yamlrocks.YAMLRocksDocument | yamlrocks.YAMLRocksDocumentView,
        cfg_parent: DictConfig,
        interpolation_handler: InterpolationHandler,
        key: Any,
        prefix: Sequence[str | int] = (),
    ):
        self._parent = parent
        self.yaml_parent = yr_parent
        self.cfg_parent = cfg_parent
        self.key = key
        self._prefix = prefix

        self._interpolation_handler = interpolation_handler

    @cached_property
    def prefix(self) -> Sequence[str | int]:
        return (*self._prefix, self.key)

    @cached_property
    def prefix_str(self) -> str:
        return ".".join(map(str, self.prefix))

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
    def parent(self) -> NodeContainer:
        return self._parent

    def get(self, key: str | int) -> Node:
        """Gets the key while asserting that this node is a NodeContainer.
        If it isnt't, raises a TypeError at runtime.

        Parameters
        ----------
        key : str | int
            Key to get.

        Returns
        -------
        Node
            The requeted node.
        """
        if not isinstance(self, NodeContainer):
            raise TypeError("This node is not a NodeContainer.")
        return self[key]

    def set(self, key: str | int, value: Any) -> None:
        """Sets the key while asserting that this node is a NodeContainer.
        If it isnt't, raises a TypeError at runtime.

        Parameters
        ----------
        key : str | int
            Key to set.
        value : Any
            New value to set.
        """
        if not isinstance(self, NodeContainer):
            raise TypeError("This node is not a NodeContainer.")
        self[key] = value

    def delete(self, key: str | int) -> None:
        """Deletes the key while asserting that this node is a NodeContainer.
        If it isnt't, raises a TypeError at runtime.

        Parameters
        ----------
        key : str | int
            Key to delete.
        """
        if not isinstance(self, NodeContainer):
            raise TypeError("This node is not a NodeContainer.")
        del self[key]

    def __repr__(self) -> str:
        return f"Node({self.prefix_str})"


def parse_key(key: str) -> tuple[list[str], str]:
    parts = key.split(".")
    return parts[:-1], parts[-1]


class NodeContainer(Node, ABC):
    def __init__(
        self,
        parent: NodeContainer,
        yr_parent: yamlrocks.YAMLRocksDocument | yamlrocks.YAMLRocksDocumentView,
        cfg_parent: DictConfig,
        interpolation_handler: InterpolationHandler,
        key: Any,
        prefix: Sequence[str | int] = (),
    ):
        self._parent = parent
        self.yaml_parent = yr_parent
        self.cfg_parent = cfg_parent
        self._interpolation_handler = interpolation_handler
        self.key = key
        self._prefix = prefix

    @abstractmethod
    def _is_key_valid(self, key: str | int) -> bool: ...

    def is_interpolation(self, key: str | int) -> bool:
        return OmegaConf.is_interpolation(self.cfg, key)

    def __contains__(self, key: str | int) -> bool:
        return self._is_key_valid(key)

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, key: str | int) -> Node:
        if not self._is_key_valid(key):
            raise ValueError(f"key {key} not in Node.")

        if isinstance(self.cfg[key], ListConfig):
            cls = NodeList
        elif isinstance(self.cfg[key], DictConfig):
            cls = NodeDict
        else:
            cls = Node

        return cls(
            self,
            self.yaml,
            self.cfg,
            self._interpolation_handler,
            key,
            self.prefix,
        )

    def __setitem__(self, key: str | int, value: Any) -> None:
        self.yaml[key] = value
        self.cfg[key] = value
        self._interpolation_handler.update(self)

    def __delitem__(self, key: str | int) -> None:
        del self.yaml[key]
        del self.cfg[key]
        self._interpolation_handler.update(self)

    def has_key(self, parts: Sequence[str | int]) -> bool:
        try:
            self.select(parts)
        except (TypeError, ValueError):
            return False
        return True

    def select(self, parts: Sequence[str | int], create_missing: bool = False) -> Node:
        node = self
        for part in parts:
            if not isinstance(node, NodeContainer):
                raise TypeError(f"Cannot select {part}. Not a container node.")
            if part not in node and create_missing:
                node[part] = {}
            node = node[part]
        return node

    def drop_key(self, keys: str, remove_empty: bool = False) -> None:
        parents, key = parse_key(keys)
        parent_node = self.select(parents)

        if not isinstance(parent_node, NodeContainer):
            raise TypeError(f"Cannot delete node {keys}. Not a container node.")

        if not remove_empty:
            del parent_node[key]
            return

        parts = keys.split(".")
        head_key_k = len(parts) - 1
        while len(parent_node.value) == 1:
            parent_node = parent_node.parent
            head_key_k -= 1

        del parent_node[parts[head_key_k]]

    def add_key(self, keys: str, value: Any) -> None:
        parents, key = parse_key(keys)
        parent_node = self.select(parents, create_missing=True)
        if not isinstance(parent_node, NodeContainer):
            raise TypeError(f"Cannot add node {keys}. Not a container node.")
        parent_node[key] = value

    def rename_key(self, start: str, end: str, remove_empty: bool = False) -> None:
        parts = start.split(".")
        start_node = self.select(parts)
        value = start_node.value
        self.add_key(end, value)
        self.drop_key(start, remove_empty)


class NodeDict(NodeContainer):
    def _is_key_valid(self, key: str | int) -> bool:
        return key in self.cfg

    def __iter__(self):
        return iter(self.value)

    def __repr__(self) -> str:
        return f"NodeDict({self.prefix_str})"


class NodeList(NodeContainer):
    def _is_key_valid(self, key: str | int) -> bool:
        return len(self.cfg) > int(key)

    def keys(self) -> Iterable[int]:
        return range(len(self))

    def append(self, value: Any) -> None:
        self.cfg.append(value)
        new_val = self.yaml_node.value
        new_val.append(value)
        self.yaml_node.value = new_val

    def __getitem__(self, key: str | int) -> Node:
        return super().__getitem__(int(key))

    def __repr__(self) -> str:
        return f"NodeList({self.prefix_str})"
