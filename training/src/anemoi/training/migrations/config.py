# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Self

import yamlrocks
from omegaconf import OmegaConf

from anemoi.training.migrations.interpolations import InterpolationHandler
from anemoi.training.migrations.nodes import NodeDict


class Config(NodeDict):
    """The entry point for the config tree.

    This is a proxy for a NodeDict that can be initialized via a config content.
    """

    def __init__(self, content: str) -> None:
        self._content = content
        self._cfg = OmegaConf.create(self._content)
        self._interpolation_handler = InterpolationHandler(self)
        self._interpolation_handler.parse_config()

    @classmethod
    def from_path(cls, path: Path | str) -> Self:
        """Create the config from its path of the filesystem.

        Parameters
        ----------
        path : Path | str
            The path to the yaml config file

        Returns
        -------
        Self
            The Config instance.
        """
        content = Path(path).read_text()
        return cls(content)

    @property
    def prefix(self) -> tuple[()]:
        """The config prefix.

        The config object doesn't have any prefix as it is the root of the config tree.
        """
        return ()

    @cached_property
    def yaml(self) -> yamlrocks.YAMLRocksDocument:
        doc = yamlrocks.loads(self._content, option=yamlrocks.OPT_ROUND_TRIP)
        assert isinstance(doc, yamlrocks.YAMLRocksDocument)
        return doc

    @property
    def yaml_node(self) -> yamlrocks.YAMLRocksNode:
        return self.yaml.node

    @property
    def cfg(self) -> Any:
        return self._cfg

    @property
    def parent(self) -> NodeDict:
        # The parent of the root node is itself.
        return self

    def to_yaml(self) -> str:
        """Export the config into yaml."""
        return self.yaml.to_yaml().decode()

    def __repr__(self) -> str:
        return f'Config("""\n{self._content}\n""")'

    def __deepcopy__(self, memo: Any) -> Self:
        return self.__class__(self.to_yaml())
