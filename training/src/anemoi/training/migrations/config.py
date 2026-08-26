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

import yamlrocks
from omegaconf import OmegaConf

from anemoi.training.migrations.nodes import NodeDict


class Config(NodeDict):
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._cfg = OmegaConf.load(self._path)

    @property
    def prefix(self) -> tuple[()]:
        return ()

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
    def parent(self) -> NodeDict:
        return self

    def _is_key_valid(self, key: Any) -> bool:
        return key in self.cfg

    def to_yaml(self) -> str:
        return self.yaml.to_yaml().decode()

    def __repr__(self) -> str:
        return f"Config({self._path})"
