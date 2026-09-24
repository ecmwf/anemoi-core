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
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Self

import yamlrocks
from omegaconf import OmegaConf

from anemoi.training.migrations.interpolations import InterpolationHandler
from anemoi.training.migrations.nodes import NodeDict

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anemoi.training.migrations.migrator import ConfigMigration

_START_SUMMARY_CONTENT = "==== MIGRATION SUMMARY ===="
_END_SUMMARY_CONTENT = "======= END SUMMARY ======="


def _split_summary_comment(comment_lines: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    pre_summary: list[str] = []
    summary: list[str] = []
    post_summary: list[str] = []
    state: Literal["PRE", "SUMMARY", "POST"] = "PRE"
    for comment_line in comment_lines:
        if comment_line == _START_SUMMARY_CONTENT:
            state = "SUMMARY"
        elif comment_line == _END_SUMMARY_CONTENT:
            state = "POST"

        if state == "PRE":
            pre_summary.append(comment_line)
        elif state == "SUMMARY":
            summary.append(comment_line)
        else:
            post_summary.append(comment_line)
    return pre_summary, summary, post_summary


class Config(NodeDict):
    """The entry point for the config tree.

    This is a proxy for a NodeDict that can be initialized via a config content.
    """

    def __init__(self, content: str, update_summary: bool = False) -> None:
        self._content = content
        self._cfg = OmegaConf.create(self._content)
        self._migration: ConfigMigration | None = None
        self._update_summary = update_summary

        self._interpolation_handler = InterpolationHandler(self)
        self._interpolation_handler.parse_config()

    @classmethod
    def from_path(cls, path: Path | str, update_summary: bool = False) -> Self:
        """Create the config from its path of the filesystem.

        Parameters
        ----------
        path : Path | str
            The path to the yaml config file
        update_summary : bool, default False
            Whether to add comments in the config when calling ``add_summary``.

        Returns
        -------
        Self
            The Config instance.
        """
        content = Path(path).read_text()
        return cls(content, update_summary)

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

    def set_migration(self, migration: ConfigMigration) -> None:
        self._migration = migration

    @property
    def parent(self) -> NodeDict:
        # The parent of the root node is itself.
        return self

    def add_summary(self, content: str) -> None:
        """Add some information about the goal of the migration script.

        The config comment is only changed if ``update_summary`` has been passed.

        Parameters
        ----------
        content : str
            The summary to add to the config.
        """
        if not self._update_summary:
            return
        pre_summary: list[str] = []
        summary: list[str] = []
        end_summary: list[str] = []
        post_summary: list[str] = []
        if self.yaml_node.comment_before is not None:
            pre_summary, summary, post_summary = _split_summary_comment(self.yaml_node.comment_before.split("\n"))

        if not len(summary):
            summary.extend(["", _START_SUMMARY_CONTENT, ""])
            end_summary.append(_END_SUMMARY_CONTENT)
            if len(post_summary):
                end_summary.append("")

        if self._migration is not None:
            summary.append(f"{self._migration.name}")
            summary.extend(["-" * len(summary[-1]), ""])
        summary.append(content)
        summary.append("")
        comment_before = [*pre_summary, *summary, *end_summary, *post_summary]
        self.yaml_node.comment_before = "\n".join(comment_before)

    def to_yaml(self) -> str:
        """Export the config into yaml."""
        return self.yaml.to_yaml().decode()

    def __repr__(self) -> str:
        return f'Config("""\n{self._content}\n""")'

    def __deepcopy__(self, memo: Any) -> Self:
        return self.__class__(self.to_yaml(), self._update_summary)
