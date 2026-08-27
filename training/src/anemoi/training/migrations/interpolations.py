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
import re
from collections import defaultdict
from collections.abc import Sequence

from omegaconf import Node as OGNode
from omegaconf.grammar_parser import parse
from omegaconf.grammar_visitor import GrammarVisitor

from anemoi.training.migrations.nodes import Node
from anemoi.training.migrations.nodes import NodeContainer
from anemoi.training.migrations.nodes import NodeDict
from anemoi.training.migrations.nodes import NodeList

LOGGER = logging.getLogger(__name__)

INTERPOLATION_PATTERN = re.compile(r"\$\{([^}]*)\}", flags=re.ASCII)


def get_interpolations(value: str) -> list[str]:
    interpolations: list[str] = []

    def node_interpolation_callback(inter_key: str, _) -> OGNode | None:
        interpolations.append(inter_key)

    def resolver_interpolation_callback(*_args, **_kwargs) -> None:
        pass

    visitor = GrammarVisitor(node_interpolation_callback, resolver_interpolation_callback, None)
    parse_tree = parse(value)
    visitor.visit(parse_tree)
    return interpolations


def replace_interpolation(value: str, interpo: str, replace: str) -> str:
    for match in INTERPOLATION_PATTERN.finditer(value):
        if match.group(1).strip() == interpo:
            start, end = match.span()
            value = value[:start] + f"${{{replace}}}" + value[end:]
    return value


class InterpolationHandler:
    """Stores all interpolation references to easily update interpolations."""

    def __init__(self, ref_node: NodeContainer) -> None:
        self.ref_node = ref_node
        self.references: dict[Sequence[str], set[Sequence[str | int]]] = defaultdict(set)
        self.reverse_refs: dict[Sequence[str | int], set[Sequence[str]]] = defaultdict(set)

    def parse_config(self) -> None:
        self._parse_config_impl(self.ref_node, self.ref_node.prefix)

    def update(self, node: Node) -> None:
        self._parse_config_impl(node, node.prefix)

    def _parse_node(self, node: NodeContainer, prefix: Sequence[str | int], key: str | int):
        parts = (*prefix, key)

        # We fisrt clear old values before recomputing.
        # This happens when an interpolation node is updated.
        for existing_ref in self.reverse_refs[parts]:
            self.references[existing_ref].remove(parts)
        del self.reverse_refs[parts]

        if node.is_interpolation(key):
            for interpo in get_interpolations(node.value[key]):
                interpo_parts = tuple(interpo.split("."))
                if not interpo.startswith(self.ref_node.prefix_str) or not self.ref_node.has_key(
                    interpo_parts[len(self.ref_node.prefix) :]
                ):
                    LOGGER.warning("%s uses missing interpolation %s.", ".".join(map(str, parts)), interpo)
                    continue
                self.references[interpo_parts].add(parts)
                self.reverse_refs[parts].add(interpo_parts)
        elif isinstance(node[key], NodeContainer):
            self._parse_config_impl(node[key], parts)

    def _parse_config_impl(
        self,
        node: Node,
        prefix: Sequence[str | int],
    ) -> None:
        if not isinstance(node, NodeContainer):
            return

        if isinstance(node, NodeDict):
            for key in node:
                self._parse_node(node, prefix, key)
        elif isinstance(node, NodeList):
            for key in range(len(node)):
                self._parse_node(node, prefix, key)
