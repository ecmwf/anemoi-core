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
from typing import Any

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


class InterpolationReferences:
    """Stores all interpolation references to easily update interpolations."""

    def __init__(self) -> None:
        self.references: dict[str, set[Sequence[Any]]] = defaultdict(set)

    def parse_node(
        self,
        node: NodeContainer,
    ) -> None:
        self._parse_node_impl(node, node, node.prefix)

    def _parse_node_impl(
        self,
        ref_node: NodeContainer,
        node: Node,
        prefix: Sequence[Any],
    ) -> None:
        if not isinstance(node, NodeContainer):
            return

        if isinstance(node, NodeDict):
            iterator = node
        elif isinstance(node, NodeList):
            iterator = range(len(node))

        for k in iterator:
            new_prefix = (*prefix, k)
            if node.is_interpolation(k):
                for interpo in get_interpolations(node.value[k]):
                    interpo_parts = interpo.split(".")
                    if interpo_parts[: len(ref_node.prefix)] != list(ref_node.prefix) or not ref_node.has_key(
                        interpo_parts[len(ref_node.prefix) :]
                    ):
                        LOGGER.warning("%s uses missing interpolation %s.", ".".join(map(str, new_prefix)), interpo)
                        continue
                    self.references[interpo].add(new_prefix)
            elif isinstance(node[k], NodeContainer):
                self._parse_node_impl(ref_node, node[k], new_prefix)
