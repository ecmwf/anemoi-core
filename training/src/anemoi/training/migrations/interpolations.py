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
from typing import NamedTuple

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


class RelativeInterpolationError(Exception):
    """Relative interpolation are not supported."""


def count_leading(niddle: str, haystack: str) -> int:
    """Returns the numbre of times that "niddle" appears in haystack at the beginning.

    Parameters
    ----------
    niddle : str
        The characted to count
    haystack : str
        The content to search

    Returns
    -------
    int
        The count.
    """
    count = 0
    for c in haystack:
        if c == niddle:
            count += 1
        else:
            break
    return count


class Interpolation(NamedTuple):
    parts: tuple[str | int, ...]
    exact_ref: str  # used for relative interpolations


class InterpolationHandler:
    """Stores all interpolation references to easily update interpolations."""

    def __init__(self, ref_node: NodeContainer) -> None:
        self.ref_node = ref_node
        self.references: dict[tuple[str | int, ...], set[Interpolation]] = defaultdict(set)
        self.reverse_refs: dict[tuple[str | int, ...], set[Interpolation]] = defaultdict(set)

    def parse_config(self) -> None:
        self._parse_config_impl(self.ref_node, self.ref_node.prefix)

    def update(self, node: Node) -> None:
        self._parse_config_impl(node, node.prefix)

    def rename(self, old_parts: Sequence[str], target: str) -> None:
        """Changes the interpolation after renaming.

        Parameters
        ----------
        old_parts : Sequence[str]
            The original parts to rename from.
        target : str
            The target interpolation path.
        """
        changes: list[tuple[Node, str | int, str]] = []
        for reference, exact_ref in self.references[tuple(old_parts)]:
            node = self.ref_node.select(reference[:-1])
            changes.append(
                (node, reference[-1], replace_interpolation(node.get(reference[-1]).value, exact_ref, target))
            )
        # Updating the nodes after the previous for loop because node.set triggers
        # self._parse_node which updates self.references.
        for node, key, new_value in changes:
            node.set(key, new_value)

    def _parse_node(self, node: NodeContainer, prefix: Sequence[str | int], key: str | int):
        parts = (*prefix, key)

        # We fisrt clear old values before recomputing.
        # This happens when an interpolation node is updated.
        for existing_ref in self.reverse_refs[parts]:
            self.references[existing_ref.parts].remove(Interpolation(parts, existing_ref.exact_ref))
        del self.reverse_refs[parts]

        if node.is_interpolation(key):
            for interpo in get_interpolations(node.value[key]):
                if interpo.startswith("."):
                    # Handle the relative interpolation case
                    leading_dots = count_leading(".", interpo)
                    prefix_idx = -(leading_dots - 1) if leading_dots > 1 else None
                    interpo_parts = tuple(map(str, node.prefix[:prefix_idx]))
                    interpo_parts = (*interpo_parts, *interpo.lstrip(".").split("."))
                else:
                    interpo_parts = tuple(interpo.split("."))
                if not interpo.startswith(self.ref_node.prefix_str) or not self.ref_node.has_key(
                    interpo_parts[len(self.ref_node.prefix) :]
                ):
                    LOGGER.warning("%s uses missing interpolation %s.", ".".join(map(str, parts)), interpo)
                    continue
                self.references[interpo_parts].add(Interpolation(parts, interpo))
                self.reverse_refs[parts].add(Interpolation(interpo_parts, interpo))
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
