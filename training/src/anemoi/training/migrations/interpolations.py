# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING
from typing import NamedTuple

from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import Node
from omegaconf import OmegaConf
from omegaconf.grammar_parser import parse
from omegaconf.grammar_visitor import GrammarVisitor

if TYPE_CHECKING:
    from anemoi.training.migrations.config import Config

INTERPOLATION_PATTERN = re.compile(r"\$\{([^}]*)\}", flags=re.ASCII)


def get_interpolations(value: str) -> list[str]:
    interpolations: list[str] = []

    def node_interpolation_callback(inter_key: str, _) -> Node | None:
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


class InterpolationReference(NamedTuple):
    prefix: str
    config_name: str
    key: str


class InterpolationReferences:
    """Stores all interpolation references to easily update interpolations."""

    def __init__(self) -> None:
        self.references: dict[str, set[InterpolationReference]] = defaultdict(set)

    def parse_config(
        self,
        config: Config,
        prefix: str | None = None,
    ) -> None:
        self._parse_config_impl(config, config.cfg, prefix)

    def _parse_config_impl(
        self,
        config: Config,
        cfg: DictConfig | ListConfig,
        prefix: str | None = None,
    ) -> None:
        prefix = prefix or ""
        raw_cfg = OmegaConf.to_container(cfg, resolve=False)
        if raw_cfg is None:
            return
        if isinstance(cfg, DictConfig):
            iterator = cfg.keys()
        else:
            iterator = range(len(raw_cfg))
        for k in iterator:
            # Check that cfg[k] is not a str before resolving it to avoid interpolation errors
            # as we only load the config file by file.
            if not isinstance(raw_cfg[k], str) and isinstance(cfg[k], (DictConfig, ListConfig)):
                self._parse_config_impl(config, cfg[k], f"{prefix}.{k}")
            if not isinstance(k, (int, str)):
                continue
            elif OmegaConf.is_interpolation(cfg, k):
                # This should be safe because in case raw_cfg is a list, k is an int
                # as it comes from the enumerate branch above.
                for interpo in get_interpolations(raw_cfg[k]):  # ty: ignore[invalid-argument-type]
                    self.references[interpo].add(
                        InterpolationReference(config.prefix, config._path.name, f"{prefix}.{k}".removeprefix("."))
                    )
