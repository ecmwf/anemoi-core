# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
from collections.abc import Mapping
from collections.abc import Sequence

from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)


def is_mapping(node: object) -> bool:
    """Return True for dict-like config nodes, including OmegaConf ``DictConfig``.

    ``DictConfig`` is not a subclass of ``dict``/``Mapping``, so both are matched.
    """
    return OmegaConf.is_dict(node) or isinstance(node, Mapping)


def is_sequence(node: object) -> bool:
    """Return True for list-like config nodes, including OmegaConf ``ListConfig`` (excluding ``str``).

    ``ListConfig`` is not a subclass of ``list``/``Sequence``, so both are matched.
    """
    return (OmegaConf.is_list(node) or isinstance(node, Sequence)) and not isinstance(node, str)


def is_container(node: object) -> bool:
    """Return True for any mapping or sequence config node."""
    return is_mapping(node) or is_sequence(node)


def resolve_subgrid_node(node: object, output_mask: object, dataset_name: str) -> None:
    """Recursively replace ``subgrid: output_mask`` placeholders with the mask tuple."""
    if is_mapping(node):
        for k, v in node.items():
            if is_container(v):
                resolve_subgrid_node(v, output_mask, dataset_name)
            elif (k, v) == ("subgrid", "output_mask"):
                node[k] = output_mask.as_tuple()
                LOGGER.info("Resolved subgrid for dataset '%s' to output_mask as tuple: %s", dataset_name, node[k])
    elif is_sequence(node):
        for item in node:
            resolve_subgrid_node(item, output_mask, dataset_name)


def resolve_subgrid(config: Mapping, output_mask: Mapping) -> None:
    """Resolve ``subgrid: output_mask`` placeholders for every dataset in ``config``."""
    for dataset_name, dataset_config in config.items():
        if dataset_config is not None:
            resolve_subgrid_node(dataset_config, output_mask[dataset_name], dataset_name)
