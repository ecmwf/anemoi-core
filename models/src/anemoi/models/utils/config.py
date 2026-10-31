# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Configuration utilities for handling dataset-specific configurations."""

from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME

COORDS_DIM = 4


def _to_primitive(value: object) -> object:
    """Copy dict subclasses (e.g. ``DotDict``) and tuples into plain dicts and lists.

    ``OmegaConf.create`` rejects dict subclasses nested inside a container, so a legacy
    config section that arrives as a ``DotDict`` must be converted first. OmegaConf
    containers are not ``dict`` instances and pass through unchanged.
    """
    if isinstance(value, dict):
        return {key: _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_primitive(item) for item in value]
    return value


# This function retrieves the configuration for multiple datasets, supporting both new and old config formats.
# Its location in the codebase may be revisited in the near future.
def get_multiple_datasets_config(config: DictConfig, default_dataset_name: str = DEFAULT_DATASET_NAME) -> dict:
    """Get multiple datasets configuration for old configs.
    Use /'data/' as the default dataset name.
    """
    if "datasets" in config:
        if isinstance(config, dict):
            return config["datasets"]
        return config.datasets

    return OmegaConf.create({default_dataset_name: _to_primitive(config)})
