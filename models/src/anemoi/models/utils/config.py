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

DEFAULT_DATASET_NAME = "data"


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

    return OmegaConf.create({default_dataset_name: config})
