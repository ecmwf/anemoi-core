# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Configuration utilities for handling dataset-specific configurations."""

from collections.abc import Mapping

from anemoi.models.config_types import to_container

DEFAULT_DATASET_NAME = "data"


def get_multiple_datasets_config(config: Mapping, default_dataset_name: str = DEFAULT_DATASET_NAME) -> dict:
    """Get multiple datasets configuration for old configs.
    Use /'data/' as the default dataset name.
    """
    if "datasets" in config:
        datasets = config["datasets"] if isinstance(config, Mapping) else getattr(config, "datasets")
        return to_container(datasets)

    return {default_dataset_name: to_container(config)}
