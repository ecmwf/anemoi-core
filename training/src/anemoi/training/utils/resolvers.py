# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from omegaconf import DictConfig
from omegaconf import OmegaConf


def dataset_metadata(dataset_config: DictConfig | str) -> dict[str, Any]:
    """Return the per-variable metadata of a dataset, for use in the config.

    Lets model components that describe variables by their physical meaning (e.g. a
    variable tokenizer) be configured with the dataset's metadata, so the model itself
    never needs to know where data comes from. Only the metadata is read, not the data.

    Usage in a config::

        variables_metadata: ${dataset_metadata:${dataloader.training.datasets.data.dataset_config}}
    """
    from anemoi.datasets import open_dataset

    if isinstance(dataset_config, DictConfig):
        dataset_config = OmegaConf.to_container(dataset_config, resolve=True)
    return open_dataset(dataset_config).variables_metadata


def register_resolvers() -> None:
    if not OmegaConf.has_resolver("dataset_metadata"):
        OmegaConf.register_new_resolver("dataset_metadata", dataset_metadata)
