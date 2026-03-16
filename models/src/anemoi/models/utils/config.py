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


def broadcast_config_keys(dictionary: dict[str, int], **kwargs) -> dict[str, int]:
    """Broadcasts values from the input dictionary to multiple keys based on the provided mapping.

    Parameters
    ----------
    dictionary : dict[str, int]
        Input dictionary containing values to be broadcasted
    **kwargs : dict[str, str | list[str]]
        Mapping of old keys to new keys for broadcasting. Each key in kwargs is an old key from the input dictionary,
        and its value is a list of new keys.

    Returns
    -------
    dict[str, int]
        New dictionary with values broadcasted to the new keys based on the provided mapping.

    Example
    -------
    >>> input_dict = {'num_params': 10}
    >>> broadcast_config_keys(input_dict, num_params='dataset1_num_params') # Broadcast to a single new key
    {'dataset1_num_params': 10}
    >>> new_keys = ['dataset1_num_params', 'dataset2_num_params']
    >>> broadcast_config_keys(input_dict, num_params=new_keys)  # Broadcast to multiple new keys
    {'dataset1_num_params': 10, 'dataset2_num_params': 10}
    """
    new_num_params = {}
    for old_key, new_keys in kwargs.items():
        if isinstance(new_keys, str):
            new_keys = [new_keys]

        for new_key in new_keys:
            if old_key not in dictionary:
                raise KeyError(f"Key '{old_key}' not found in the input dictionary for broadcasting.")

            new_num_params[new_key] = dictionary[old_key]

    return new_num_params
