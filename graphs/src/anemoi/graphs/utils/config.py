# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from omegaconf import DictConfig
from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "data"

def get_multiple_datasets_config(config: DictConfig, default_dataset_name: str = DEFAULT_DATASET_NAME) -> dict:
    """Get multiple datasets configuration for old configs.
    Use /'data/' as the default dataset name.
    """
    if "datasets" in config:
        if isinstance(config, dict):
            return config["datasets"]
        return config.datasets

    return OmegaConf.create({default_dataset_name: config})

def integrate_data_nodes_in_config(config: DictConfig):
        
    # Introduce Data Nodes in graph config
    train_configs = get_multiple_datasets_config(config.dataloader.training)

    val_configs = {}
    if hasattr(config.dataloader, "validation"):
        val_configs = get_multiple_datasets_config(config.dataloader.validation)

    test_configs = {}
    if hasattr(config.dataloader, "test"):
        test_configs = get_multiple_datasets_config(config.dataloader.test)

    dataset_configs = {
        **train_configs,
        **val_configs,
        **test_configs,
    }

    for dataset_name, dataset_config in dataset_configs.items():
        if dataset_name not in config.graph.nodes:
            LOGGER.info("Creating graph node entry for dataset '%s'", dataset_name)
            dataset_reader_config = dataset_config.dataset_config
            if isinstance(dataset_reader_config, (DictConfig, dict)):
                if "dataset" not in dataset_reader_config:
                    msg = f"Dataset '{dataset_name}' is missing 'dataset' key."
                    raise ValueError(msg)
                dataset_source = dataset_reader_config["dataset"]
            else:
                dataset_source = dataset_reader_config

            if dataset_source is None:
                msg = (
                    f"Dataset source is None for dataset '{dataset_name}'. Check dataloader.dataset_config.dataset."
                )
                raise ValueError(msg)

            # Add dataset nodes from dataloader into graph recepe
            config.graph.nodes[dataset_name] = {
                "node_builder": {"_target_": "anemoi.graphs.nodes.AnemoiDatasetNodes", "dataset": dataset_source},
                "attributes": config.graph.attributes.nodes,
            }
        else:
            LOGGER.info("Graph node entry for dataset '%s' is already specified in the config.", dataset_name)
    
    return config