import os

import pytest
import yaml
from omegaconf import DictConfig


@pytest.fixture
def new_config() -> DictConfig:
    return DictConfig(
        {
            "data": {
                "frequency": "6h",
                "timestep": "6h",
                "data_handlers": {
                    "era5": {
                        "dataset": {
                            # "set_group": "era5",
                            "dataset": "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
                        },
                        "processors": {
                            "normalizer": {
                                "_target_": "anemoi.models.preprocessing.normalizer.InputNormalizer",
                                "config": {"default": "mean-std", "std": ["tp"]},
                            },
                        },
                    },
                    "amsr2": {
                        ## NOW: Only 1 group is supported for each "key" (dh)
                        "dataset": {
                            "dataset": "/etc/ecmwf/nfs/dh1_home_a/mafp/work/obs/data/vz/obs-2018-11.vz",
                            "select": ["amsr_h180.*"],
                        },
                    },
                },
            },
            "dataloader": {
                "sampler": {
                    "training": {
                        "start": 2018,
                        "end": 2019,
                    },
                    "validation": {
                        "start": 2019,
                        "end": 2020,
                    },
                },
                "prefetch_factor": 2,
                "pin_memory": True,
                "persistent_workers": True,
                "grid_indices": {"_target_": "anemoi.training.data.grid_indices.FullGrid", "nodes_name": "data"},
                "read_group_size": 1,
                "num_workers": {
                    "training": 8,
                    "validation": 8,
                },
                "batch_size": {
                    "training": 2,
                    "validation": 4,
                },
                "limit_batches": {
                    "training": None,
                    "validation": None,
                },
            },
            "model": {
                "sample": {
                    "input": {
                        "era5": {
                            "variables": [
                                "cos_latitude",
                                "sin_latitude",
                                "10u",
                                "2t",
                                "2d",
                                "q_100",
                                "q_1000",
                            ],
                            "timedeltas": ["-6h", "0h"],
                        },
                        "amsr2": {
                            "variables": ["rawbt_1", "rawbt_2", "rawbt_3"],
                            "timedeltas": ["0h"],
                        },
                    },
                    "target": {
                        "era5": {
                            "variables": ["10u", "10v", "2t", "q_1000", "tp"],
                            "timedeltas": ["6h"],
                        },
                        "amsr2": {
                            "variables": ["rawbt_1", "rawbt_2", "rawbt_3"],
                            "timedeltas": ["6h"],
                        },
                    },
                },
                "model": {"_target_": "anemoi.models.models.AnemoiMultiModel"},
                "encoder": {"_target_": "anemoi.models.layers.mapper.GraphTransformerForwardMapper"},
                "processor": {"_target_": "anemoi.models.layers.processor.GraphTransformerProcessor"},
                "decoder": {"_target_": "anemoi.models.layers.mapper.GraphTransformerBackwardMapper"},
            },
            "training": {
                "rollout": {
                    "start": 1,
                    "epoch_increment": 0,
                    "max": 1,
                },
            },
        },
    )


@pytest.fixture
def downscaling_config() -> DictConfig:
    config_path = os.path.join(os.path.dirname(__file__), "downscaling.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
