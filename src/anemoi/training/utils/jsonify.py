import datetime
from pathlib import Path

import torch
from anemoi.models.data_indices.collection import BaseIndex
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.data_indices.tensor import BaseTensorIndex
from omegaconf import DictConfig
from omegaconf import ListConfig


def map_config_to_primitives(config):
    """Ensure that the metadata information is JSON-serializable."""
    if config is None:
        return None

    if isinstance(config, dict):
        return {k: map_config_to_primitives(v) for k, v in config.items()}

    if isinstance(config, (list, tuple)):
        return [map_config_to_primitives(v) for v in config]

    if isinstance(config, (int, float, str, bool)):
        return config

    if isinstance(config, Path):
        return str(config)

    if isinstance(config, DictConfig):
        return map_config_to_primitives(dict(config))

    if isinstance(config, ListConfig):
        return map_config_to_primitives(list(config))

    if isinstance(config, datetime.date):
        return config.isoformat()

    if isinstance(config, torch.Tensor):
        return map_config_to_primitives(config.tolist())

    if isinstance(config, IndexCollection):
        return map_config_to_primitives(config.todict())

    if isinstance(config, BaseTensorIndex):
        return map_config_to_primitives(config.todict())

    if isinstance(config, BaseIndex):
        return map_config_to_primitives(config.todict())

    raise ValueError(f"Cannot serialize object of type {type(config)}")
