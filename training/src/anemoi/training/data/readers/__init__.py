# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from omegaconf import DictConfig

from anemoi.training.data.readers.gridded import GriddedDataReader
from anemoi.training.data.readers.protocol import DataReader
from anemoi.training.data.readers.tabular import TabularDataReader
from anemoi.training.data.readers.trajectory import TrajectoryDataReader
from anemoi.training.data.readers.utils import _as_dict
from anemoi.training.data.readers.utils import _normalize_reader_config

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DataReader",
    "GriddedDataReader",
    "TabularDataReader",
    "TrajectoryDataReader",
    "create_reader",
]


def create_reader(config: dict | DictConfig) -> DataReader:
    """Create the data reader for one dataset entry of the dataloader configuration.

    The reader is chosen from the kind of data, inferred from ``dataset_config`` (tabular
    datasets set a ``window`` and a ``frequency``, all others are gridded), and from whether
    the entry has a ``trajectory`` block.
    """
    reader_config = _normalize_reader_config(config)
    trajectory_config = _as_dict(reader_config.pop("trajectory", None))

    if trajectory_config is not None:
        sampling = _as_dict(trajectory_config.get("sampling")) if isinstance(trajectory_config, dict) else None
        LOGGER.info("Creating TrajectoryDataReader...")
        return TrajectoryDataReader(**reader_config, sampling=sampling)

    if "window" in reader_config["dataset_config"] and "frequency" in reader_config["dataset_config"]:
        LOGGER.info("Creating TabularDataReader...")
        return TabularDataReader(**reader_config)

    LOGGER.info("Creating a GriddedDataReader...")
    return GriddedDataReader(**reader_config)
