# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import warnings
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import Any
from typing import List

import numpy as np
from omegaconf import DictConfig
from rich.console import Console
from rich.tree import Tree as _RichTree

from anemoi.datasets import open_dataset
from anemoi.models.data_structure.dicts import DynamicDataDict
from anemoi.models.data_structure.dicts import StaticDataDict
from anemoi.models.data_structure.dicts import _resolve_omega_conf_reference
from anemoi.models.data_structure.offsets import _DatesBlock

LOGGER = logging.getLogger(__name__)


def search_and_open_dataset(dataset, start, end, search_path):
    try:
        # expected use case : dataset name or full path
        return open_dataset(dataset, start=start, end=end)
    except ValueError as e:
        if search_path is None:
            raise e
        e.add_note(f"Also tried with search_path={search_path}")
        try:
            # also expected use case : a config with dataset name and search_path defined in config
            try:
                return open_dataset(f"{search_path}/{dataset}.zarr", start=start, end=end)
            except ValueError:
                return open_dataset(f"{search_path}/{dataset}.vz", start=start, end=end)
        except ValueError:
            # less expected a config with dataset name with .zarr extension and search_path defined in config
            return open_dataset(f"{search_path}/{dataset}", start=start, end=end)


class ReadPattern:
    def __init__(self, data_handler, container, group: str, variables, add_to_i, multiply_i):
        self.container = container
        self.group = group
        self.variables = variables
        self.add_to_i = add_to_i
        self.multiply_i = multiply_i
        self.data_handler = data_handler.find_data_handler(group)

        self._ds = self.data_handler.select(group, variables)

    def __call__(self, i) -> DynamicDataDict:
        j = i * self.multiply_i + self.add_to_i
        return dict(j=j, group=self.group, ds=self._ds)

    def __repr__(self):
        return f"ReadPattern({self.group=}, {self.variables=}, {self.add_to_i}, {self.multiply_i})"


class DataHandler(ABC):
    """Provides data from multiple datasets, groups.
    The data handler is used by the sample provider.

    The data handler keeps track of the data access patterns from the sample providers
    and provide a unified view of the data. This allows optimising data access.

    Data handlers should not be instantiated directly, but through the build function.
    """

    def __init__(self, **kwargs):
        self._read_pattern: dict[Any, ReadPattern] = {}
        self._loaded_data: dict = {}

    @abstractmethod
    def static(self, group: str) -> StaticDataDict:
        pass

    # @abstractmethod
    def __getitem__(self, *args, group: str) -> DynamicDataDict:
        pass

    @abstractmethod
    def dates_block(self, group: str) -> _DatesBlock:
        pass

    def register_read_pattern(self, container, group: str, /, variables, add_to_i, multiply_i):
        self._read_pattern[container] = ReadPattern(
            data_handler=self,
            container=container,
            group=group,
            variables=variables,
            add_to_i=add_to_i,
            multiply_i=multiply_i,
        )

    def grouped_read(self, *index):
        """Context manager to group read patterns in one call."""
        data = {}
        for c in self._read_pattern:
            kwargs = self._read_pattern[c](*index)
            data[c] = self.dynamic(**kwargs)
        return data

    def get_item(self, container, data) -> DynamicDataDict:
        return data[container]

    @abstractmethod
    def _tree(self, prefix=""):
        pass

    def __repr__(self):
        console = Console(record=True)
        tree = self._tree()
        with console.capture() as capture:
            console.print(tree, overflow="ellipsis")
        return capture.get()


class OneDatasetDataHandler(DataHandler):
    """Provide access to data for one dataset. A gridded dataset."""

    def __init__(
        self,
        data_group: str,
        dataset: str,
        start,
        end,
        search_path: str | None = None,
        extra_configs: dict | None = None,
    ):
        super().__init__()
        if extra_configs is None:
            extra_configs = {}

        self._dataset = dataset
        self._search_path = search_path
        self.groups = [data_group]  # note that self.groups is a list
        self.extra_configs = extra_configs
        self.start = start
        self.end = end

    @cached_property
    def dimensions(self) -> List[str]:
        # dimensions should be read from dataset when it is implemented in anemoi-datasets
        if hasattr(self._ds, "dimensions"):
            return self._ds.dimensions
        else:
            # hack for now, need to update anemoi-datasets to provide dimensions
            if hasattr(self._ds, "latitudes"):
                return ["variables", "ensembles", "values"]
            else:
                return ["variables", "values"]

    def dates_block(self, group: str) -> _DatesBlock:
        assert group in self.groups, f"group {group} not in {self.groups}"
        # hack for now, need to update anemoi-datasets to provide missing
        missing = self._ds.missing if hasattr(self._ds, "missing") else []
        return _DatesBlock(self._ds.start_date, self._ds.end_date, self._ds.frequency, missing)

    @cached_property
    def _ds(self) -> object:  # returns a gridded anemoi-datasets object
        return search_and_open_dataset(self._dataset, start=self.start, end=self.end, search_path=self._search_path)

    def __repr__(self):
        if isinstance(self.extra_configs, dict):
            extra = ", ".join(f"{k}={v}" for k, v in self.extra_configs.items())
        else:
            extra = extra.__class__.__name__ + " " + str(self.extra_configs)[:10]
        return f"{super().__repr__()}(dataset={self._dataset}, extra_configs={self.extra_configs})"

    def _tree(self, prefix=""):
        tree = _RichTree(prefix if prefix else "")
        tree.add(f"groups: {', '.join(self.groups)}")
        tree.add(f"dataset: {self._dataset}")
        if self.start:
            tree.add(f"start: {self.start}")
        if self.end:
            tree.add(f"end: {self.end}")
        if self.extra_configs:
            tree.add(f"extra_configs: {', '.join(f'{k}={v}' for k, v in self.extra_configs.items())}")
        return tree


class GriddedDatasetDataHandler(OneDatasetDataHandler):

    def select(self, group: str, variables: List[str]):
        assert group in self.groups, (group, self.groups)
        return open_dataset(self._ds, select=variables)

    def static(self, group: str, variables: List[str]) -> StaticDataDict:
        ds = self.select(group, variables)
        return StaticDataDict(
            name_to_index=ds.name_to_index,
            statistics=ds.statistics,
            statistics_tendencies=ds.statistics_tendencies,
            supporting_arrays=ds.supporting_arrays(),
            resolution=ds.resolution,
            metadata=ds.metadata,
            variables=ds.variables,
            num_features=len(ds.variables),
            dimensions=self.dimensions,
            extra_configs=self.extra_configs,
        )

    def dynamic(self, j, group, ds) -> DynamicDataDict:
        date = ds.dates[j]
        return DynamicDataDict(
            data=ds[j],
            latitudes=ds.latitudes,
            longitudes=ds.longitudes,
            timedeltas=None,
            date_str=date.astype(datetime.datetime).isoformat(),
        )


class RecordsDataHandler(OneDatasetDataHandler):
    def select(self, group: str, variables: List[str]):
        assert group in self.groups, (group, self.groups)

        if isinstance(self._dataset, dict):
            ds = open_dataset(**self._dataset, select=[f"{group}.{v}" for v in variables])
        else:
            ds = open_dataset(self._dataset, select=[f"{group}.{v}" for v in variables])

        return ds

    def static(self, group: str, variables: List[str]) -> StaticDataDict:
        ds = self.select(group, variables)
        return StaticDataDict(
            name_to_index=ds.name_to_index[group],
            statistics=ds.statistics[group],
            statistics_tendencies=None,
            metadata=ds.metadata,
            resolution=None,
            supporting_arrays=None,
            variables=ds.variables[group],
            num_features=len(ds.variables[group]),
            dimensions=self.dimensions,
            extra_configs=self.extra_configs,
        )

    def dynamic(self, j, group, ds) -> DynamicDataDict:
        timedeltas = ds[j].timedeltas[group]
        timedeltas = timedeltas.astype("timedelta64[s]").astype(np.int64)

        return DynamicDataDict(
            data=ds[j][group],
            latitudes=ds[j].latitudes[group],
            longitudes=ds[j].longitudes[group],
            timedeltas=timedeltas,
            date_str=ds.dates[j].astype(datetime.datetime).isoformat(),
        )


class MultiDatasetDataHandler(DataHandler):
    """Uses several datahandles to provide access to multiple groups"""

    def __init__(self, *data_handlers):
        super().__init__()
        self._data_handlers = data_handlers
        self.groups = [g for dh in data_handlers for g in dh.groups]

    def static(self, group, variables: List[str]) -> StaticDataDict:
        return self.find_data_handler(group).static(group, variables)

    def dynamic(self, /, group, **kwargs) -> DynamicDataDict:
        return self.find_data_handler(group).dynamic(group=group, **kwargs)

    def dates_block(self, group):
        return self.find_data_handler(group).dates_block(group)

    def find_data_handler(self, group):
        for dh in self._data_handlers:
            if group in dh.groups:
                return dh
        raise KeyError(f"group {group} not found in any data_handler {self._data_handlers}")

    def _tree(self, prefix=""):
        # for pretty printing of dicts
        name = prefix if prefix else ""
        tree = _RichTree(name)
        for i, dh in enumerate(self._data_handlers):
            tree.add(dh._tree(prefix=f"{i}"))
        return tree


def _data_handler_factory(sources: dict, start, end, search_path) -> DataHandler:
    data_handlers = []

    for cfg in sources:
        # if not defined in cfg, use the provided global search_path, start, end
        cfg["search_path"] = cfg.get("search_path", search_path)
        cfg["start"] = cfg.get("start", start)
        cfg["end"] = cfg.get("end", end)
        if "dataset" not in cfg:
            raise ValueError("Each source in data_handler config must contain a 'dataset' key")

        def find_class(name: str):
            # For now, just check for known dataset types
            # but anemoi-datasets should provide a way to get the right class
            # such as ds.is_gridded or ds.is_grouped
            ds = search_and_open_dataset(name, start=cfg["start"], end=cfg["end"], search_path=cfg["search_path"])
            from anemoi.datasets.data.records import BaseRecordsDataset

            if isinstance(ds, BaseRecordsDataset):
                return RecordsDataHandler
            return GriddedDatasetDataHandler

        try:
            cls = find_class(cfg["dataset"])
            dh = cls(**cfg)
            data_handlers.append(dh)
        except Exception as e:
            warnings.warn(f"Failed to create data handler for {cfg['dataset']}: {e}")

    return MultiDatasetDataHandler(*data_handlers)


def build_data_handler(config: dict, /, kind: str | None) -> DataHandler:
    initial_config = config.copy()
    config.pop("aliases", None)  # remove aliases if present

    if kind not in ["training", "validation", "test", None]:
        raise ValueError(f"build_data_handler: kind={kind} must be one of ['training', 'validation', 'test', None]")
    if "sources" not in config:
        raise ValueError("build_data_handler: config must contain 'sources' key")
    if kind and kind not in config:
        raise ValueError(f"build_data_handler: config must contain '{kind}' key")

    if isinstance(config, DictConfig):
        # found an omegaconf DictConfig, resolve it first
        config = _resolve_omega_conf_reference(config)
    config = config.copy()

    if kind:
        overwrite_config = config[kind]
        for k, v in overwrite_config.items():
            if k in config:
                LOGGER.debug(f"Overriding config.{k} with config.{kind}.{k}")
            config[k] = v

    dh = _data_handler_factory(config["sources"], config.get("start"), config.get("end"), config.get("search_path"))

    dh._initial_config = initial_config
    return dh
