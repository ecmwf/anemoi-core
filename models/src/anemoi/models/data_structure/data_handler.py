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


class DataHandler(ABC):
    """Provides data from multiple datasets, data_groups.
    The data handler is used by the sample provider.

    The data handler keeps track of the data requests from the sample providers
    and provide a unified view of the data. This allows optimising data access.

    Data handlers should not be instantiated directly, but through the build function.
    """

    def __init__(self, **kwargs):
        self.registry = {}

    @abstractmethod
    def static(self, data_group: str) -> StaticDataDict:
        pass

    # @abstractmethod
    def __getitem__(self, i: int, data_group: str) -> DynamicDataDict:
        pass

    @abstractmethod
    def dates_block(self, data_group: str) -> _DatesBlock:
        pass

    @abstractmethod
    def register_request(self, container, data_group: str, variables: List[str], **kwargs):
        pass

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

    _cache: dict = {}

    def __init__(
        self, data_group: str, dataset: str, start, end, search_path: str | None = None, extra: dict | None = None
    ):
        super().__init__()
        if extra is None:
            extra = {}

        self._dataset = dataset
        self._search_path = search_path
        self.data_groups = [data_group]  # note that self.data_groups is a list
        self.extra = extra
        self.start = start
        self.end = end

    def register_request(self, container, data_group: str, variables, add_to_i, multiply_i):
        self.registry[container] = dict(
            container=container,
            ds=self.select(data_group, variables),
            data_group=data_group,
            variables=variables,
            add_to_i=add_to_i,
            multiply_i=multiply_i,
        )

    def group_requests(self, i):
        # context manager to group requests
        self._cache[i] = {}
        yield
        print(f"❌ Clearing cache for i={i}")
        del self._cache[i]

    def get_item(self, container, i: int) -> DynamicDataDict:
        assert container in self.registry, f"Container {container} not registered in {self.registry}"

        if not self._cache[i]:
            for p in self.registry:
                self._cache[i][p] = self._get_item(p, i)
            print(f"✅ Cache populated with {len(self._cache[i])} items for i={i}")

        print("❤️ Using cached item for", container)
        return self._cache[i][container]

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

    def dates_block(self, data_group: str) -> _DatesBlock:
        assert data_group in self.data_groups, f"Data_group {data_group} not in {self.data_groups}"
        # hack for now, need to update anemoi-datasets to provide missing
        missing = self._ds.missing if hasattr(self._ds, "missing") else []
        return _DatesBlock(self._ds.start_date, self._ds.end_date, self._ds.frequency, missing)

    @cached_property
    def _ds(self) -> object:  # returns a gridded anemoi-datasets object
        try:
            # expected use case : dataset name or full path
            return open_dataset(self._dataset, start=self.start, end=self.end)
        except ValueError as e:
            if self._search_path is not None:
                e.add_note(f"Also tried with search_path={self._search_path}")
                try:
                    # also expected use case : a config with dataset name and search_path defined in config
                    return open_dataset(f"{self._search_path}/{self._dataset}.zarr", start=self.start, end=self.end)
                except ValueError:
                    # less expected a config with dataset name with .zarr extension and search_path defined in config
                    return open_dataset(f"{self._search_path}/{self._dataset}", start=self.start, end=self.end)
            raise e

    def __repr__(self):
        if isinstance(self.extra, dict):
            extra = ", ".join(f"{k}={v}" for k, v in self.extra.items())
        else:
            extra = extra.__class__.__name__ + " " + str(self.extra)[:10]
        return f"{super().__repr__()}(dataset={self._dataset}, extra={extra})"

    def _tree(self, prefix=""):
        tree = _RichTree(prefix if prefix else "")
        tree.add(f"data_groups: {', '.join(self.data_groups)}")
        tree.add(f"dataset: {self._dataset}")
        if self.start:
            tree.add(f"start: {self.start}")
        if self.end:
            tree.add(f"end: {self.end}")
        if self.extra:
            tree.add(f"extra: {', '.join(f'{k}={v}' for k, v in self.extra.items())}")
        return tree


class GriddedDatasetDataHandler(OneDatasetDataHandler):

    def select(self, data_group: str, variables: List[str]):
        assert data_group in self.data_groups, f"Data_group {data_group} not in {self.data_groups}"
        return open_dataset(self._ds, select=variables)

    def static(self, data_group: str, variables: List[str]) -> StaticDataDict:
        ds = self.select(data_group, variables)
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
            extra=self.extra,
        )

    def _get_item(self, container, i: int) -> DynamicDataDict:
        request = self.registry[container]

        ds = request["ds"]
        add_to_i = request["add_to_i"]
        multiply_i = request["multiply_i"]
        j = i * multiply_i + add_to_i

        date = ds.dates[j]
        date = date.astype(datetime.datetime)  # remove this when anemoi-datasets returns consistent datetime objects

        return DynamicDataDict(
            data=ds[j],
            latitudes=ds.latitudes,
            longitudes=ds.longitudes,
            timedeltas=None,
            date_str=date.isoformat(),
        )


class RecordsDataHandler(OneDatasetDataHandler):
    def select(self, data_group: str, variables: List[str]):
        assert data_group in self.data_groups, f"Data_group {data_group} not in {self.data_groups}"

        variables = [f"{data_group}.{v}" for v in variables]
        return open_dataset(self._dataset, select=variables)

    def static(self, data_group: str, variables: List[str]) -> StaticDataDict:
        ds = self.select(data_group, variables)
        return StaticDataDict(
            name_to_index=ds.name_to_index[data_group],
            statistics=ds.statistics[data_group],
            statistics_tendencies=None,
            metadata=ds.metadata,
            resolution=None,
            supporting_arrays=None,
            variables=ds.variables[data_group],
            num_features=len(ds.variables[data_group]),
            dimensions=self.dimensions,
            extra=self.extra,
        )

    def _get_item(self, container, i: int) -> DynamicDataDict:
        request = self.registry[container]

        ds = request["ds"]
        data_group = request["data_group"]
        # variables = request["variables"] # will be used to select only some variables
        add_to_i = request["add_to_i"]
        multiply_i = request["multiply_i"]
        j = i * multiply_i + add_to_i

        timedeltas = ds[j].timedeltas[data_group]
        timedeltas = timedeltas.astype("timedelta64[s]").astype(np.int64)

        return DynamicDataDict(
            data=ds[j][data_group],
            latitudes=ds[j].latitudes[data_group],
            longitudes=ds[j].longitudes[data_group],
            timedeltas=timedeltas,
            date_str=ds.dates[j].isoformat(),
        )


class MultiDatasetDataHandler(DataHandler):
    """Uses several datahandles to provide access to multiple groups"""

    def __init__(self, *data_handlers):
        super().__init__()

        self.data_groups = []
        for dh in data_handlers:
            self.data_groups.extend(dh.data_groups)

        self._data_handlers = data_handlers

    def static(self, data_group, variables: List[str]) -> StaticDataDict:
        data_handler = self._find_data_handler(data_group)
        return data_handler.static(data_group, variables)

    def register_request(self, container, *, data_group, **kwargs):
        dh = self._find_data_handler(data_group)
        dh.register_request(container, data_group=data_group, **kwargs)

    def get_item(self, container, i: int, *, data_group):
        dh = self._find_data_handler(data_group)
        return dh.get_item(container, i)

    def dates_block(self, data_group):
        dh = self._find_data_handler(data_group)
        return dh.dates_block(data_group)

    def _find_data_handler(self, data_group):
        for dh in self._data_handlers:
            if data_group in dh.data_groups:
                return dh
        raise KeyError(f"Data_group {data_group} not found in any data_handler")

    def group_requests(self, i):
        # context manager to group requests
        ctx = [dh.group_requests(i) for dh in self._data_handlers]
        for c in ctx:
            next(c)
        yield

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
            ds = open_dataset(name)
            from anemoi.datasets.data.records import RecordsDataset

            if isinstance(ds, RecordsDataset):
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

    return _data_handler_factory(config["sources"], config.get("start"), config.get("end"), config.get("search_path"))
