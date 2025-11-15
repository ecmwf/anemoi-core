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
import time
from collections import defaultdict
from contextlib import contextmanager
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


@contextmanager
def timer(label="Block"):
    t = time.time()
    try:
        yield
    finally:
        t_elapsed = 1000 * (time.time() - t)
        if LOGGER.isEnabledFor(logging.DEBUG):
            print(f"{label} in {t_elapsed:.3f} ms")


def search_and_open_dataset(dataset, start, end, search_path, data_group=None):
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


class OneDatasetDataHandler:
    """Provide access to data for one dataset, wich contains potientally multiple groups."""

    def __init__(
        self,
        dataset: str,
        start=None,
        end=None,
        search_path: str | None = None,
        extra_configs: dict | None = None,
        data_group: str | None = None,
    ):
        if extra_configs is None:
            extra_configs = {}
        self._static_caches = defaultdict(dict)

        self._search_path = search_path
        self._start = start
        self._end = end

        ds = search_and_open_dataset(dataset, start=start, end=end, search_path=search_path, data_group=data_group)
        if data_group is not None:
            ds = open_dataset(ds, set_group=data_group)
        self._ds = ds
        self.extra_configs = extra_configs

        self.groups = self._ds.groups

        # for display purposes, it shoud read from self._ds.datasets
        name = dataset
        while isinstance(name, dict):
            name = name.get("dataset", "unknown_dataset")
        while "/" in name:
            name = name.split("/")[-1]
        while "." in name:
            name = name.split(".")[0]
        self._dataset_name = name

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

    def __repr__(self):
        console = Console(record=True)
        tree = self._tree()
        with console.capture() as capture:
            console.print(tree, overflow="ellipsis")
        return capture.get()

    def _tree(self, prefix=""):
        tree = _RichTree(prefix if prefix else "")
        tree.add(f"groups: {self.groups}")
        tree.add(f"dataset: {self._dataset_name}")
        tree.add(f"start: {self._ds.start_date}")
        tree.add(f"end: {self._ds.end_date}")
        if self.extra_configs:
            tree.add(f"extra_configs: {', '.join(f'{k}={v}' for k, v in self.extra_configs.items())}")
        return tree

    def _selected(self, group: str, variables: List[str]):
        if (group, tuple(variables)) not in self._static_caches:
            self._static_caches[(group, tuple(variables))] = cached = {}

            full_name_to_index = self._ds.name_to_index[group]
            variables_index = np.array([full_name_to_index[v] for v in variables])
            cached["variables_index"] = variables_index
            cached["name_to_index"] = {v: i for i, v in enumerate(variables)}
            cached["dates"] = self._ds.dates
            cached["dimensions"] = self.dimensions

            statistics = self._ds.statistics[group]
            statistics = {k: v[variables_index] for k, v in statistics.items()}
            cached["statistics"] = statistics

            cached["statistics_tendencies"] = "not implemented yet"
            cached["supporting_arrays"] = (
                self._ds.supporting_arrays() if hasattr(self._ds, "supporting_arrays") else None
            )
            cached["resolution"] = self._ds.resolution if hasattr(self._ds, "resolution") else None
            cached["num_features"] = len(variables)

        return self._static_caches[(group, tuple(variables))]

    def static(self, *requests) -> StaticDataDict:
        assert len(requests) == 1, "OneDatasetDataHandler.static only accepts one request at a time"
        group, variables = requests[0]
        cache = self._selected(group, variables)

        res = StaticDataDict(
            name_to_index=cache["name_to_index"],
            statistics=cache["statistics"],
            statistics_tendencies=cache["statistics_tendencies"],
            supporting_arrays=cache["supporting_arrays"],
            resolution=cache["resolution"],
            metadata=None,  # self._ds.metadata,
            variables=variables,
            num_features=cache["num_features"],
            dimensions=cache["dimensions"],
            extra_configs=self.extra_configs,
        )
        return res

    def dynamic(self, *requests) -> DynamicDataDict:
        assert len(requests) == 1, "OneDatasetDataHandler.dynamic only accepts one request at a time"
        j, group, variables = requests[0]
        assert isinstance(j, int), j

        cache = self._selected(group, variables)

        with timer(f"  From {self._dataset_name} [{j}] {len(variables)} variables from {group}"):
            record = self._ds[j]
            data = record[group]

        # apply variable selection according to the dimensions 'variables'
        var_axis = self.dimensions.index("variables")
        index = [slice(None)] * data.ndim
        index[var_axis] = cache["variables_index"]
        data = data[tuple(index)]

        return DynamicDataDict(
            data=data,
            latitudes=record.latitudes[group],
            longitudes=record.longitudes[group],
            timedeltas=record.timedeltas[group],
            date_str=cache["dates"][j].astype(datetime.datetime).isoformat(),
        )


class DataHandler:
    """Uses several datahandles to provide access to multiple groups"""

    def __init__(self, *sources):
        data_handlers = [OneDatasetDataHandler(**cfg) for cfg in sources]

        self.data_handlers = {}
        for dh in data_handlers:
            for g in dh.groups:
                if g in self.data_handlers:
                    raise KeyError(f"Duplicate group '{g}' found in data handlers")
                self.data_handlers[g] = dh

    def dispatch(self, group: str) -> OneDatasetDataHandler:
        try:
            return self.data_handlers[group]
        except KeyError:
            raise KeyError(
                f"Group '{group}' not found in any data handler. Available groups: {list(self.data_handlers.keys())}"
            )

    def static(self, *requests):
        return [self.dispatch(group).static((group, variables)) for group, variables in requests]

    def dynamic(self, *requests):
        # potential optimisation: group by data_handler to reduce number of reads
        return [self.dispatch(group).dynamic((i, group, variables)) for i, group, variables in requests]

    def dates_block(self, group):
        return self.dispatch(group).dates_block(group)

    def _tree(self, prefix=""):
        # for pretty printing of dicts
        name = prefix if prefix else ""
        tree = _RichTree(name)
        for group, dh in self.data_handlers.items():
            tree.add(dh._tree(prefix=f"{group}"))
        return tree

    def __repr__(self):
        console = Console(record=True)
        tree = self._tree()
        with console.capture() as capture:
            console.print(tree, overflow="ellipsis")
        return capture.get()


def build_data_handler(config: dict, /, kind: str) -> DataHandler:
    initial_config = config.copy()
    config.pop("aliases", None)  # remove aliases if present

    if kind not in config:
        raise ValueError(f"build_data_handler: kind={kind} must be one of ['training', 'validation', 'test']")
    if "sources" not in config:
        raise ValueError("build_data_handler: config must contain 'sources' key")

    if isinstance(config, DictConfig):
        # found an omegaconf DictConfig, resolve it first
        config = _resolve_omega_conf_reference(config)
    config = config.copy()

    sources = config["sources"]
    for cfg in sources:
        if "search_path" not in cfg:
            # if not defined in cfg, use the provided global search_path
            cfg["search_path"] = config[kind].get("search_path", config.get("search_path"))

        # start and end always come from the kind-specific config
        if "start" in config[kind]:
            cfg["start"] = config[kind]["start"]
        if "end" in config[kind]:
            cfg["end"] = config[kind]["end"]

        if "dataset" not in cfg:
            raise ValueError("Each source in registry config must contain a 'dataset' key")

    dh = DataHandler(*sources)
    dh._initial_config = initial_config
    return dh
