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
from functools import cached_property
from typing import List

import numpy as np
import torch
from rich.console import Console
from rich.tree import Tree as _RichTree

from anemoi.datasets import open_dataset
from anemoi.models.data_structure.offsets import DatesBlock

LOGGER = logging.getLogger(__name__)


def format_array(k, v):
    try:
        if isinstance(v, np.ndarray) and v.ndim > 1:
            minimum = np.min(v, axis=tuple(range(1, v.ndim)))
            maximum = np.max(v, axis=tuple(range(1, v.ndim)))
            mean = np.nanmean(v, axis=tuple(range(1, v.ndim)))
            stdev = np.nanstd(v, axis=tuple(range(1, v.ndim)))
            return f"np.array{v.shape} {mean}±{stdev} [{minimum},{maximum}]"

        if isinstance(v, np.ndarray):
            minimum = np.min(v)
            maximum = np.max(v)
            mean = np.nanmean(v)
            stdev = np.nanstd(v)
            return f"np.array{v.shape} {mean}±{stdev} [{minimum},{maximum}]"

        import torch

        if isinstance(v, torch.Tensor):
            shape = ", ".join(str(dim) for dim in v.size())
            v = v[~torch.isnan(v)].flatten()
            if v.numel() == 0:
                minimum = float("nan")
                maximum = float("nan")
                mean = float("nan")
                stdev = float("nan")
            else:
                minimum = torch.min(v).item()
                maximum = torch.max(v).item()
                mean = torch.mean(v.float()).item()
                stdev = torch.std(v.float()).item()
            return f"tensor({shape}) {v.device}, {mean:.5f}±{stdev:.1f}[{minimum:.1f}/{maximum:.1f}]"

        return "no-min, no-max"

    # except (ValueError, ImportError, RuntimeError):
    #    return f"{k}: [no-min, no-max]"
    except Exception as e:
        return f"[error: {e!s}]"


def format_value(k, v):
    if isinstance(v, dict):
        return f"dict({len(v)} items) {'+'.join(v.keys())}"
    if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
        return format_array(k, v)
    return str(v)


class BaseDict(dict):
    def __repr__(self):
        console = Console(record=True)
        tree = self._tree()
        with console.capture() as capture:
            console.print(tree, overflow="ellipsis", no_wrap=True)
        return capture.get()

    def _tree(self, prefix=None):
        name = prefix if prefix else ""
        tree = _RichTree(name)
        sorted_ = {k: self[k] for k in sorted(self.keys())}
        for k, v in sorted_.items():
            tree.add(f"{k}: {format_value(k, v)}")
        return tree

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def copy(self):
        return self.__class__(self)


class StaticDict(BaseDict):
    pass


class DynamicDict(BaseDict):
    pass


class DataHandler:
    """Provides data from multiple datasets"""

    def __init__(self, **kwargs):
        self.requests = []

    def register_request(self, request):
        self.requests.append(request)

    def static(self, data_group):
        raise NotImplementedError(f"{self.__class__.__name__}.static is not implemented")

    def __getitem__(self, i, data_group):
        raise NotImplementedError(f"{self.__class__.__name__}.__getitem__ is not implemented")

    def __len__(self):
        raise NotImplementedError(f"{self.__class__.__name__}.__len__ is not implemented")

    def _tree(self, prefix=None):
        name = prefix if prefix else ""
        tree = _RichTree(name)
        for k in sorted(self.data_groups):
            tree.add(f"{k}: ")
        return tree

    def __repr__(self):
        console = Console(record=True)
        tree = self._tree()
        with console.capture() as capture:
            console.print(tree, overflow="ellipsis")
        return capture.get()


class OneDatasetDataHandler(DataHandler):
    """Provide access to data for one dataset"""

    def __init__(self, data_group: str, dataset: str, start, end, search_path=None, extra=None):
        super().__init__()
        if extra is None:
            extra = {}

        self._dataset = dataset
        self._search_path = search_path
        self.data_groups = [data_group]  # note that self.data_groups is a list
        self.extra = extra
        self.start = start
        self.end = end

        # dimensions should be read from dataset when it is implemented in anemoi-datasets
        self.dimensions = ["variables", "ensembles", "values"]

    def __len__(self):
        return len(self._ds)

    def dates_block(self, data_group) -> DatesBlock:
        assert data_group in self.data_groups, f"Data_group {data_group} not in {self.data_groups}"
        return DatesBlock(self._ds.start_date, self._ds.end_date, self._ds.frequency, self._ds.missing)

    @cached_property
    def _ds(self):
        try:
            # expected use case : dataset name or full path
            return open_dataset(self._dataset)
        except ValueError as e:
            if self._search_path is not None:
                e.add_note(f"Also tried with search_path={self._search_path}")
                try:
                    # also expected use case : a config with dataset name and search_path defined in config
                    return open_dataset(f"{self._search_path}/{self._dataset}.zarr")
                except ValueError:
                    # less expected a config with dataset name with .zarr extension and search_path defined in config
                    return open_dataset(f"{self._search_path}/{self._dataset}")
            raise e

    def __repr__(self):
        if isinstance(self.extra, dict):
            extra = ", ".join(f"{k}={v}" for k, v in self.extra.items())
        else:
            extra = extra.__class__.__name__ + " " + str(self.extra)[:10]
        return f"{super().__repr__()}(dataset={self._dataset}, extra={extra})"


class MultiDatasetDataHandler(DataHandler):
    """Uses several datahandles to provide access to multiple datasets"""

    def __init__(self, *data_handlers):
        super().__init__()

        self.data_groups = []
        for dh in data_handlers:
            self.data_groups.extend(dh.data_groups)

        self._data_handlers = data_handlers

    def static(self, data_group):
        assert data_group in self.data_groups, f"Data_group {data_group} not in {self.data_groups}"
        for dh in self._data_handlers:
            if data_group in dh.data_groups:
                return dh.static(data_group)
        assert False

    def __getitem__(self, i, data_group):
        assert data_group in self.data_groups, f"Data_group {data_group} not in {self.data_groups}"
        for dh in self._data_handlers:
            if data_group in dh.data_groups:
                return dh.__getitem__(i, data_group)
        assert False

    def __len__(self):
        lengths = [len(dh) for dh in self._data_handlers]
        if not lengths:
            return 0
        if not all(l_ == lengths[0] for l_ in lengths):
            raise ValueError(f"All data_handlers must have the same length, got {lengths}")
        return lengths[0]

    def get_static(self, data_group, variables: List[str]) -> StaticDict:
        """Prepare a subselection from a data_group"""
        data_handler = self._find_data_handler(data_group)
        ds = open_dataset(data_handler._ds, select=variables)
        return StaticDict(
            name_to_index=ds.name_to_index,
            statistics=ds.statistics,
            statistics_tendencies=ds.statistics_tendencies,
            metadata=ds.metadata,
            supporting_arrays=ds.supporting_arrays(),
            resolution=ds.resolution,
            variables=ds.variables,
            num_features=len(ds.variables),
            dimensions=data_handler.dimensions,
            extra=data_handler.extra,
        )

    def register_request(self, *, data_group, **kwargs):
        dh = self._find_data_handler(data_group)
        request = DynamicRequest(dh, data_group, **kwargs)
        super().register_request(request)
        return request

    def dates_block(self, data_group):
        dh = self._find_data_handler(data_group)
        return dh.dates_block(data_group)

    def _find_data_handler(self, data_group):
        for dh in self._data_handlers:
            if data_group in dh.data_groups:
                return dh
        raise KeyError(f"Data_group {data_group} not found in any data_handler")


class DynamicRequest:
    def __init__(
        self,
        data_handler: DataHandler,
        data_group: str,
        variables: List[str],
        add_to_i: int,
        multiply_i: int,
    ):
        self._data_handler = data_handler
        self.data_group = data_group
        self.variables = variables
        self._add_to_i = add_to_i
        self._multiply_i = multiply_i
        self._ds = open_dataset(data_handler._ds, select=variables)

    def __getitem__(self, i):
        j = i * self._multiply_i + self._add_to_i
        if j < 0:
            warnings.warn(f"Index {j} is negative, this may lead to unexpected results")

        res = DynamicDict()

        res.data = self._ds[j]

        try:
            res.latitudes = self._ds.get_latitudes(i)
        except Exception:
            res.latitudes = self._ds.latitudes

        try:
            res.longitudes = self._ds.get_longitudes(i)
        except Exception:
            res.longitudes = self._ds.longitudes

        try:
            res.timedeltas = self._ds.get_timedeltas(i)
        except Exception:
            res.timedeltas = None

        res.date_str = self._ds.dates[i].astype(datetime.datetime).isoformat()

        return res


def _data_handler_factory(sources: dict, start, end, search_path) -> DataHandler:
    data_handlers = []
    for cfg in sources:
        cfg["search_path"] = cfg.get("search_path", search_path)
        cfg["start"] = cfg.get("start", start)
        cfg["end"] = cfg.get("end", end)
        # this will need to be changed for observations datasets using another DataHandler class than OneDatasetDataHandler
        data_handlers.append(OneDatasetDataHandler(**cfg))
    return MultiDatasetDataHandler(*data_handlers)


def build_data_handler(config_datagroups: dict, kind: str | None) -> DataHandler:
    if kind not in ["training", "validation", "test", None]:
        raise ValueError(f"build_data_handler: kind={kind} must be one of ['training', 'validation', 'test', None]")

    start = config_datagroups.get(kind, {}).get("start", None)
    end = config_datagroups.get(kind, {}).get("end", None)
    search_path = config_datagroups.get("search_path", None)

    return _data_handler_factory(sources=config_datagroups["sources"], start=start, end=end, search_path=search_path)
