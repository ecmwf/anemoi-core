# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np

from anemoi.models.data_structure.dicts import DynamicDataDict
from anemoi.models.data_structure.sample_provider.base import Context
from anemoi.models.data_structure.sample_provider.base import Forward

LOGGER = logging.getLogger(__name__)


class Stack(Forward):
    """Concatenate multiple sample providers.
    For the data, we stack them along a new dimension.
    For static metadata, and for latitudes, longitudes and timedeltas, we just take the first one
    """

    def __init__(self, _context: Context, name, providers):
        self.name = name
        self._providers = providers
        first = next(iter(providers.values()))
        super().__init__(_context, first)

    def visit(self, visitor):
        visitor(self)
        for p in self._providers.values():
            p.visit(visitor)

    @property
    def static(self):
        res = self._forward.static.copy()
        if self.name in res.dimensions:
            raise ValueError(f"Cannot stack along existing dimension '{self.name}' in {list(res.dimensions)}")
        res["dimensions"] = [self.name] + res.dimensions
        res["offsets"] = [v.static["offset"] for v in self._providers.values()]
        return res

    def _getitem(self, data):
        multi = {k: v._getitem(data) for k, v in self._providers.items()}
        res = DynamicDataDict()
        first = next(iter(multi.values()))
        for k, v in first.items():
            assert k in ["data", "latitudes", "longitudes", "timedeltas", "date_str"], k
            if k == "data":
                res[k] = np.stack([v[k] for v in multi.values()])
                continue
            if k == "date_str":
                res[k] = [v[k] for v in multi.values()]
                continue
            res[k] = v
        return res


class StackAsLists(Forward):
    """Merge multiple sample providers as list
    For the data, we create lists.
    For static metadata, we also take the first one
    """

    def __init__(self, _context: Context, name, providers):
        self.name = name
        self._providers = providers
        first = next(iter(providers.values()))
        super().__init__(_context, first)

    def visit(self, visitor):
        visitor(self)
        for p in self._providers.values():
            p.visit(visitor)

    @property
    def static(self):
        res = self._forward.static.copy()
        if self.name in res.dimensions:
            raise ValueError(f"Cannot stack along existing dimension '{self.name}' in {list(res.dimensions)}")
        res["dimensions"] = [self.name] + res.dimensions
        res["offsets"] = [v.static["offset"] for v in self._providers.values()]
        return res

    def _getitem(self, data):
        multi = {k: v._getitem(data) for k, v in self._providers.items()}
        res = DynamicDataDict()
        first = next(iter(multi.values()))
        for k, v in first.items():
            assert k in ["data", "latitudes", "longitudes", "timedeltas", "date_str"], k
            res[k] = [v[k] for v in multi.values()]
        return res
