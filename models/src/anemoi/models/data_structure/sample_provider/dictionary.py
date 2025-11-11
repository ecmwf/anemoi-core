# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings

from rich.tree import Tree as _RichTree

from anemoi.models.data_structure.sample_provider.base import Context
from anemoi.models.data_structure.sample_provider.base import SampleProvider
from anemoi.models.data_structure.sample_provider.base import _sample_provider_factory

LOGGER = logging.getLogger(__name__)


class SampleProviderDictionary(SampleProvider):
    # A dictionary of sample providers
    # forwards everything to the sub-sample providers and aggregate the results as dictionaries

    def __init__(self, _context: Context, providers: dict):
        self.context = _context
        self._providers = {}
        for k, cfg in providers.items():
            cfg = cfg.copy()
            if "data_group" in cfg:
                if cfg["data_group"] != k:
                    raise ValueError(
                        f"Group key in the dictionary '{k}' does not match the data_group '{cfg['data_group']}'"
                    )
                warnings.warn(
                    f"data_group '{cfg['data_group']}' in the dictionary entry is redundant with the key '{k}'",
                    UserWarning,
                )
            if "data_group" not in cfg:
                cfg["data_group"] = k
            self._providers[k] = _sample_provider_factory(_context, cfg)

    def visit(self, visitor):
        visitor(self)
        for p in self._providers.values():
            p.visit(visitor)

    @property
    def static(self):
        return {k: v.static for k, v in self._providers.items()}

    def _getitem(self, data):
        return {k: v._getitem(data) for k, v in self._providers.items()}

    def _tree(self, prefix=None):
        name = ""  # self.__class__.__name__
        if prefix:
            name = f"{prefix}: {name}"
        tree = _RichTree(name)
        for k, v in self._providers.items():
            tree.add(v._tree(prefix=k))
        return tree
