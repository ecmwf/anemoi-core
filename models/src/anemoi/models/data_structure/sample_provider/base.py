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
from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np
from omegaconf import DictConfig
from rich.console import Console

from anemoi.models.data_structure.data_handler import build_data_handler
from anemoi.models.data_structure.dicts import StaticDataDict
from anemoi.models.data_structure.dicts import _resolve_omega_conf_reference
from anemoi.models.data_structure.offsets import DatesGathererVisitor
from anemoi.models.data_structure.offsets import sum_offsets

LOGGER = logging.getLogger(__name__)


class Context(dict):
    # This internal class is used to manage context variables.
    # Avoiding passing many parameters in the kwargs

    def __init__(self, parent_context=None, **kwargs):
        super().__init__()
        if parent_context is not None:
            self.update(parent_context)

        # handle offset specially, to sum them if both parent and current context have it
        # has no effect if offsets are not chained
        if "offset" in kwargs:
            offset_1 = self.get("offset", "0h")
            offset_2 = kwargs.pop("offset", "0h")
            self.offset = sum_offsets(offset_1, offset_2)

        if "frequency" in kwargs:
            if "frequency" in self:
                # because is is untested
                raise ValueError(f"Cannot override frequency in context, already set to {self['frequency']}")

        self.update(kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __repr__(self):
        return f"Context({super().__repr__()})"


class SampleProvider(ABC):
    # Base class for sample providers
    # it defines the interface and some common methods
    # sample_providers should not be instanciated directly, but through the build function

    _dates_block = None
    _offset = None
    missing = None

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def new(cls, _context: Context, *args, **kwargs):
        # the default is to call the constructor
        # but subclasses can override this to implement custom logic
        # this allow a class to return a different class
        return cls(_context, *args, **kwargs)

    @property
    @abstractmethod
    def static(self):
        raise NotImplementedError(f"{self.__class__.__name__}.static is not implemented")

    @abstractmethod
    def __getitem__(self, *args):
        pass

    def __len__(self):
        if self._dates_block is None:
            warnings.warn("Length requested before dates_block is set")
            return None
        return len(self._dates_block)

    def visit(self, visitor):
        # default implementation is to just call the visitor on the visited object
        visitor(self)

    @property
    def _dates_block_in_dataset(self):
        return None

    def filter_available_dates(self, dates_block):
        return dates_block

    def update_index_offsets(self: "SampleProvider", dates_block):
        pass

    def set_dates_block(self, date_block):
        self._dates_block = date_block

    @abstractmethod
    def _tree(self, prefix=None):
        # for display purposes
        pass

    def __repr__(self):
        console = Console(record=True)
        tree = self._tree()
        with console.capture() as capture:
            console.print(tree, overflow="ellipsis")
        return capture.get()


class Forward(SampleProvider):
    def __init__(self, _context: Context, forward: SampleProvider):
        self.context = _context
        self._forward = forward

    def visit(self, visitor):
        visitor(self)
        self._forward.visit(visitor)

    @property
    def static(self):
        return self._forward.static

    def __getitem__(self, data):
        return self._forward.__getitem__(data)

    def _tree(self, prefix=None):
        return self._forward._tree(prefix=prefix)


class Indices:
    def __init__(self):
        self._names_to_index = {}
        self._categories = {}
        self._masks = {}

    def add_categories(self, name, name_to_index, categories):
        self._names_to_index[name] = name_to_index
        self._categories[name] = categories
        self._masks = {}
        print("✅", categories)
        print("❌", name_to_index)
        for cat_name, vars in categories.items():
            self._masks[cat_name] = np.array([name_to_index[v] for v in vars])

    def __call__(self, **kwargs):
        if len(kwargs) == 0:
            raise ValueError("At least one variable category must be specified")
        if len(kwargs) > 1:
            raise NotImplementedError("Only one variable category can be requested at a time")

        name = list(kwargs.keys())[0]
        category = kwargs[name]

        if name not in self._names_to_index:
            raise ValueError(f"Unknown name '{name}', available: {list(self._names_to_index.keys())}")
        if category not in self._variables_categories[name]:
            raise ValueError(
                f"Unknown variable category '{category}' for name '{name}', available: {list(self._variables_categories[name].keys())}"
            )
        return self._variables_masks[name][category]

    def __repr__(self):
        lst = [""]
        for name, categories in self._categories.items():
            lst.append(f"  {name}: {list(categories.keys())}")
        return "\n".join(lst)


class AddCategories(Forward):
    def __init__(self, _context: Context, forward, variables: dict = None, offsets: dict = None):
        super().__init__(_context, forward)
        self.variables = variables
        self.offsets = offsets

    @property
    def static(self):
        res = self._forward.static.copy()
        assert isinstance(res, StaticDataDict)

        if "indices" not in res:
            res.indices = Indices()

        if self.variables is not None:
            res.indices.add_categories("variables", name_to_index=res["name_to_index"], categories=self.variables)

        if self.offsets is not None:
            name_to_index = {name: i for i, name in enumerate(res["offsets"])}
            res.indices.add_categories("offsets", name_to_index=name_to_index, categories=self.offsets)

        return res

    def _tree(self, prefix=None):
        tree = self._forward._tree(prefix=prefix)
        if self.variables is not None:
            tree.add(f"[bold magenta]Variables categories:[/bold magenta] {self.variables}")
        if self.offsets is not None:
            tree.add(f"[bold magenta]Offsets categories:[/bold magenta] {self.offsets}")
        return tree


def _sample_provider_factory(_context: Context, cfg: Any) -> SampleProvider:
    LOGGER.debug(f"Building sample provider from config: {cfg}")
    cfg = cfg.copy()
    if "frequency" in cfg:
        frequency = cfg.pop("frequency")
        _context = Context(_context, frequency=frequency)

    match cfg:
        case DictConfig():
            # found an omegaconf DictConfig, resolve it first
            cfg = _resolve_omega_conf_reference(cfg)
            return _sample_provider_factory(_context, cfg)

        case {"sample": dict() as sample} if len(cfg) == 1:
            return _sample_provider_factory(_context, sample)

        case {"dimensions": dimensions, **config}:
            _context = Context(_context, dimensions=dimensions)
            return _sample_provider_factory(_context, config)

        case {"offset": offset, **config}:
            # should not be used directly by user.
            _context = Context(_context, offset=offset)
            return _sample_provider_factory(_context, config)

        case {"offsets": offsets, **config}:
            _context = Context(_context, offsets=offsets)
            return _sample_provider_factory(_context, config)

        case {"groups": dict() as dictionary} if len(cfg) == 1:
            # create a dictionary of sample providers
            from anemoi.models.data_structure.sample_provider.dictionary import SampleProviderDictionary

            return SampleProviderDictionary.new(_context, dictionary)

        case _:
            # finally, create a container
            from anemoi.models.data_structure.sample_provider.container import Container

            return Container.new(_context, cfg)


def build_sample_provider(cfg: Any, /, kind=None, data_handler=None) -> SampleProvider:
    cfg = cfg.copy()
    cfg.pop("aliases", None)  # remove aliases if present

    initial_config = cfg.copy()

    if kind is None:
        kind = cfg.pop("kind", None)
    if data_handler is None:
        data_handler_config = cfg.pop("data_handler")
        data_handler = build_data_handler(data_handler_config, kind=kind)

    # the context will be available to all sample providers downstream
    # and used to pass information such as data_handler, offset, etc.
    context = Context(data_handler=data_handler)

    sp = _sample_provider_factory(context, cfg)
    # at this point the sample provider is not finished yet,
    # in particular, the dates are not computed yet

    # we use the visitor pattern to traverse the sample provider tree
    # and allow each sample provider to communicate
    # the visitor reads all the offsets and compute the overall date range
    visitor = DatesGathererVisitor()
    sp.visit(visitor.read_date_offsets)
    dates_block = visitor.dates_block
    LOGGER.debug(f"Computed available dates: {dates_block}")

    # then, update the indices to each container
    sp.visit(lambda obj: obj.update_index_offsets(dates_block))
    sp.visit(lambda obj: obj.set_dates_block(dates_block))

    sp.missing = dates_block.missing_indices
    sp._initial_config = initial_config
    return sp
