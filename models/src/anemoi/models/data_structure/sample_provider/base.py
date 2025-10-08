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

from omegaconf import DictConfig
from rich.console import Console

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

    def __getitem__(self, *args):
        dh = self.context["data_handler"]
        data = dh.grouped_read(*args)
        return self._getitem(data)

    @abstractmethod
    def _getitem(self, data):
        pass

    def __len__(self):
        if self._dates_block is None:
            warnings.warn("Length requested before dates_block is set")
            return None
        return len(self._dates_block)

    def visit(self, visitor):
        # default implementation is to just call the visitor on the visited object
        visitor(self)

    def update_index_offsets(self: "SampleProvider", dates_block):
        pass

    @property
    def _dates_block_in_dataset(self):
        return None

    def register_read_pattern(self):
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

    def _getitem(self, data):
        return self._forward._getitem(data)

    def _tree(self, prefix=None):
        return self._forward._tree(prefix=prefix)


class InsertStatic(Forward):
    def __init__(self, _context: Context, forward, static: dict):
        super().__init__(_context, forward)
        self.add_to_static = static

    @property
    def static(self):
        res = self._forward.static.copy()
        assert isinstance(res, StaticDataDict)
        for k, v in self.add_to_static.items():
            if k in res:
                raise ValueError(f"Cannot add '{k}' to static, already present in {list(res.keys())}")
            res[k] = v
        return res

    def _tree(self, prefix=None):
        tree = self._forward._tree(prefix=prefix)
        tree.add(f"+{','.join(self.add_to_static.keys())} to static")
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

        case {"dictionary": dict() as dictionary} if len(cfg) == 1:
            # create a dictionary of sample providers
            from anemoi.models.data_structure.sample_provider.dictionary import SampleProviderDictionary

            return SampleProviderDictionary.new(_context, dictionary)

        case {"offset": offset, **config}:
            # create a sample provider with an offset, updating the context
            # note the singular "offset", not "offsets"
            # this is used internally when expanding offsets in a list or dict
            # and is not expected to be used by the user directly
            # (although it would work, and set the offset for all downstream sample providers
            # we may want to support this in the future and extend it to "dimensions" or other keys
            # as well)
            _context = Context(_context, offset=offset)
            return _sample_provider_factory(_context, config)

        case _:
            # finally, create a container
            from anemoi.models.data_structure.sample_provider.container import Container

            return Container.new(_context, cfg)


def build_sample_provider(cfg: Any, data_handler) -> SampleProvider:
    # the context will be available to all sample providers downstream
    # and used to pass information such as data_handler, offset, etc.
    context = Context(data_handler=data_handler)
    initial_config = cfg.copy()

    sp = _sample_provider_factory(context, cfg)
    # at this point the sample provider is not finished yet,
    # in particular, the dates are not computed yet

    # we use the visitor pattern to traverse the sample provider tree
    # and allow each sample provider to communicate
    # the visitor reads all the offsets and compute the overall date range
    visitor = DatesGathererVisitor()
    sp.visit(visitor.read_date_offsets)
    dates_block = visitor.dates_block
    LOGGER.debug(f"Computed date range: {dates_block}")

    # then, update the indices to each container
    sp.visit(lambda obj: obj.update_index_offsets(dates_block))
    sp.visit(lambda obj: obj.set_dates_block(dates_block))

    # finally, we can register each container's request to the data handler
    sp.visit(lambda obj: obj.register_read_pattern())

    sp.missing = dates_block.missing_indices
    sp._initial_config = initial_config
    return sp
