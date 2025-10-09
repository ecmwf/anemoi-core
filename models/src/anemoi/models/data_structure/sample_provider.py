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

import einops
import numpy as np
from omegaconf import DictConfig
from rich.console import Console
from rich.tree import Tree as _RichTree

from anemoi.models.data_structure.data_handler import DynamicDataDict
from anemoi.models.data_structure.data_handler import StaticDataDict
from anemoi.models.data_structure.data_handler import _resolve_omega_conf_reference
from anemoi.models.data_structure.offsets import OffsetManagerVisitor
from anemoi.models.data_structure.offsets import find_required_steps_for_rollout
from anemoi.models.data_structure.offsets import sum_offsets

LOGGER = logging.getLogger(__name__)


def _merge_sublists(d):
    # merge a dict of lists into a single list
    res = []
    for v in d.values():
        if not isinstance(v, list):
            raise ValueError(f"Expected list for offsets, got {type(v)}: {v}")
        res += v
    return res


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
    def __getitem__(self, i):
        pass

    def __len__(self):
        if self._dates_block is None:
            warnings.warn("Length requested before dates_block is set")
            return None
        return len(self._dates_block)

    def visit(self, visitor):
        # default implementation is to just call the visitor on the visited object
        visitor(self)

    def _dates_block_in_dataset(self):
        return None

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


class SampleProviderDictionary(SampleProvider):
    # A dictionary of sample providers
    # forwards everything to the sub-sample providers and aggregate the results as dictionaries

    def __init__(self, _context: Context, providers: dict):
        self.context = _context
        self._providers = {k: _sample_provider_factory(_context, cfg) for k, cfg in providers.items()}

    def visit(self, visitor):
        visitor(self)
        for p in self._providers.values():
            p.visit(visitor)

    @property
    def static(self):
        return {k: v.static for k, v in self._providers.items()}

    def __getitem__(self, i):
        return {k: v[i] for k, v in self._providers.items()}

    def _tree(self, prefix=None):
        name = ""  # self.__class__.__name__
        if prefix:
            name = f"{prefix}: {name}"
        tree = _RichTree(name)
        for k, v in self._providers.items():
            tree.add(v._tree(prefix=k))
        return tree


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

    def __getitem__(self, i):
        return self._forward[i]

    def __len__(self):
        return len(self._forward)

    def _tree(self, prefix=None):
        return self._forward._tree(prefix=prefix)


class Rearrange(Forward):

    def __init__(self, _context: Context, forward, dimensions):
        super().__init__(_context, forward)

        if not isinstance(dimensions, (list, tuple)):
            raise ValueError(f"Expected list/tuple for dimensions, got {type(dimensions)}: {dimensions}")
        if not all(isinstance(d, str) for d in dimensions):
            raise ValueError(f"Expected list/tuple of strings for dimensions, got {dimensions}")

        if "batch" in dimensions:
            raise ValueError("Cannot reshape on 'batch' dimension, it is implicit and always first in the tensors.")

        self.dimensions = dimensions
        self._previous_dimensions = self._forward.static.dimensions

        if not set(self.dimensions).issubset(set(self._previous_dimensions)):
            raise ValueError(
                f"Dimensions mismatch, previous: {self._previous_dimensions}, new: {self.dimensions}, "
                f"must be a subset of previous"
            )

        previous = [d if d in self.dimensions else "1" for d in self._previous_dimensions]
        new = [_ for _ in self.dimensions]
        self.einops_rearrange_str = f"{' '.join(previous)} -> {' '.join(new)}"

    @property
    def static(self):
        res = self._forward.static.copy()
        assert isinstance(res, StaticDataDict)
        res["dimensions"] = self.dimensions
        # add here something about the new dimensions shape
        return res

    def __getitem__(self, i):
        res = DynamicDataDict()
        for k, v in self._forward[i].items():
            match k:
                case "latitudes" | "longitudes" | "timedeltas" | "date_str":
                    res[k] = v
                case "data":
                    try:
                        res["data"] = einops.rearrange(v, self.einops_rearrange_str)
                    except Exception as e:
                        LOGGER.error(f"{e} while rearranging {(v.shape)} with '{self.einops_rearrange_str}'")
                        LOGGER.error(f"{self}")
                        raise e
                case _:
                    raise ValueError(f"Unexpected key '{k}' in sample provider")
        return res

    def _tree(self, prefix=None):
        name = self.__class__.__name__
        if prefix:
            name = f"{prefix}: {name}"
        tree = self._forward._tree(prefix=prefix)
        tree.add(f"previous dimensions: {self._previous_dimensions}")
        tree.add(f"dimensions: {self.dimensions}")
        tree.add(f"einops rearrange: {self.einops_rearrange_str}")
        return tree


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
        return res

    def __getitem__(self, i):
        multi = {k: v[i] for k, v in self._providers.items()}
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
        return res

    def __getitem__(self, i):
        multi = {k: v[i] for k, v in self._providers.items()}
        res = DynamicDataDict()
        first = next(iter(multi.values()))
        for k, v in first.items():
            assert k in ["data", "latitudes", "longitudes", "timedeltas", "date_str"], k
            res[k] = [v[k] for v in multi.values()]
        return res


class _InsertInside(Forward):
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


class Container(SampleProvider):
    _promise = None

    def __init__(self, _context: Context, container: dict):
        self._context = _context
        self.container = container
        self.variables = container["variables"]
        self.data_group = container["data_group"]
        self.extra = container.get("extra", {})
        self.dh = _context["data_handler"]
        self._offset = _context.get("offset", "0h")
        self._static_request = dict(data_group=self.data_group, variables=self.variables)

    def _dates_block_in_dataset(self):
        return self.dh.dates_block(self.data_group)

    def finalise(self, add_to_i, multiply_i):
        self._dynamic_request = dict(**self._static_request, add_to_i=add_to_i, multiply_i=multiply_i)

    def register_request(self):
        self._promise = self.dh.register_request(**self._dynamic_request)

    def visit(self, visitor):
        visitor(self)

    @classmethod
    def new(cls, _context: Context, container: dict):
        container = container.copy()
        assert "variables" in container, f"Must specify variables, got {container}"

        match container:
            # order matters here

            case {"variables": dict()}:
                # if variables categories, pop them and insert them in static, no further processing
                categories = container.pop("variables")
                container["variables"] = _merge_sublists(categories)
                ALLOWED = ["forcings", "prognostics", "diagnostics"]
                if not all(k in ALLOWED for k in categories.keys()):
                    raise ValueError(f"Expected keys in {ALLOWED} for variables, got {list(categories.keys())}")
                forward = _sample_provider_factory(_context, container)
                return _InsertInside.new(_context, forward=forward, static=dict(variables_categories=categories))

            case {"offsets": dict()}:
                # if key "offsets " is present and with a special format (categories of rollout)
                # handle it here, expanding to a list of offsets and inserting the original dict in static
                # no further processing
                # note: offset as a list is handled below

                categories = container.pop("offsets")
                if len(categories) == 1 and "rollout" in categories:
                    container["offsets"] = find_required_steps_for_rollout(**categories["rollout"])
                else:
                    container["offsets"] = _merge_sublists(categories)
                forward = _sample_provider_factory(_context, container)
                return _InsertInside.new(_context, forward=forward, static=dict(offsets_categories=categories))

            case {"dimensions": list() as dims} if all(isinstance(d, str) for d in dims):
                # found "dimensions" key with simple format (a list of string)
                # rearrange the tensors accordingly to match these dimensions
                dimensions = container.pop("dimensions")
                forward = _sample_provider_factory(_context, container)
                return Rearrange.new(_context, forward=forward, dimensions=dimensions)

            case {"dimensions": [["offsets"], *dimensions]}:
                # Found "dimensions" key with complex format: [["offsets"], str, str, ...]
                assert container["dimensions"][0] == ["offsets"]
                assert "offsets" in container, f"Expected 'offsets' in container when using dimensions {container}"
                _, *dimensions = container.pop("dimensions")
                offsets = container.pop("offsets")
                multi_offset = {}
                for offset in offsets:
                    cfg = container.copy()
                    cfg["offset"] = offset
                    cfg["dimensions"] = dimensions
                    multi_offset[offset] = _sample_provider_factory(_context, cfg)
                return StackAsLists.new(_context, "offsets", multi_offset)

            case {"dimensions": list()}:
                # Found "dimensions" key with unknown format: [["offsets"], str, str, ...]
                raise ValueError(f"Unsupported 'dimensions' value in {container}")

            case {"offsets": list()}:
                # Found "offsets" key with simple format: [str, str, ...]
                # create a Stack along "offsets" dimension
                # note : offset as a dict is handled above
                offsets = container.pop("offsets")
                multi_offset = {}
                for offset in offsets:
                    cfg = container.copy()
                    cfg["offset"] = offset
                    multi_offset[offset] = _sample_provider_factory(_context, cfg)
                return Stack.new(_context, "offsets", multi_offset)

        return cls(_context, container)

    @property
    def static(self):
        self._static = self.dh.get_static(**self._static_request).copy()
        extra = {**self._static.pop("extra", {}), **self.extra}
        if extra:
            self._static["extra"] = extra
        return self._static

    def __getitem__(self, i):
        return self._promise[i]

    def _tree(self, prefix=None):
        name = ""  # self.__class__.__name__
        if prefix:
            name = f"{prefix}: {name}"
        tree = _RichTree(name)
        tree.add(f"container: {self.container}")
        return tree


def _sample_provider_factory(_context: Context, cfg: Any) -> SampleProvider:
    LOGGER.debug(f"Building sample provider from config: {cfg}")
    cfg = cfg.copy()
    match cfg:
        case DictConfig():
            # found an omegaconf DictConfig, resolve it first
            cfg = _resolve_omega_conf_reference(cfg)
            return _sample_provider_factory(_context, cfg)

        case {"dictionary": dict() as dictionary} if len(cfg) == 1:
            # create a dictionary of sample providers
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
            return Container.new(_context, cfg)


def build_sample_provider(cfg: Any, data_handler) -> SampleProvider:
    # the context will be available to all sample providers downstream
    # and used to pass information such as data_handler, offset, etc.
    context = Context(data_handler=data_handler)

    sp = _sample_provider_factory(context, dict(dictionary=cfg))
    # at this point the sample provider is not finished yet,
    # in particular, the dates are not computed yet

    # we use the visitor pattern to traverse the sample provider tree
    # and allow each sample provider to communicate
    visitor = OffsetManagerVisitor()
    # first, the visitor will read all the offsets and compute the overall date range
    sp.visit(visitor.read_date_offsets)
    LOGGER.debug(f"Computed date range: {visitor.dates_block}")
    # then, the visitor will write back the indices to each container
    sp.visit(visitor.write_index_offsets)
    sp.missing = visitor.dates_block.missing_indices()

    # finally, we can register each container's request to the data handler
    def register_requests(node):
        if isinstance(node, Container):
            node.register_request()
            LOGGER.debug(f"Registered request for container {node.container}")
            LOGGER.debug(f"  request: {node._dynamic_request}")
        return True

    sp.visit(register_requests)
    sp.visit(visitor.write_dates_block)

    return sp
