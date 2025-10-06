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
from typing import Any

import einops
import numpy as np
from omegaconf import DictConfig
from rich.console import Console
from rich.tree import Tree as _RichTree

from anemoi.models.data_structure.data_handler import DynamicDict
from anemoi.models.data_structure.data_handler import StaticDict
from anemoi.models.data_structure.offsets import OffsetManagerVisitor
from anemoi.models.data_structure.offsets import sum_offsets

LOGGER = logging.getLogger(__name__)


def _resolve_omega_conf_reference(config):
    from omegaconf import OmegaConf

    config = OmegaConf.create(config)
    config = OmegaConf.to_container(config, resolve=True)
    return config


class Context(dict):
    """Avoid passing many parameters in the kwargs"""

    def __init__(self, parent_context=None, **kwargs):
        super().__init__()
        if parent_context is not None:
            self.update(parent_context)

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


class SampleProvider:
    _dates_block = None
    missing = None

    @classmethod
    def new(cls, _context: Context, *args, **kwargs):
        return cls(_context, *args, **kwargs)

    def __repr__(self):
        console = Console(record=True)
        tree = self._tree()
        with console.capture() as capture:
            console.print(tree, overflow="ellipsis")
        return capture.get()

    @property
    def static(self):
        raise NotImplementedError(f"{self.__class__.__name__}.static is not implemented")

    def __getitem__(self, i):
        raise NotImplementedError(f"{self.__class__.__name__}.__getitem__ is not implemented")

    _len = None

    def __len__(self):
        return self._len

    def visit(self, visitor):
        visitor(self)

    def set_request(self, **kwargs):
        pass


class DictionarySampleProvider(SampleProvider):
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
        assert isinstance(res, StaticDict)
        res["dimensions"] = self.dimensions
        # add here something about the new dimensions shape
        return res

    def __getitem__(self, i):
        res = DynamicDict()
        for k, v in self._forward[i].items():
            assert k in ["data", "latitudes", "longitudes", "timedeltas", "date_str"], k
            if k == "data":
                try:
                    res["data"] = einops.rearrange(v, self.einops_rearrange_str)
                except Exception as e:
                    e.add_note(f"{e} while rearranging {(v.shape)} with '{self.einops_rearrange_str}'")
                    e.add_note(f"{self}")
                    raise e
                continue
            res[k] = v
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
        res = DynamicDict()
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
        res = DynamicDict()
        first = next(iter(multi.values()))
        for k, v in first.items():
            assert k in ["data", "latitudes", "longitudes", "timedeltas", "date_str"], k
            res[k] = [[v[k] for v in multi.values()]]
        return res


class _InsertInside(Forward):
    def __init__(self, _context: Context, forward, static: dict):
        super().__init__(_context, forward)
        self.add_to_static = static

    @property
    def static(self):
        res = self._forward.static.copy()
        assert isinstance(res, StaticDict)
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
    _request = None

    def __init__(self, _context: Context, container: dict):
        self._context = _context
        self.container = container
        self.variables = container["variables"]
        self.data_group = container["data_group"]
        self.extra = container.get("extra", {})
        self.dh = _context["data_handler"]
        self._offset = _context.get("offset", "0h")
        self._dates_block = self.dh.dates_block(self.data_group)

        self.static_request = self.dh.get_static(data_group=self.data_group, variables=self.variables)

    def set_request(self, **kwargs):
        self._request = self.dh.register_request(data_group=self.data_group, variables=self.variables, **kwargs)

    @property
    def dynamic_selection(self):
        if self._request is None:
            warnings.warn("Dynamic selection not set yet")
        return self._request

    def visit(self, visitor):
        visitor(self)

    @classmethod
    def new(cls, _context: Context, container: dict):
        assert "variables" in container, f"Must specify variables when using offsets, got {container}"
        if "variables" in container and isinstance(container["variables"], dict):
            container = container.copy()
            categories = container.pop("variables")
            variables = []
            for k, v in categories.items():
                if not isinstance(v, list):
                    raise ValueError(f"Expected list for variables, got {type(v)}: {v}")
                variables += v
            container["variables"] = variables
            forward = _sample_provider_factory(_context, container)
            return _InsertInside.new(_context, forward=forward, static=dict(variables_categories=categories))

        if "dimensions" in container:
            if all(isinstance(d, str) for d in container["dimensions"]):
                dimensions = container.pop("dimensions")
                forward = _sample_provider_factory(_context, container)
                return Rearrange.new(_context, forward=forward, dimensions=dimensions)
            else:
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

        if "offsets" in container:
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
        res = self.static_request
        extra = res.pop("extra", {})
        extra.update(self.extra)
        if extra:
            res["extra"] = extra
        return res

    def __getitem__(self, i):
        return self.dynamic_selection[i]

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
    if isinstance(cfg, SampleProvider):
        return cfg

    if isinstance(cfg, DictConfig):
        cfg = _resolve_omega_conf_reference(cfg)

    def assert_empty(d):
        if d:
            LOGGER.warning(f"Unused config keys: {d}")
            raise ValueError(f"Expected empty dict, got {d}")

    if "dictionary" in cfg:
        dictionary = cfg.pop("dictionary")
        assert_empty(cfg)
        return DictionarySampleProvider.new(_context, dictionary)

    if "offset" in cfg:
        offset = cfg.pop("offset")
        _context = Context(_context, offset=offset)
        return _sample_provider_factory(_context, cfg)

    return Container.new(_context, cfg)


def build_sample_provider(cfg: Any, data_handler) -> SampleProvider:
    context = Context(data_handler=data_handler)
    sp = _sample_provider_factory(context, dict(dictionary=cfg))
    visitor = OffsetManagerVisitor()
    sp.visit(visitor.read_date_offsets)
    LOGGER.debug(f"SampleProvider date range: {visitor.dates_block}")
    sp.visit(visitor.write_index_offsets)
    sp.missing = visitor.dates_block.missing_indices()
    return sp
