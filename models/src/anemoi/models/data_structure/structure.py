# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import warnings
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Union

import torch
from rich import print
from torch.utils.checkpoint import checkpoint

from anemoi.models.data_structure.formatting import anemoi_dict_to_str
from anemoi.models.data_structure.path_keys import SEPARATOR
from anemoi.models.data_structure.path_keys import encode_path_if_needed
from anemoi.models.data_structure.path_keys import join_paths

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
NestedTensor = Union[
    torch.Tensor,
    Sequence["NestedTensor"],  # list/tuple of NestedTensor
    Mapping[str | int, "NestedTensor"],  # dict with str/int keys
]


def deep_freeze_dict(d):
    # MappingProxyType is not used because it is not of type dict and this is confusing
    if not isinstance(d, dict):
        return d
    return FrozenDict({k: deep_freeze_dict(v) for k, v in d.items()})


class FrozenDict(dict):
    def __setitem__(self, key, value):
        raise TypeError("FrozenDict is immutable, cannot set item '{key}'")

    def __delitem__(self, key):
        raise TypeError("FrozenDict is immutable, cannot delete item '{key}'")

    def clear(self):
        raise TypeError("FrozenDict is immutable")

    def pop(self, key, default=None):
        raise TypeError("FrozenDict is immutable, cannot pop item '{key}'")

    def popitem(self):
        raise TypeError("FrozenDict is immutable")

    def setdefault(self, key, default=None):
        raise TypeError("FrozenDict is immutable, cannot set default for item '{key}'")

    def update(self, *args, **kwargs):
        raise TypeError("FrozenDict is immutable, cannot update")

    def __hash__(self):
        return hash(frozenset(self.items()))


class TreeDict(dict):

    _is_frozen = False

    def __init__(self, *args, **kwargs):
        # Nothing special in the __init__, this is an actual python dict

        # we are only converting the keys to strings usable as Module names for pytorch ModuleDict
        # i.e. convert funny characters (including '.') in a reversible way.
        dic = dict(*args, **kwargs)
        for k, v in dic.items():
            self[k] = v

    def copy(self):
        return self.__class__(self)
        # not implemented on purpose to avoid confusion
        raise NotImplementedError("use copy.copy or copy.deepcopy instead")

    def freeze(self):
        for k, v in self.items():
            if v is None:
                continue
            # if hasattr(v, "freeze"):
            #    v = v.freeze()
            if not isinstance(v, dict):
                raise ValueError(f"Unexpected leaf for key '{k}', got {type(v)} instead of dict")
            v = deep_freeze_dict(v)
            self[k] = v
        self._is_frozen = True
        return self

    def unfreeze(self):
        for k, v in self.items():
            if v is None:
                continue
            if not isinstance(v, dict):
                raise ValueError(f"Unexpected leaf for key '{k}', got {type(v)} instead of dict")
            if isinstance(v, FrozenDict):
                v = dict(v)
                self[k] = v
        self._is_frozen = False
        return self

    def __copy__(self):
        return self.__class__(self)

    def __deepcopy__(self, memo):
        import copy

        return self.__class__(copy.deepcopy(dict(self), memo))

    @classmethod
    def new_empty(cls):
        return cls()

    def __repr__(self):
        return anemoi_dict_to_str(self)

    def to_str(self, name):
        return name + " " + self.__repr__()

    def __setitem__(self, path, value):
        if self._is_frozen:
            raise TypeError("TreeDict is frozen, cannot set item '{path}'")
        path = encode_path_if_needed(path)
        # if not path:
        #    raise KeyError("Empty path is not allowed")
        if isinstance(value, TreeDict):
            for p, v in value.items():
                #                assert p, f"Empty sub-path is not allowed when setting '{path}' to a TreeDict"
                new_path = join_paths(path, p)
                self[new_path] = v
            return
        super().__setitem__(path, value)

    def __getitem__(self, path):
        path = encode_path_if_needed(path)
        if path in self:
            return super().__getitem__(path)
        prefix = path + SEPARATOR
        matching = {p[len(prefix) :]: v for p, v in self.items() if p.startswith(prefix)}
        if not matching:
            raise KeyError(f"Key '{path}' not found in TreeDict with keys: {list(self.keys())}")
        return self.__class__(matching)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    # def __getattr__(self, name):
    #     try:
    #         return self[name]
    #     except KeyError:
    #         raise AttributeError(f"'TreeDict' object has no attribute '{name}'")

    def as_function(self):
        res = Function()
        for path, v in self.items():
            if not callable(v):
                raise ValueError(f"Unexpected leaf for key '{path}', got {type(v)}")
            res[path] = v
        return res

    def as_module_dict(self):
        res = AnemoiModuleDict()
        for path, v in self.items():
            if not isinstance(v, torch.nn.Module):
                raise ValueError(f"Unexpected leaf for key '{path}', got {type(v)}")
            res[path] = v
        return res

    def as_batch(self):
        res = Batch()
        for path, v in self.items():
            res[path] = v
        return res

    def as_dict(self):
        return {k: v.as_dict() if isinstance(v, TreeDict) else v for k, v in self.items()}

    def wrap(self, key):
        return self.map(lambda v: {key: v})

    def unwrap(self, key):
        return self.map(lambda v: v[key])

    def select_content(self, *keys):
        """Usage:
        select_content(["latitudes", "longitudes"]) returns a TreeDict with boxes containing only the selected keys
        select_content("data") returns a TreeDict with the content of the "data" key, no box
        select_content("latitudes", "longitudes") returns a tuple of Dicts, each containing the content of the corresponding key, no boxing
        select_content(["latitudes"], ["longitudes"]) returns a tuple of Dicts, each containing the content of the corresponding key, with boxing
        """
        if len(keys) > 1:
            # return tuple when asked for multiple keys or multiple key lists
            return tuple(self.select_content(k) for k in keys)
        keys = keys[0]

        def select(box):
            if not isinstance(box, dict):
                raise ValueError(f"Expected dict, got {type(box)}, cannot select_content on {type(box)}")
            for k in keys:
                if isinstance(k, str) and k not in box:
                    raise ValueError(f"Key '{k}' not found in dict. Available keys are: {list(box.keys())}")
            res = {}
            for k, v in box.items():
                if isinstance(keys, (tuple, list, set)):
                    if k in keys:
                        res[k] = v
                elif isinstance(keys, str):
                    if k == keys:
                        res[k] = v
                elif callable(keys):
                    if keys(k):
                        res[k] = v
                else:
                    raise ValueError(f"Unexpected type for keys: {type(keys)} '{keys}'")
            return res

        return self.map(select)

    def __add__(self, other):
        if not isinstance(other, TreeDict):
            raise ValueError(f"Cannot add {type(self)} and {type(other)}")
        return self.each + other.each

    def merge_leaves(self, *args, **kwargs):
        assert False, "This method has been renamed use + operator : TreeDict() + TreeDict()"

    def map(self, func, *args, **kwargs):
        res = self.__class__()
        for path, leaf in self.items():
            try:
                args_ = [a[path] if isinstance(a, TreeDict) else a for a in args]
                kwargs_ = {k: v[path] if isinstance(v, TreeDict) else v for k, v in kwargs.items()}
                new = func(leaf, *args_, **kwargs_)
                res[path] = new
            except Exception as e:
                e.add_note(f"When processing key '{path}'")
                raise e
        return res

    def map_expanded(self, func, *args, **kwargs):
        res = self.__class__()
        for path, leaf in self.items():
            try:
                args_ = [a[path] if isinstance(a, TreeDict) else a for a in args]
                kwargs_ = {k: v[path] if isinstance(v, TreeDict) else v for k, v in kwargs.items()}
                new = func(*args_, **leaf, **kwargs_)
                res[path] = new
            except Exception as e:
                e.add_note(f"When processing key '{path}'")
                raise e
        return res

    @property
    def each(self):
        return LeafAccessor(self)

    @property
    def first(self):
        """Return the first value in the TreeDict. Useful for accessing properties common to all boxes."""
        return next(iter(self.values()))

    def matches_keys(self, other):
        return set(self.keys()) == set(other.keys())

    def contains_keys(self, other):
        """Check that the keys of self contain the keys of other.

        Parameters
        ----------
        other : TreeDict
            The other TreeDict to compare with.

        Returns
        -------
        bool
            True if self contains all keys of other, False otherwise.
        """
        keys = set(self.keys())
        for k in other.keys():
            for k_ in keys:
                if k == k_ or k.startswith(k_ + SEPARATOR):
                    break
            else:
                return False
        return True


class BaseAccessor:
    def __init__(self, parent):
        self.parent = parent

    def copy(self):
        return self.parent.__class__(self._apply_dict_method("copy"))

    def get(self, *args, **kwargs):
        return self.map(lambda x: x.get(*args, **kwargs))
        return self.parent.__class__(self._apply_dict_method("get"), *args, **kwargs)

    def pop(self, *args, **kwargs):
        return self.parent.__class__(self._apply_dict_method("pop", *args, **kwargs))

    def popitem(self, *args, **kwargs):
        return self.parent.__class__(self._apply_dict_method("popitem", *args, **kwargs))

    def reversed(self):
        print("untested code")
        return self.parent.__class__(self._apply_dict_method("reversed"))

    def values(self):
        print("untested code")
        return self.parent.__class__(self._apply_dict_method("values"))

    def __or__(self, other):
        print("untested code")
        return self.parent.__class__(self._apply_dict_method("__or__", other))

    def __getitem__(self, key: str):
        return self.parent.__class__(self._apply_dict_method("__getitem__", key))

    def filter(self, *keys_or_lists, remove=False):
        """Usage:
        select_content(["latitudes", "longitudes"]) returns a TreeDict with boxes containing only the selected keys
        select_content("data") returns a TreeDict with the content of the "data" key, no box
        select_content("latitudes", "longitudes") returns a tuple of Dicts, each containing the content of the corresponding key, no boxing
        select_content(["latitudes"], ["longitudes"]) returns a tuple of Dicts, each containing the content of the corresponding key, with boxing
        """
        if len(keys_or_lists) > 1:
            # return tuple when asked for multiple keys or multiple key lists
            return tuple(self.filter(k, remove) for k in keys_or_lists)

    def __setitem__(self, key: str, value):
        self._apply_dict_method("__setitem__", key, value)

    def update(self, *args, **kwargs):
        self._apply_dict_method("update", *args, **kwargs)

    # def __ior__(self, other):
    #     print("untested code")
    #     self._apply_dict_method("__ior__", other)

    def __add__(self, other):
        # this method does not exists for python dict
        # this is to add adding two Dicts to merge their leaves
        #
        # only this is supported:
        # a + b  where a and b are TreeDict , see TreeDict.__add__
        #
        # we may want to support
        # a.each + b where a and b are TreeDict
        # a.each + c where a is TreeDict and b is regular (constant) dict

        if not isinstance(other, BaseAccessor):
            raise ValueError(f"Cannot add {type(self)} and {type(other)}")

        res = self.result_parent_class()
        for path, box in self.parent.items():
            if not isinstance(box, dict):
                raise ValueError(f"Unexpected leaf for key '{path}' for the first operand, got {type(box)}")

            other_box = other.parent[path]

            if not isinstance(other_box, dict):
                raise ValueError(f"Unexpected leaf for key '{path}' for the second operand, got {type(other_box)}")

            res[path] = _merge_dicts_allow_overwrite_none(box, other_box)

        return res

    def _apply_dict_method(self, method_name, *args, **kwargs):
        def function_finder(path, leaf):
            if not isinstance(leaf, dict):
                raise ValueError(f"Expected dict for key '{path}', got {type(leaf)}")
            return getattr(leaf, method_name)

        return self._parallel_apply_on_leaves(function_finder, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        def function_finder(path, leaf):
            if not callable(leaf):
                raise ValueError(f"Expected callable for key '{path}', got {type(leaf)}")
            return leaf

        res = self.result_parent_class()
        for k, v in self._parallel_apply_on_leaves(function_finder, *args, **kwargs):
            res[k] = v
        return res

    def map(self, func, *args, **kwargs):
        warnings.warn("don't use .each.map, use .map directly")
        return self.parent.map(func, *args, **kwargs)

    def _parallel_apply_on_leaves(self, func_finder, *args, **kwargs):
        def resolve(x, path, log):
            if not isinstance(x, TreeDict):
                return x
            try:
                return x[path]
            except KeyError as e:
                e.add_note(f"While processing key '{path}'. Available keys in {log} are: {list(x.keys())}")
                raise e

        output = []
        for path, leaf in self.parent.items():
            # when args or kwargs are TreeDict, extract the corresponding leaf
            args_ = [resolve(a, path, i) for i, a in enumerate(args)]
            kwargs_ = {k: resolve(v, path, k) for k, v in kwargs.items()}
            # find the function/method to apply and apply it
            func = func_finder(path, leaf)
            try:
                res = func(*args_, **kwargs_)
            except Exception as e:
                e.add_note(f"While processing key '{path}', leaf={leaf}, {args_=}, {kwargs_=}")
                raise e
            # stack in a list so that the caller decides how to handle it
            output.append((path, res))
        return output


def _merge_dicts_allow_overwrite_none(d1, d2):
    merged = d1.copy()
    for key, value in d2.items():
        if key not in merged:
            merged[key] = value
            continue
        # found conflicting key
        if merged[key] is not None and value is not None:
            raise ValueError(f"Conflicting key '{key}' found, values: {merged[key]} and {value}")
        if merged[key] is None:
            merged[key] = value
        assert value is None

    return merged


class LeafAccessor(BaseAccessor):
    @property
    def result_parent_class(self):
        return self.parent.__class__


class ModuleDictAccessor(BaseAccessor):
    @property
    def result_parent_class(self):
        return TreeDict


class Batch(TreeDict):
    pass


class AnemoiModuleDict(torch.nn.ModuleDict):
    emoji = "🔥"

    @property
    def each(self):
        return ModuleDictAccessor(self)

    def call_with_checkpointing(self, *args, **kwargs):
        return checkpoint(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        assert False, "not used?, should be deleted?"
        first = None
        if args:
            first = args[0]
            res = first.__class__()
        elif kwargs:
            first = next(iter(kwargs.values()))
            res = first.__class__()
        else:
            res = TreeDict()

        for path, module in self.items():
            print(f"applying module {module} at path {path}")
            args_ = [a[path] if isinstance(a, TreeDict) else a for a in args]
            kwargs_ = {}
            for k, v in kwargs.items():
                if isinstance(v, TreeDict):
                    try:
                        v = v[path]
                    except KeyError as e:
                        e.add_note(f"While processing path kwargs '{k}'. Available paths are: {list(v.keys())}")
                        raise e
                kwargs_[k] = v
            res[path] = module(*args_, **kwargs_)
        return res

    def as_anemoi_dict(self):
        res = TreeDict()
        for path, v in self.items():
            res[path] = v
        return res

    def __repr__(self):
        return self.as_anemoi_dict().__repr__()

    def to_str(self, name=""):
        return self.as_anemoi_dict().to_str(self.emoji + name)


class Function(TreeDict):
    emoji = "()"

    def __call__(self, other, *args, _output_box=True, **kwargs):
        assert False, "dead code?"
