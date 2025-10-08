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
from rich.console import Console
from rich.tree import Tree as _RichTree

LOGGER = logging.getLogger(__name__)


def _resolve_omega_conf_reference(config):
    from omegaconf import OmegaConf

    config = OmegaConf.create(config)
    config = OmegaConf.to_container(config, resolve=True)
    return config


def format_value(k, v, level=0) -> str | _RichTree:
    # for pretty printing of dicts
    match v:
        case np.ndarray():
            if v.size == 0:
                return _RichTree(f"{k} : np.array{v.shape} empty")
            if v.ndim > 1:
                minimum = np.min(v, axis=tuple(range(1, v.ndim)))
                maximum = np.max(v, axis=tuple(range(1, v.ndim)))
                mean = np.nanmean(v, axis=tuple(range(1, v.ndim)))
                return _RichTree(f"{k} : np.array{v.shape} {mean} [{minimum},{maximum}]")
            else:
                minimum = np.min(v)
                maximum = np.max(v)
                mean = np.nanmean(v)
                return _RichTree(f"{k} : np.array{v.shape} {mean} [{minimum},{maximum}]")

    import torch

    match v:
        case torch.Tensor():
            shape = ", ".join(str(dim) for dim in v.size())
            v = v[~torch.isnan(v)].flatten()
            if v.numel() == 0:
                minimum = float("nan")
                maximum = float("nan")
                mean = float("nan")
            else:
                minimum = torch.min(v).item()
                maximum = torch.max(v).item()
                mean = torch.mean(v.float()).item()
            return _RichTree(f"{k} : tensor({shape}) {v.device}, {mean} [{minimum},{maximum}]")

    if level >= 1:
        return f"{k}: {str(v)}"

    if len(str(v)) < 50:
        return f"{k}: {str(v)}"

    match v:
        case dict():
            tree = _RichTree(f"{k}: dict({len(v)} keys)")
            if len(v) > 20:
                items = list(v.items())
                tree.add(format_value(items[0][0], items[0][1], level + 1))
                tree.add("...")
                tree.add(format_value(items[-1][0], items[-1][1], level + 1))
                return tree
            for k1, v1 in v.items():
                tree.add(format_value(k1, v1, level + 1))
            return tree
        case list() | tuple() | set():
            tree = _RichTree(f"{k}: list({len(v)} items)")
            if len(v) > 20:
                tree.add(format_value("[0]", v[0], level + 1))
                tree.add("...")
                tree.add(format_value(f"[{len(v)}]", v[-1], level + 1))
                return tree
            for i, item in enumerate(v):
                tree.add(format_value(f"[{i}]", item, level + 1))
            return tree
        case _:
            return f"{k}: {str(v)}"


class BaseDict(dict):
    # allow accessing keys as attributes
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    # allow setting keys as attributes
    def __setattr__(self, name, value):
        self[name] = value

    # add copy here to get a copy of the same class
    def copy(self):
        return self.__class__(self)

    # pretty print
    def __repr__(self):
        console = Console(record=True)
        tree = self._tree()
        with console.capture() as capture:
            console.print(tree, overflow="ellipsis", no_wrap=True)
        return capture.get()

    # pretty print as a tree using rich
    def _tree(self, prefix=None):
        name = prefix if prefix else ""
        tree = _RichTree(name)
        sorted_ = {k: self[k] for k in sorted(self.keys())}
        for k, v in sorted_.items():
            tree.add(format_value(k, v))
        return tree


class DictWithDots(BaseDict):
    # avoid the name DotDict which is used by other packages
    # we may want to use these other DotDict instead of this one
    pass


# Define classes to allow type checking
class StaticDataDict(BaseDict):
    # allow accessing keys as attributes
    def __getattr__(self, name):
        if name in self:
            v = self[name]
            if isinstance(v, dict):
                v = DictWithDots(v)
            return v
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


class DynamicDataDict(BaseDict):
    pass
