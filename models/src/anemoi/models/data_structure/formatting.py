# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import datetime
import os

import numpy as np
import torch
from rich.console import Console
from rich.tree import Tree as _RichTree

from anemoi.models.data_structure.offsets import offset_to_string

# this file is quite long and has a lot of knowledge about the other types
# it is not too bad because everything related to display is here

ICON_BOX = "📦"
ICON_LEAF = " "
ICON_LEAF_BOX_NOT_FOUND = "🍀"

VERBOSITY = int(os.environ.get("ANEMOI_CONFIG_VERBOSE_STRUCTURE", "0"))
DEBUG = VERBOSITY == 2


KNOWN_EXTRA_KEYS = [
    "statistics",
    "name_to_index",
    "normaliser",
    "inputer",
    "extra",
    "metadata",
    "number_of_features",
    "dimensions_order",
    "rollout_usage",
]


def choose_icon(k, v):
    return dict(
        latitudes="🌍",
        longitudes="🌍",
        timedeltas="🕐",
        data="🔢",
        rollout="🤖♻️ ",
        rollout_usage="♻️ ",
    ).get(k, None)


def format_shorten(k, v):
    if len(str(v)) < 50:
        return f"{v}"
    return f"{str(v)[:50]}..."


def format_timedeltas(k, v):
    def _to_str(x):
        x = int(x)
        x = datetime.timedelta(seconds=x)
        return offset_to_string(x)

    try:
        if isinstance(v, np.ndarray):
            minimum = _to_str(np.min(v))
            maximum = _to_str(np.max(v))
            return f"np.array{v.shape} [{minimum},{maximum}]"

        if isinstance(v, torch.Tensor):
            minimum = _to_str(torch.min(v).item())
            maximum = _to_str(torch.max(v).item())
            return f"tensor({v.shape}) [{minimum},{maximum}]"

        return "no-min, no-max"

    except (ValueError, ImportError):
        return "[no-min, no-max]"


def format_array(k, v):
    try:
        if isinstance(v, np.ndarray):
            minimum = np.min(v)
            maximum = np.max(v)
            mean = np.nanmean(v)
            stdev = np.nanstd(v)
            return f"np.array{v.shape} {mean:.5f}±{stdev:.1f}"  # [{minimum:.1f},{maximum:.1f}]"

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


def format_key_value(k, v):
    txt = f"{v.__class__.__name__}"
    assert k != "Value"
    assert "Value" not in str(k)
    assert "Value" not in str(v)

    if k == "timedeltas":
        txt = format_timedeltas(k, v)
    elif k == "rollout_usage":
        txt = ""
        for step, lst in v.items():
            usages = ",".join(f"{kind}({when})" for kind, when in lst)
            txt += f"step{step}:{usages} "

    elif isinstance(v, (int, float, bool)):
        txt = f"{v}"
    elif isinstance(v, str):
        txt = f"'{v}'"
    elif isinstance(v, (list, tuple)):
        txt = format_shorten(k, str(v))
    elif isinstance(v, dict):
        txt = format_shorten(k, f"dict with {len(v)} keys " + ",".join(f"{k_}" for k_ in v.keys()))
    elif isinstance(v, np.ndarray):
        txt = format_array(k, v)
    try:
        import torch

        if isinstance(v, torch.Tensor):
            txt = format_array(k, v)
    except ImportError:
        pass
    if isinstance(v, np.datetime64):
        txt = f'np.datetime64("{v!s}")'
    if isinstance(v, datetime.datetime):
        txt = f"datetime({v})"
    from anemoi.models.data_structure.sample_provider import Rollout

    if isinstance(v, Rollout):
        txt = str(v)
    return txt


class AnemoiTree:
    def __init__(self, obj):
        assert isinstance(obj, dict), type(obj)

        from anemoi.models.data_structure.structure import TreeDict

        assert isinstance(obj, TreeDict), type(obj)

        self.obj = obj
        self.name = obj.__class__.__name__

    def build_tree(self):
        from anemoi.models.data_structure.path_keys import decode_path_if_needed

        if not self.obj:
            return f"{self.obj.__class__.__name__} (empty)"

        tree = _RichTree(self.name)

        max_length = 0
        for k, v in self.obj.items():
            k = decode_path_if_needed(k)
            max_length = max(max_length, len(str(k)))

        for k, v in self.obj.items():
            k = decode_path_if_needed(k)
            k = str(k).ljust(max_length)
            if hasattr(v, "tree") and callable(v.tree):  # when it is a Rollout
                tree.add(v.tree(prefix=k))
                continue

            if isinstance(v, dict):
                if VERBOSITY == 0:
                    box = OneLineBox(k, v)
                else:
                    box = VerboseBox(k, v)
                box.add_to_tree(tree)
                continue

            value = Value("root-path", k, v)
            value.add_to_tree(tree)

        return tree


class Box:
    def __init__(self, path, box):
        assert isinstance(box, dict), type(box)
        self.path = path

        def priority(k):
            if str(k).startswith("_"):
                return 10
            return dict(data=1, latitudes=2, longitudes=3, timedeltas=4).get(k, 9)

        order = sorted(box.keys(), key=priority)
        box = {k: box[k] for k in order}

        self.values = {k: Value(path, k, v) for k, v in box.items()}


class VerboseBox(Box):

    def add_to_tree(self, tree):
        name = self.path
        if "_reference_date" in self.values:
            name += f" (Reference {self['_reference_date']})"
        t = _RichTree(name)
        for v in self.values.values():
            v.add_to_tree(t)
        tree.add(t)


class OneLineBox(Box):

    def add_to_tree(self, tree):

        has_known_extra = False
        has_underscores = False

        if len(self.values) == 1:
            self.values[list(self.values.keys())[0]].add_to_tree(tree)
            return

        txt = ""
        for k, v in self.values.items():
            if k == "data":
                x = v.icon + " "
                x += format_key_value("data", v.value)
                x = x.replace("data : ", "")
                x = x[:30] + ("…" if len(x) > 30 else "")
                txt += x
                continue

            if v.icon:
                txt += v.icon
                continue

            if v._is_known_extra:
                has_known_extra = True
                continue

            if str(k).startswith("_"):
                has_underscores = True
                continue

            txt += "+" + k

        if has_underscores:
            txt = "_ " + txt

        if has_known_extra:
            txt = "* " + txt

        tree.add(f"{self.path}: {txt}")


class Value:
    def __init__(self, path, key, value):
        if ".prognostics." in key:
            key = "🔄  " + key
        if ".diagnostics." in key:
            key = "out " + key
        if ".forcings." in key:
            key = "in  " + key
        self.path = path
        self.key = key
        self.value = value
        self.icon = choose_icon(key, value)
        self._is_known_extra = key in KNOWN_EXTRA_KEYS

    def add_to_tree(self, tree):
        if not DEBUG and str(self.key).startswith("_"):
            return

        if self.key == "rollout_usage":
            subtree = _RichTree(f"{self.icon} {self.key} : {self.value}")
            tree.add(subtree)
            return

        if self.key == "rollout":
            if DEBUG:
                tree.add(self.value.tree(prefix=self.key))
            else:
                tree.add(_RichTree(f"{self.icon} {self.key} : {self.value}"))
            return

        if isinstance(self.value, dict):
            if not self.value:
                tree.add(f"{self.icon or ICON_LEAF} {self.key} : ❌ <empty-dict> {self.value}")
                return

        tree.add(f"{self.icon or ICON_LEAF} {self.key} : {format_key_value(self.key, self.value)}")


def anemoi_dict_to_str(obj):
    x = AnemoiTree(obj)
    tree = x.build_tree()

    console = Console(record=True)
    with console.capture() as capture:
        console.print(tree, overflow="ellipsis")
    return capture.get()
