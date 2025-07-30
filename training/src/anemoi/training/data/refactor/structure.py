import datetime

import numpy as np
from rich.console import Console
from rich.tree import Tree

from anemoi.utils.dates import frequency_to_string


class StructureMixin:
    def __repr__(self, **kwargs):
        console = Console(record=True, width=120)
        tree = self.tree(**kwargs)
        with console.capture() as capture:
            console.print(tree)
        return capture.get()


class TupleStructure(StructureMixin, tuple):
    # beware, inheriting from tuple, do not use __init__ method

    def tree(self, prefix="", **kwargs):
        tree = Tree(prefix + "🔗")
        for v in self:
            tree.add(v.tree(**kwargs))
        return tree

    def apply(self, func):
        return TupleStructure([x.apply(func) for x in self])

    def __call__(self, structure, **kwargs):
        assert isinstance(structure, TupleStructure), f"Expected TupleStructure, got {type(structure)}: {structure}"
        return TupleStructure(func(elt, **kwargs) for func, elt in zip(self, structure))

    def __getattr__(self, name):
        return [getattr(x, name) for x in self]

    def _as_native(self):
        return tuple(x._as_native() for x in self)


class DictStructure(StructureMixin, dict):
    def __init__(self, content):
        super().__init__(content)
        for k in self.keys():
            if not hasattr(self, k):
                setattr(self, k, self[k])

    def tree(self, prefix="", **kwargs):
        tree = Tree(prefix + "📖")
        for k, v in self.items():
            tree.add(v.tree(prefix=f"{k} : ", **kwargs))
        return tree

    def apply(self, func):
        return DictStructure({k: v.apply(func) for k, v in self.items()})

    def __getattr__(self, name: str):
        return {k: getattr(v, name) for k, v in self.items()}

    def __call__(self, structure, **kwargs):
        assert isinstance(structure, DictStructure), f"Expected DictStructure, got {type(structure)}: {structure}"
        assert set(self.keys()) == set(structure.keys()), f"Keys do not match: {self.keys()} vs {structure.keys()}"
        return DictStructure({k: self[k](structure[k], **kwargs) for k in self.keys()})

    def _as_native(self):
        return {k: v._as_native() for k, v in self.items()}


class LeafStructure(StructureMixin):
    def __init__(self, **content):
        for k, v in content.items():
            setattr(self, k, v)
        self._names = list(content.keys())

    def tree(self, prefix="", verbose=False, **kwargs):
        tree = Tree(f"{prefix}  📦 {', '.join(self._names)}")
        if hasattr(self, "shape"):
            tree.add(f"Shape: {self.shape}")
        if hasattr(self, "statistics"):
            tree.add(f"Statistics: {str(self.statistics)[:50]}")
        if hasattr(self, "latitudes"):
            try:
                txt = f"[{np.min(self.latitudes):.2f}, {np.max(self.latitudes):.2f}]"
            except ValueError:
                txt = "[no np.min/np.max]"
            tree.add(f"Latitudes: [{txt}")
        if hasattr(self, "longitudes"):
            try:
                txt = f"[{np.min(self.longitudes):.2f}, {np.max(self.longitudes):.2f}]"
            except ValueError:
                txt = "[no np.min/np.max]"
            tree.add(f"Longitudes: [{txt}]")
        if hasattr(self, "timedeltas"):
            try:
                minimum = np.min(self.timedeltas)
                maximum = np.max(self.timedeltas)
                minimum = int(minimum)
                maximum = int(maximum)
                minimum = datetime.timedelta(seconds=minimum)
                maximum = datetime.timedelta(seconds=maximum)
                minimum = frequency_to_string(minimum)
                maximum = frequency_to_string(maximum)
            except ValueError:
                minimum = "no np.min"
                maximum = "no np.max"
            tree.add(f"Timedeltas: [{minimum},{maximum}]")
        if hasattr(self, "data"):
            tree.add(f"Data: array of shape {self.data.shape} with mean {np.nanmean(self.data):.2f}")
        if hasattr(self, "dataspecs") and verbose:
            tree.add(f"dataspecs: {self.dataspecs}")
        return tree

    def apply(self, func):
        new = func(**{k: getattr(self, k) for k in self._names})
        name = func.__name__
        return LeafStructure(**{"dataspecs": self.dataspecs, name: new})

    def __call__(self, structure, function=None, input=None, result=None, **kwargs):
        assert isinstance(structure, LeafStructure), f"Expected LeafStructure, got {type(structure)}: {structure}"

        if not function:
            if len(self._names) != 2:
                raise NotImplementedError(f"Expected two names in LeafStructure, got {self._names}. ")
            for function in self._names:
                if function == "dataspecs":
                    continue
                break

        input_name = input or "data"
        result_name = result or input_name

        func = getattr(self, function)

        if not callable(func):
            raise ValueError(f"Expected a callable function in {self.dataspecs}, got {type(func)}: {func}")

        x = getattr(structure, input_name)
        y = func(x)

        return LeafStructure(**{"dataspecs": structure.dataspecs, result_name: y})

    def _as_native(self, key=None):
        if key is not None:
            return getattr(self, key, None)
        return {k: getattr(self, k) for k in self._names}


def structure_factory(**content):
    check_structure(**content)
    dataspecs = content["dataspecs"]

    if isinstance(dataspecs, str) and dataspecs.endswith(".tensor"):
        return LeafStructure(**content)

    if isinstance(dataspecs, (list, tuple)):
        lst = []
        for i in range(len(dataspecs)):
            lst.append(structure_factory(**{key: content[key][i] for key in content}))
        return TupleStructure(lst)

    assert isinstance(dataspecs, dict), "Expected dicts"
    dic = {}
    for k in dataspecs.keys():
        dic[k] = structure_factory(**{key: content[key][k] for key in content})
    return DictStructure(dic)


def check_structure(**content):
    assert "dataspecs" in content, "Missing 'dataspecs' in content"

    dataspecs = content["dataspecs"]
    for v in content.values():
        if isinstance(dataspecs, str) and dataspecs.endswith(".tensor"):
            continue

        if isinstance(dataspecs, dict):
            assert isinstance(
                v,
                dict,
            ), f"Expected all values to be dict, got {type(v)} != {type(dataspecs)} whith {v} and {dataspecs}"
            assert set(v.keys()) == set(
                dataspecs.keys(),
            ), f"Expected the same keys, got {list(v.keys())} vs. {list(dataspecs.keys())}"

        if isinstance(dataspecs, (list, tuple)):
            assert isinstance(
                v,
                (list, tuple),
            ), f"Expected all values to be lists or tuples, got {type(v)} != {type(dataspecs)} whith {v} and {dataspecs}"
            assert len(v) == len(dataspecs), f"Expected the same length as first, got ✅{v}✅ vs ❌{dataspecs}❌"
