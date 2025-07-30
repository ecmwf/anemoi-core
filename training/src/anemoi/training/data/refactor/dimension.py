import warnings
from functools import cached_property

from omegaconf import DictConfig
from omegaconf import ListConfig


class VariablesList:
    def __init__(self, variables: list[str] | dict, data=None):
        if data is not None:
            warnings.warn(
                "Using 'data' argument is deprecated, use variables with '.' instead",
                DeprecationWarning,
            )
            if not isinstance(variables, (list, tuple)):
                raise ValueError(
                    f"Expected list or tuple for variables, got {type(variables)}: {variables}, data={data}",
                )
            self.lst = [f"{data}.{v}" for v in variables]
            return

        assert data is None, data

        if isinstance(variables, dict):
            self.lst = []
            for group, vars_ in variables.items():
                if isinstance(vars_, str):
                    vars_ = [vars_]
                if not isinstance(vars_, (list, tuple)):
                    raise ValueError(f"Expected list or tuple for variables, got {type(vars_)}: {vars_}")
                for v in vars_:
                    if not isinstance(v, str):
                        raise ValueError(f"Expected string for variable, got {type(v)}: {v}")
                    if "." in v:
                        raise ValueError(
                            f"Variable '{v}' should not contain a group name ('.' expected) in {variables})",
                        )
                self.lst += [f"{group}.{v}" for v in vars_]
            return

        if not isinstance(variables, (list, tuple)):
            raise ValueError(f"Expected list or tuple for variables, got {type(variables)}: {variables}")

        for v in variables:
            if not isinstance(v, str):
                raise ValueError(f"Expected string for variable, got {type(v)}: {v} in {variables}")
            if "." not in v:
                raise ValueError(f"Variable '{v}' does not contain a group name ('.' expected) in {variables})")
        self.lst = variables

    @property
    def as_list(self):
        return self.lst

    @cached_property
    def as_dict(self):
        dic = {}
        for v in self.lst:
            group, var = v.split(".", 1)
            if group not in dic:
                dic[group] = []
            dic[group].append(var)
        return dic

    def __repr__(self):
        return f"({', '.join(self.lst)})"


class Dimension:
    def __init__(self, **raw):
        self.raw = raw

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.raw.items())})"


class IterableDimension(Dimension):
    name = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        key = list(kwargs.keys())[0]
        assert key == self.name
        if isinstance(kwargs[key], ListConfig):
            kwargs[self.name] = list(kwargs[key])
        self.values = kwargs[key]
        self.check()

    def check(self):
        if not isinstance(self.values, (list, tuple)):
            raise ValueError(f"Not implemented for non-list values in {self.name}: {self.values}")

    def __repr__(self):
        return f"{self.name}({', '.join(map(str, self.values))})"

    def __len__(self):
        return len(self.values)


class OffsetDimension(IterableDimension):
    name = "offset"


class VariablesDimension(IterableDimension):
    name = "variables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.values, dict):
            self.values = [f"{k}.{v}" for k, vals in self.values.items() for v in vals]

    def check(self):
        pass


class RepeatDimension(IterableDimension):
    name = "repeat"


class DataDimension(Dimension):
    name = "data"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "name" in kwargs:
            self.name = kwargs["name"]

    def __len__(self):
        return "undefined length, not iterable"


class SelectionDimension(Dimension):
    # not really used, yet
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        key = list(kwargs.keys())[0]
        assert key == self.name
        self.values = kwargs[key]


class EnsembleDimension(SelectionDimension):
    name = "ensembles"


class ValuesDimension(SelectionDimension):
    name = "values"


def dimension_factory(raw_dim):
    if isinstance(raw_dim, Dimension):
        return raw_dim
    if isinstance(raw_dim, DictConfig):
        raw_dim = dict(raw_dim)
    assert isinstance(raw_dim, dict), f"Expected dict, got {type(raw_dim)}: {raw_dim}"
    if "variables" in raw_dim:
        return VariablesDimension(**raw_dim)
    if "offset" in raw_dim:
        return OffsetDimension(**raw_dim)
    if "ensembles" in raw_dim:
        warnings.warn("Ensemble dimensions are not implemented yet, ignoring the config")
        return EnsembleDimension(**raw_dim)
    if "values" in raw_dim:
        warnings.warn("Values dimensions are not implemented yet, ignoring the config")
        return ValuesDimension(**raw_dim)
    if "repeat" in raw_dim:
        warnings.warn("repeat should only be used for testing.")
        return RepeatDimension(**raw_dim)
    if "lat_lon" in raw_dim:
        raise ValueError("'lat_lon' dimension is not supported'")
    return DataDimension(**raw_dim)
