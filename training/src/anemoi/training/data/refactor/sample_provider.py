import datetime
import warnings
from functools import cached_property

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import ListConfig
from rich import print
from rich.console import Console
from rich.tree import Tree

from anemoi.training.data.refactor.data_handler import DataHandler
from anemoi.training.data.refactor.dimension import DataDimension
from anemoi.training.data.refactor.dimension import IterableDimension
from anemoi.training.data.refactor.dimension import VariablesDimension
from anemoi.training.data.refactor.dimension import VariablesList
from anemoi.training.data.refactor.dimension import dimension_factory
from anemoi.training.data.refactor.structure import structure_factory
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta


def resolve_reference(config):
    from omegaconf import OmegaConf

    config = OmegaConf.create(config)
    config = OmegaConf.to_container(config, resolve=True)
    return config


def seconds(td):
    return round(td.total_seconds())


class Context:
    start = None
    end = None
    frequency = None
    sources = None
    _offset = None
    _all_nodes = None
    _visible_nodes = None
    _root = None

    def __init__(
        self,
        start=None,
        end=None,
        frequency=None,
        sources=None,
        offset=None,
        _parent=None,
        **kwargs,
    ):
        if "data_config" in kwargs:
            raise ValueError("The 'data_config' argument is deprecated. Use 'sources' instead.")
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")
        del kwargs

        self._parent = _parent
        self._copy_from_parent(_parent)

        if offset is None:
            offset = "0h"
        if isinstance(offset, str):
            offset = frequency_to_timedelta(offset)
        self._offset = self._offset + offset if self._offset else offset

        self.start = self.start if start is None else start
        self.end = self.end if end is None else end

        self.frequency = self.frequency if frequency is None else frequency_to_timedelta(frequency)

        self._all_nodes = self._all_nodes if self._all_nodes is not None else []
        self._visible_nodes = self._visible_nodes if self._visible_nodes is not None else []

        self.sources = self.sources if sources is None else resolve_reference(sources)
        assert self.sources is not None, f"Sources cannot be None, got {self.sources}"

        def processor_factory(config, name_to_index=None, statistics=None):
            return instantiate(
                config,
                name_to_index_training_input=name_to_index,
                statistics=statistics,
            )

        self.processor_factory = processor_factory

        assert isinstance(
            self.offset,
            datetime.timedelta,
        ), f"Expected timedelta for offset, got {type(self.offset)}: {self.offset}"

    def _copy_from_parent(self, _parent):
        if _parent is None:
            return
        if not isinstance(_parent, Context):
            raise TypeError(f"Expected Context as parent, got {type(_parent)}: {_parent}")
        self.start = _parent.start
        self.end = _parent.end
        self.frequency = _parent.frequency
        self.sources = _parent.sources
        self._offset = _parent._offset
        self._all_nodes = _parent._all_nodes
        self._visible_nodes = _parent._visible_nodes
        self._root = _parent._root

    @property
    def offset(self):
        if self._offset is None:
            return frequency_to_timedelta("0h")
        return self._offset

    def register(self, obj):
        if not isinstance(obj, SampleProvider):
            raise TypeError(f"Expected SampleProvider, got {type(obj)}: {obj}")
        assert obj not in self._all_nodes, f"Object {obj} is already registered in {self._all_nodes}"

        self._all_nodes.append(obj)

    def register_visible(self, obj):
        assert obj not in self._visible_nodes, f"Object {obj} is already registered in {self._visible_nodes}"
        self._visible_nodes.append(obj)

    def __repr__(self):
        return f"Context(start={self.start}, end={self.end}, offset={self.offset})"


class SampleProvider:
    min_offset = None
    max_offset = None

    def __init__(self, _context: Context, _parent):
        _context.register(self)
        self._context = _context
        self._parent = _parent
        self._frequency = _context.frequency
        self.offset = frequency_to_timedelta(_context.offset)

    def set_min_max_offsets(self, minimum=None, maximum=None, dropped_samples=None):
        self.min_offset = minimum
        self.max_offset = maximum
        self.dropped_samples = dropped_samples

    @property
    def is_root(self):
        return self._parent is None

    def mutate(self):
        return self

    def __len__(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    def latitudes(self, item: int):
        raise NotImplementedError

    def longitudes(self, item: int):
        raise NotImplementedError

    def timedeltas(self, item: int):
        raise NotImplementedError

    @property
    def name_to_index(self, item: int):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def statistics(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def dataspecs(self):
        return self._path(prefix="")

    def _path(self, prefix):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def extra(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def normaliser(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def imputer(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def processors(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def configs(self):
        raise ValueError("Obsolete, please use self.normaliser or self.imputer or self.extra instead")

    def get_native(self, item: int):
        dic = self[item]
        if "dataspecs" not in dic:
            dic["dataspecs"] = self.dataspecs
        return dic

    def get_obj(self, item):
        raw = self.get_native(item)
        if "dataspecs" not in raw:
            raw["dataspecs"] = self.dataspecs
        return structure_factory(**raw)

    def get_things_that_do_not_depend_on_the_index(self):
        """Returns a structure that does not depend on the index."""
        return structure_factory(
            name_to_index=self.name_to_index,
            statistics=self.statistics,
            dataspecs=self.dataspecs,
            extra=self.extra,
            normaliser=self.normaliser,
            imputer=self.imputer,
        )

    def create_structure_from_batch(self, batch):
        """Creates a structure from a batch of data."""
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict for batch, got {type(batch)}: {batch}")
        return structure_factory(
            dataspecs=self.dataspecs,
            **batch,
        )

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def frequency(self):
        return self._frequency

    def __repr__(self):
        console = Console(record=True, width=120)
        tree = self.tree()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def tree(self, label: str = None, **kwargs):
        raise NotImplementedError("Subclasses must implement tree method in " + self.__class__.__name__)

    def _check_item(self, item: int):
        if not isinstance(item, (int, np.integer)):
            raise TypeError(f"Not implemented for non-integer indexing {type(item)}")

    def shuffle(self, *args, **kwargs):
        # TODO: remove double self, self
        return ShuffledSampleProvider(self._context, self, self, *args, **kwargs)


class ForwardSampleProvider(SampleProvider):
    def __init__(self, _context, _parent, sample: SampleProvider):
        super().__init__(_context, _parent)
        self._forward = sample

    def __len__(self):
        return len(self._forward)

    @property
    def name_to_index(self):
        return self._forward.name_to_index

    @property
    def statistics(self):
        return self._forward.statistics

    def _path(self, prefix):
        return self._forward._path(prefix)

    @property
    def extra(self):
        return self._forward.extra

    @property
    def normaliser(self):
        return self._forward.normaliser

    @property
    def imputer(self):
        return self._forward.imputer

    @property
    def processors(self):
        return self._forward.processors

    @property
    def shape(self):
        return self._forward.shape

    @property
    def frequency(self):
        return self._forward.frequency


class ShuffledSampleProvider(ForwardSampleProvider):
    label = "Shuffled"
    emoji = "🎲"

    def __init__(self, _context, _parent, sample: SampleProvider, seed: int = None):
        super().__init__(_context, _parent, sample)
        self.seed = seed
        length = len(self._forward)
        self.idx = np.arange(length)
        if seed is not None:
            np.random.seed(seed)
        self.idx = np.random.permutation(self.idx)

    def __getitem__(self, item: int):
        print(f"Shuffling : requested {item}, provided {self.idx[item]}")
        return self._forward.__getitem__(self.idx[item])

    def tree(self, prefix=""):
        tree = Tree(prefix + self.emoji + self.label + f" (seed={self.seed})")
        subtree = self._forward.tree()
        tree.add(subtree)
        return tree


class DictSampleProvider(SampleProvider):
    label = "dict"
    emoji = "📖"

    def __init__(self, _context: Context, _parent, dictionary: dict):
        super().__init__(_context, _parent)

        if not isinstance(dictionary, dict):
            raise TypeError(f"Expected dictionary, got {type(dictionary)}: {dictionary}")
        if len(dictionary) == 0:
            raise ValueError("Dictionary is empty, cannot create sample provider.")
        for k in dictionary:
            if not isinstance(k, str):
                raise ValueError(f"Keys in dictionary must be strings, got {type(k)}, {k}")

        def normalise_key(k):
            new_k = "".join([x.lower() if x.isalnum() else "_" for x in k])
            if k != new_k:
                warnings.warn(f"Normalising key '{k}' to '{new_k}'")
            return new_k

        dictionary = {normalise_key(k): v for k, v in dictionary.items()}

        for k, v in dictionary.items():
            if not isinstance(v, dict):
                raise ValueError(f"Expected dictionary for sample provider, got {type(v)}: {v}. ")

        self._samples = {k: sample_provider_factory(_context, **v) for k, v in dictionary.items()}

    def __getattr__(self, key):
        if key in self._samples:
            return self._samples[key]
        raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")

    def __getitem__(self, item):
        dict_of_dict = {k: s[item] for k, s in self._samples.items()}
        first = dict_of_dict[list(dict_of_dict.keys())[0]]
        keys = list(first.keys())
        dic = {}
        for key in keys:
            dic[key] = {k: dict_of_dict[k][key] for k in self._samples.keys()}
        return dic

    @property
    def name_to_index(self):
        return {k: v.name_to_index for k, v in self._samples.items()}

    @property
    def statistics(self):
        return {k: v.statistics for k, v in self._samples.items()}

    def _path(self, prefix):
        first_dot = "" if prefix == "" else "."
        return {k: v._path(f"{prefix}{first_dot}{k}") for k, v in self._samples.items()}

    @property
    def extra(self):
        return {k: v.extra for k, v in self._samples.items()}

    @property
    def normaliser(self):
        return {k: v.normaliser for k, v in self._samples.items()}

    @property
    def imputer(self):
        return {k: v.imputer for k, v in self._samples.items()}

    @property
    def processors(self):
        return {k: v.processors for k, v in self._samples.items()}

    @property
    def shape(self):
        return {k: v.shape for k, v in self._samples.items()}

    def tree(self, prefix=""):
        tree = Tree(prefix + self.label)
        for k, v in self._samples.items():
            subtree = v.tree(prefix=f'"{k}" : ')
            tree.add(subtree)
        return tree

    def __len__(self):
        lengths = [len(s) for s in self._samples.values()]
        assert lengths, "No samples in dictionary, cannot determine length."
        assert (
            len(set(lengths)) == 1
        ), f"Samples in dictionary have different lengths: {lengths}. Cannot determine length."
        return lengths[0]


class _FilterSampleProvider(SampleProvider):
    emoji = "filter-emoji"
    label = "_Filter"

    keyword = None

    def __init__(self, _context: Context, _parent, **kwargs):
        super().__init__(_context, _parent)
        kwargs = kwargs.copy()

        assert self.keyword in kwargs, f"Keyword '{self.keyword}' not found in {kwargs}"
        self.values = kwargs.pop(self.keyword)

        new_context = Context(**{"_parent": _context, self.keyword: self.values})
        self._forward = sample_provider_factory(new_context, **kwargs)

        # shift = self._offset_as_timedelta // self._forward.frequency

    @property
    def name_to_index(self):
        return self._forward.name_to_index

    @property
    def statistics(self):
        return self._forward.statistics

    def _path(self, prefix):
        return self._forward._path(prefix)

    @property
    def extra(self):
        return self._forward.extra

    @property
    def normaliser(self):
        return self._forward.normaliser

    @property
    def imputer(self):
        return self._forward.imputer

    @property
    def processors(self):
        return self._forward.processors

    @property
    def shape(self):
        return self._forward.shape

    def __getitem__(self, item: int):
        return self._forward.__getitem__(item)

    def tree(self, prefix: str = ""):
        tree = self._forward.tree()
        tree.label = tree.label + f" ({self.emoji} {self.values})"
        return tree

    def __len__(self):
        return len(self._forward)


class OffsetSampleProvider(_FilterSampleProvider):
    emoji = "⏱️"
    label = "Offset"
    keyword = "offset"


class TupleSampleProvider(SampleProvider):
    emoji = "🔗"
    label = "tuple"

    def __init__(self, _context: Context, _parent, tuple_: dict):
        super().__init__(_context, _parent)

        tuple_ = tuple_.copy()
        loops = tuple_.pop("loop")
        template = tuple_.pop("template")
        if tuple_:
            raise ValueError(f"Unexpected keys in tuple: {tuple_}")

        self.iterables = [dimension_factory(i) for i in loops]

        for i in self.iterables:
            if not isinstance(i, IterableDimension):
                raise ValueError(f"Expected iterable dimension in all loop dimensions, got {type(i)} for {i}")

        self.template = template

    def __len__(self):
        lenghts = [len(s) for s in self._samples]
        assert len(set(lenghts)) == 1, f"Samples in tuple have different lengths: {lenghts}. Cannot determine length."
        return lenghts[0]

    def mutate(self):
        if not self.iterables:
            return sample_provider_factory(self._context, **self.template)

        if len(self.iterables) == 1:
            iterable = self.iterables[0]
            self._samples = []
            for v in iterable.values:
                config = {iterable.name: v, "structure": self.template}
                sample = sample_provider_factory(self._context, _parent=self._parent, **config)
                self._samples.append(sample)
            self._samples = tuple(self._samples)
            return self

        assert len(self.iterables) > 1
        *others, iterable = self.iterables
        new_config = {
            "tuple": {
                "loop": others,
                "template": {
                    "tuple": {
                        "loop": [iterable],
                        "template": self.template,
                    },
                },
            },
        }
        return sample_provider_factory(self._context, _parent=self._parent, **new_config)

    def __getitem__(self, item: int):
        def recurse(x):
            if isinstance(x, SampleProvider):
                return x.__getitem__(item)
            if isinstance(x, tuple):
                tuple_of_dicts = tuple(recurse(elt) for elt in x)
                first = tuple_of_dicts[0]
                keys = list(first.keys())
                dic = {}
                for key in keys:
                    # (tuple_of_dicts[key][i] for i in range(len(tuple_of_dicts))) for key in keys()}
                    _tuple = tuple(tuple_of_dicts[i][key] for i in range(len(tuple_of_dicts)))
                    dic[key] = _tuple
                return dic
            assert False, f"Unknown type {type(x)} : {x}"

        return recurse(self._samples)

    @property
    def name_to_index(self):
        return [s.name_to_index for s in self._samples]

    @property
    def statistics(self):
        return [s.statistics for s in self._samples]

    def _path(self, prefix):
        return [s._path(f"{prefix}.{i}") for i, s in enumerate(self._samples)]

    @property
    def extra(self):
        return [s.extra for s in self._samples]

    @property
    def normaliser(self):
        return [s.normaliser for s in self._samples]

    @property
    def imputer(self):
        return [s.imputer for s in self._samples]

    @property
    def processors(self):
        return [s.processors for s in self._samples]

    @property
    def shape(self):
        return [s.shape for s in self._samples]

    def tree(self, prefix=""):
        txt = prefix + self.emoji + self.label + f" ({len(self._samples)} samples)"
        tree = Tree(txt)
        for s in self._samples:
            tree.add(s.tree())
        return tree


class TensorSampleProvider(SampleProvider):
    emoji = "🔢"
    label = "tensor"

    _tuple_sample_provider = None

    def __init__(self, _context: Context, _parent, tensor: dict):
        dims = tensor
        super().__init__(_context, _parent)

        self.dimensions = [dimension_factory(dim) for dim in dims]
        if len(dims) != len(list(set([dim.name for dim in self.dimensions]))):
            raise ValueError(f"Duplicate dimension names in tuple {dims}")

        self.order = [dim.name for dim in self.dimensions]  # keep the dim order to reshape

        template = [d for d in self.dimensions if isinstance(d, DataDimension)]
        if len(template) > 1:
            raise ValueError(f"Expected a single data dimension, got {template}")
        if len(template) == 0:
            template = [d for d in self.dimensions if isinstance(d, VariablesDimension)]
        if len(template) == 0:
            raise ValueError(f"Expected a data or variables dimension, got {self.dimensions}, nothing to point to data")
        self.template = template[0]

        self._template_sample_provider = sample_provider_factory(_context, _parent=self, **self.template.raw)

        self.loops = [d for d in self.dimensions if isinstance(d, IterableDimension) and d != self.template]

        config = {
            "tuple": {
                "loop": self.loops,
                "template": self.template.raw,
            },
        }
        self._tuple_sample_provider = sample_provider_factory(_context, _parent=self, **config)

    def __len__(self):
        return len(self._tuple_sample_provider)

    def __getitem__(self, item: int):
        data = self._tuple_sample_provider.__getitem__(item)

        def process_element(k, elt):
            if k == "data":
                elt_ = np.array(elt)
                elt = self.transpose(elt_)
                return elt
            return elt

        assert "data" in data, f"Expected 'data' key in {data}, got {data}"
        return {k: process_element(k, v) for k, v in data.items()}

    @property
    def name_to_index(self):
        name_to_index = dict(variables=self._template_sample_provider.name_to_index)
        if hasattr(self._tuple_sample_provider, "iterables"):
            for dim in self._tuple_sample_provider.iterables:
                name_to_index[dim.name] = dict(enumerate(dim.values))
        return name_to_index

    @property
    def statistics(self):
        return dict(variables=self._template_sample_provider.statistics)

    def _path(self, prefix):
        return f"{prefix}.tensor"

    @property
    def extra(self):
        return self._template_sample_provider.extra

    @property
    def normaliser(self):
        return self._template_sample_provider.normaliser

    @property
    def imputer(self):
        return self._template_sample_provider.imputer

    @property
    def processors(self):
        return self._template_sample_provider.processors

    @property
    def shape(self):
        # TODO read from data handler and have 'dynamic' shape
        return self[0].shape

    def _flatten(self, x):
        if isinstance(x, dict):
            return x
        assert len(set([str(_) for _ in x])) == 1, f"Expected a single name_to_index, got {x} for {self}"
        x = x[0]
        if isinstance(x, dict):
            return x
        raise NotImplementedError(
            f"name_to_index not implemented for tensor with more than two dimensions: {self.dimensions}",
        )

    def transpose(self, array):
        # Transpose the array to match the order of requested dimensions
        # TODO : clean up this logic, maybe use ... from einops
        array = np.array(array)

        if not isinstance(array, np.ndarray):
            return array

        dimensions = self.dimensions
        order = self.order.copy()

        if len(dimensions) < array.ndim:
            # if there are less dimensions than the array has, we need to add empty dimensions
            missing_dims = array.ndim - len(dimensions)
            dimensions = dimensions + [DataDimension(name=f"dim_{i}") for i in range(missing_dims)]
            order += [f"dim_{i}" for i in range(missing_dims)]

        assert (
            len(dimensions) == array.ndim
        ), f"Expected {len(self.dimensions)} dimensions, got {array.ndim} for {array}"
        assert len(order) == len(dimensions), f"Expected {len(self.dimensions)} order, got {len(order)} for {array}"

        current_order = [
            dim.name for dim in dimensions if isinstance(dim, IterableDimension) and not dim == self.template
        ] + [self.template.name]
        if len(current_order) != len(order):
            missing_dims = len(order) - len(current_order)
            current_order += [f"dim_{i}" for i in range(missing_dims)]

        assert len(current_order) == len(order), f"Current order {current_order} does not match requested order {order}"

        import einops

        return einops.rearrange(array, " ".join(current_order) + " -> " + " ".join(order))

    def tree(self, prefix=""):
        tree = Tree(f"{prefix}{self.emoji} {self.label}")
        for i, d in enumerate(self.dimensions):
            if isinstance(d, IterableDimension):
                tree.add(f" dim {i} : {d}")
            elif isinstance(d, DataDimension):
                tree.add(self._tuple_sample_provider.tree(prefix=f"dim {i} : {d}"))
            else:
                tree.add(f'? dim {i} "{d.name}" = {d.raw}')

        if self._tuple_sample_provider is not None:
            subtree = self._tuple_sample_provider.tree(prefix="(debug) ")
            tree.add(subtree)
        return tree


class VariablesSampleProvider(SampleProvider):
    emoji = "⛁"  #  🆎
    # emoji = "🧩"
    label = "Variables"

    i_offset = None
    dropped_samples = None

    def __init__(self, _context: Context, _parent, variables: dict | list[str], data: str = None):
        super().__init__(_context, _parent)
        self.variables = VariablesList(variables, data=data)
        if len(self.variables.as_dict) > 1:
            raise ValueError(
                f"Expected a single group of variables, got {list(self.variables.as_dict.keys())} in {variables}",
            )

        dic = self.variables.as_dict
        self.group = list(dic.keys())[0]

    @property
    def name_to_index(self):
        return self.data_handler.name_to_index

    @property
    def statistics(self):
        return self.data_handler.statistics

    def _path(self, prefix):
        assert False, "Should not be called, Tensor should be a parent and handle the request"

    @property
    def extra(self):
        return self.data_handler.extra

    @property
    def normaliser(self):
        return self.data_handler.normaliser

    @property
    def imputer(self):
        return self.data_handler.imputer

    @property
    def processors(self):
        return self.data_handler.processors

    @property
    def shape(self):
        return self.data_handler.shape

    def set_min_max_offsets(self, minimum=None, maximum=None, dropped_samples=None):
        self.min_offset = minimum
        self.max_offset = maximum
        self.dropped_samples = dropped_samples
        self.actual_offset = self.offset - self.min_offset

        def _(x):
            return frequency_to_string(x) if x else "0h"

        if self.frequency != self._context.frequency:
            print(f"Warning: Frequency mismatch: {_(self.frequency)} != {_(self._context.frequency)}. ")
            print(self)
            raise NotImplementedError(
                f"Frequency mismatch: {_(self.frequency)} != {_(self._context.frequency)}. "
                "For now, the frequency must match the context frequency."
                "This will be implemented in the future if needed.",
            )

        i_offset = seconds(self.actual_offset) / seconds(self.frequency)
        if i_offset != int(i_offset):
            print("❌", self)
            msg = (
                f"Offset {_(self.offset)} or {_(self.min_offset)} is not a multiple of frequency {_(self.frequency)}, for {self}. "
                f"i_offset = {i_offset} is not an integer."
            )
            raise ValueError(msg)
        self.i_offset = int(i_offset)

    def __len__(self):
        if self.dropped_samples is None:
            return None
        return len(self.data_handler) - self.dropped_samples

    @cached_property
    def data_handler(self):
        simple_names = self.variables.as_dict[self.group]
        return DataHandler(self.group, variables=simple_names, sources=self._context.sources)

    def __getitem__(self, item):
        actual_item = item + self.i_offset

        if actual_item < 0 or actual_item >= len(self.data_handler):
            print("❌", self)
            msg = f"Item {item} ({actual_item}) is out of bounds with i_offset {self.i_offset}, lenght of the dataset is {len(self.data_handler)} and dropped_samples is {self.dropped_samples}."
            raise IndexError(msg)

        return self.data_handler._get(actual_item, request=["data", "latitudes", "longitudes", "timedeltas"])

    def tree(self, prefix: str = ""):
        def _(x):
            return frequency_to_string(x) if x else "0h"

        string_offset = _(self.offset)
        if not self.offset:
            string_offset = "0h"

        txt = f"{prefix}{self.emoji} {self.label} ("
        txt += f"{self.group}:{'/'.join(self.variables.as_list)}, offset={string_offset}"
        txt += ")"
        tree = Tree(txt)
        if self.min_offset is not None:
            tree.add(f"global_min_offset={_(self.min_offset)}")
        if self.max_offset is not None:
            tree.add(f"global_max_offset={_(self.max_offset)}")
        tree.add(f"lenght={len(self)}")
        if self.i_offset is not None:
            explain = f"({_(self.offset)} - {_(self.min_offset)})/{_(self.frequency)} = {_(self.actual_offset)}/{_(self.frequency)} = {self.i_offset}"
            tree.add(f"i -> i + {self.i_offset} because {explain}")
        return tree


def sample_provider_factory(_context=None, **kwargs):
    kwargs = kwargs.copy()

    if "context" in kwargs:
        raise NotImplementedError(
            "The 'context' argument is deprecated, use directly start, end, frequency, sources instead",
        )

    if _context is None:
        _context = Context(
            sources=kwargs.pop("sources", None),
            start=kwargs.pop("start", None),
            end=kwargs.pop("end", None),
            frequency=kwargs.pop("frequency"),
        )

    if "_parent" not in kwargs:
        kwargs["_parent"] = None
        # print(f"Building sample provider : {kwargs}")

    if "offset" in kwargs:
        obj = OffsetSampleProvider(_context, **kwargs)
    elif "dictionary" in kwargs:
        obj = DictSampleProvider(_context, **kwargs)
    elif "tensor" in kwargs:
        obj = TensorSampleProvider(_context, **kwargs)
    elif "tuple" in kwargs:
        kwargs["tuple_"] = kwargs.pop("tuple")
        obj = TupleSampleProvider(_context, **kwargs)
    elif "variables" in kwargs:
        if isinstance(kwargs["variables"], ListConfig):
            kwargs["variables"] = list(kwargs["variables"])
        obj = VariablesSampleProvider(_context, **kwargs)
    elif "repeat" in kwargs:
        repeat = kwargs.pop("repeat")
        obj = sample_provider_factory(_context, **kwargs)
    elif "structure" in kwargs:
        if isinstance(kwargs["structure"], SampleProvider):
            return kwargs["structure"]  # not mutate here?: todo: think about it
        if isinstance(kwargs["structure"], DictConfig):
            kwargs["structure"] = dict(kwargs["structure"])
        if isinstance(kwargs["structure"], dict):
            obj = sample_provider_factory(_context, **kwargs["structure"])
        else:
            raise ValueError(
                f"Expected dictionary for 'structure', got {type(kwargs['structure'])}: {kwargs['structure']}",
            )
    else:
        assert False, f"Unknown sample type for kwargs {kwargs}"
    obj = obj.mutate()

    if obj.is_root:
        # set the min/max offsets for the root sample provider
        nodes = obj._context._all_nodes
        all_offsets = [node.offset for node in nodes]
        all_offsets = [frequency_to_timedelta(o) for o in all_offsets]
        all_offsets = all_offsets + [frequency_to_timedelta("0h")]
        minimum = min(all_offsets)
        maximum = max(all_offsets)

        freq = obj._context.frequency

        freq_seconds = round(freq.total_seconds())

        def round_down(dt):
            seconds = round(dt.total_seconds())
            return datetime.timedelta(seconds=seconds - (seconds % freq_seconds))

        minimum = round_down(minimum)

        def round_up(dt):
            seconds = round(dt.total_seconds())
            remainder = seconds % freq_seconds
            if remainder == 0:
                return datetime.timedelta(seconds=seconds)
            return datetime.timedelta(seconds=seconds + (freq_seconds - remainder))

        maximum = round_up(maximum)

        assert round(minimum.total_seconds()) % round(freq.total_seconds()) == 0
        assert round(maximum.total_seconds()) % round(freq.total_seconds()) == 0
        assert minimum <= maximum, f"Minimum offset {minimum} is greater than maximum offset {maximum}"

        dropped_samples = -seconds(minimum) / seconds(freq)
        dropped_samples += seconds(maximum) / seconds(freq)
        assert int(dropped_samples) == dropped_samples
        dropped_samples = int(dropped_samples)

        for node in nodes:
            node.set_min_max_offsets(minimum=minimum, maximum=maximum, dropped_samples=dropped_samples)

    return obj
