# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import itertools
import os
import warnings
from abc import abstractmethod
from collections import defaultdict
from functools import cached_property

import einops
import numpy as np
import yaml
from omegaconf import DictConfig
from rich.console import Console
from rich.tree import Tree as _RichTree

from anemoi.models.data_structure.data_handler import DataHandler
from anemoi.models.data_structure.offsets import normalise_offset
from anemoi.models.data_structure.offsets import offset_to_string
from anemoi.models.data_structure.offsets import offset_to_timedelta
from anemoi.models.data_structure.offsets import substract_offsets
from anemoi.models.data_structure.offsets import sum_offsets
from anemoi.models.data_structure.path_keys import check_dictionary_key
from anemoi.models.data_structure.structure import TreeDict


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
    sharder = None
    sources = None
    _offset = None
    rollout = None
    rollout_usage = None
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
        rollout=None,
        rollout_usage=None,
        sharder=None,
        _parent=None,
    ):
        self._parent = _parent
        self._copy_from_parent(_parent)

        if offset is None:
            offset = "0h"
        if isinstance(offset, str):
            offset = offset_to_timedelta(offset)
        self._offset = self._offset + offset if self._offset else offset

        self.rollout = self.rollout if rollout is None else rollout
        self.rollout_usage = self.rollout_usage if rollout_usage is None else rollout_usage

        self.start = self.start if start is None else start
        self.end = self.end if end is None else end

        self.frequency = self.frequency if frequency is None else offset_to_timedelta(frequency)
        self.sharder = self.sharder if sharder is None else sharder

        self._all_nodes = self._all_nodes if self._all_nodes is not None else []
        self._visible_nodes = self._visible_nodes if self._visible_nodes is not None else []

        self.sources = self.sources if sources is None else resolve_reference(sources)
        assert self.sources is not None, f"Sources cannot be None, got {self.sources}"

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
        self.sharder = _parent.sharder
        self.sources = _parent.sources
        self._offset = _parent._offset
        self.rollout = _parent.rollout
        self.rollout_usage = _parent.rollout_usage
        self._all_nodes = _parent._all_nodes
        self._visible_nodes = _parent._visible_nodes
        self._root = _parent._root

    @property
    def offset(self):
        if self._offset is None:
            return offset_to_timedelta("0h")
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


class StaticMetadata(dict):
    # static metadata
    ALLOWED_KEYS = ("latitudes", "longitudes", "timedeltas", "rollout")

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            # if not k in self.ALLOWED_KEYS:
            #     raise ValueError(f"Unknown metadata key {k}, expected one of {self.ALLOWED_KEYS}")
            if not isinstance(v, TreeDict):
                raise TypeError(f"Expected TreeDict for metadata key {k}, got {type(v)}: {v}")
            # if k is None:
            #     raise ValueError("Metadata key cannot be None")
        super().__init__(**kwargs)

    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(f"No metadata key {item}, available keys: {list(self.keys())}")

    def __setattr__(self, name, value):
        if not isinstance(value, TreeDict):
            # if value is not None and not isinstance(value, TreeDict):
            raise TypeError(f"Expected TreeDict for metadata key {name}, got {type(value)}: {value}")
        self[name] = value

    def __repr__(self):
        res = "Static Metadata for a sample:\n"
        for k, v in self.items():
            res += f"  {k}: {type(v)}\n"
        return res


class SampleProvider:
    min_offset = None
    max_offset = None

    def __init__(self, _context: Context, _parent):
        _context.register(self)
        self._context = _context
        self._parent = _parent
        self._frequency = _context.frequency
        self.rollout = _context.rollout
        self.rollout_usage = _context.rollout_usage
        self.sharder = _context.sharder
        self.offset = offset_to_timedelta(_context.offset)

    # @property
    # def static_info(self):
    #     return self._get_static(None)

    # public
    def __getitem__(self, item):
        return self._get_item(None, item)

    def rollout_info(self):
        res = self._get_rollout_info()
        if res is None:
            return None
        return RolloutDict(res)

    # public
    @property
    def static_metadata(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    def _get_static(self, request):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    def _get_item(self, request, item):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    def _get_rollout_info(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

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
    def dataschema(self):
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
    def dataschema(self):
        return self._forward.dataschema

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

    def _get_rollout_info(self):
        return self._forward._get_rollout_info()


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

    def _get_item(self, request, item: int):
        print(f"Shuffling : requested {item}, provided {self.idx[item]}")
        return self._forward._get_item(request, self.idx[item])

    def tree(self, prefix=""):
        tree = _RichTree(prefix + self.emoji + self.label + f" (seed={self.seed})")
        subtree = self._forward.tree()
        tree.add(subtree)
        return tree


class _ForwardSampleProvider(SampleProvider):
    @property
    def shape(self):
        return self._forward.shape

    def _get_item(self, request, item: int):
        return self._forward._get_item(request, item)

    def _get_static(self, request):
        return self._forward._get_static(request)

    @property
    def static_metadata(self):
        return self._forward.static_metadata

    def _get_rollout_info(self):
        return self._forward._get_rollout_info()

    @property
    def dataschema(self):
        return self._forward.dataschema

    def tree(self, prefix: str = ""):
        return self._forward.tree()

    def __len__(self):
        return len(self._forward)


class _LoopSampleProvider(_ForwardSampleProvider):
    def __init__(self, _context: Context, _parent, for_each: dict):
        super().__init__(_context, _parent)

        if not isinstance(for_each, (list, tuple)) or not len(for_each) == 2:
            raise ValueError(f"Expected list of length 2 in for_each: loop, template. Got {len(for_each)}: {for_each}")

        loop_on, template = for_each
        if not isinstance(loop_on, dict) or not len(loop_on) == 1:
            raise ValueError(
                f"Expected dict with one key for the first element of for_each, got {type(loop_on)}: in {for_each}",
            )

        key, values = list(loop_on.items())[0]
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Expected list/tuple to loop on values, got {type(values)}: {values} in {for_each}")

        new_config = self.new_config(key, values, template)
        self._forward = _sample_provider_factory(_context, _parent=_parent, **new_config)
        self.values = values

    @abstractmethod
    def new_config(self, key, values, template):
        pass


class TupleLoopSampleProvider(_LoopSampleProvider):
    def new_config(self, key, values, template):
        return dict(tuple=[{key: value, **template} for value in values])


class DictionaryLoopSampleProvider(_LoopSampleProvider):
    def new_config(self, key, values, template):
        return dict(dictionary={str(value): {key: value, **template} for value in values})


class _DictSampleProvider(SampleProvider):
    label = "Dictionary"
    emoji = "📖"

    def _get_item(self, request, item):
        res = TreeDict()
        for k, sample in self._samples.items():
            res[k] = sample._get_item(request, item)
        return res

    def _get_static(self, request):
        res = TreeDict()
        for k, sample in self._samples.items():
            res[k] = sample._get_static(request)
        return res

    @property
    def static_metadata(self):
        metadatas = {k: sample.static_metadata for k, sample in self._samples.items()}
        names = set()
        for metadata in metadatas.values():
            names.update(metadata.keys())

        res = defaultdict(TreeDict)
        for dict_key, metadata in metadatas.items():
            for name in metadata.keys():
                res[name][dict_key] = metadata[name]

        return StaticMetadata(**res)

    def _get_rollout_info(self):
        raise NotImplementedError(f"Not implemented for {self.__class__.__name__}.")

    @property
    def dataschema(self):
        return dict(type="dict", content={k: v.dataschema for k, v in self._samples.items()}, _anemoi_schema=True)

    @property
    def shape(self):
        return {k: v.shape for k, v in self._samples.items()}

    def tree(self, prefix=""):
        if not hasattr(self, "_samples"):
            return _RichTree(prefix + self.label + "<partially-initialised>")
        tree = _RichTree(prefix + self.label)
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


class DictSampleProvider(_DictSampleProvider):
    def __init__(self, _context: Context, _parent, dictionary: dict):
        super().__init__(_context, _parent)

        self.check_input(dictionary)

        # dictionary = {encode_path_if_needed(k): v for k, v in dictionary.items()}
        self._samples = {k: _sample_provider_factory(_context, **v, _parent=self) for k, v in dictionary.items()}

    def check_input(self, dictionary):
        if not isinstance(dictionary, dict):
            raise TypeError(f"Expected dictionary, got {type(dictionary)}: {dictionary}")
        if len(dictionary) == 0:
            raise ValueError("Dictionary is empty, cannot create sample provider.")

        for k, v in dictionary.items():
            check_dictionary_key(k)
            if not isinstance(v, dict):
                raise ValueError(f"Expected dictionary for sample provider, got {type(v)}: {v}. ")

    def _get_rollout_info(self):
        res = TreeDict()
        for k, sample in self._samples.items():
            res[k] = sample._get_rollout_info()
        return res


# domain
#   ((step0, step1), input, (path1, path2))
#   ((step0, step1), target, path3)


# domain
#   - ((0h, 6h, 12h, 18h), input,  (-12h, -6h, 0h))
#   - ((0h, 6h, 12h, 18h), target, (6h))
# actions:
#   - (0h,  input,  (-12h, -6h, 0h)), database
#   - ('*', input,  (-12h, -6h)), previous_input
#   - ('*', input,  0h), previous_output
#   - ('*', target, '*'), database
class Rearranger:
    def __init__(self, domain, actions):
        def to_tuple(x):
            if isinstance(x, (list, tuple)):
                return tuple(x)
            return (x,)

        to_tuples = lambda x: tuple(to_tuple(_) for _ in x)

        self.actions = tuple((to_tuples(d), act) for d, act in actions)
        self.domain = tuple(to_tuples(d) for d in domain)

    def expand_actions(self, step=None, role=None, key=None, action=None):
        def action_match_element(action, element):
            step, role, key = element
            action_steps, action_roles, action_keys = action
            if not (step in action_steps or "*" in action_steps):
                return False
            if not (role in action_roles or "*" in action_roles):
                return False
            if not (key in action_keys or "*" in action_keys):
                return False
            return True

        def find_action(element):
            for action, name in self.actions:
                if action_match_element(action, element):
                    return name
            raise ValueError(f"No action found for element {element} in {self.actions}")

        for steps, roles, keys in self.domain:
            for step_, role_, key_ in itertools.product(steps, roles, keys):
                if step is not None and step_ != step:
                    continue
                if role is not None and role_ != role:
                    continue
                if key is not None and key_ != key:
                    continue
                action_ = find_action((step_, role_, key_))
                if action is not None and action_ != action:
                    continue
                yield (step_, role_, key_), action_

    def __call__(self, role=None, step=None, key=None, **kwargs):
        found = list(self.expand_actions(role=role, step=step, key=key))
        if not found:
            for available in self.expand_actions(role=role):
                print(available)
            raise ValueError(f"No action found for role={role}, step={step}, key={key}")

        res = TreeDict()
        for (step_, role_, key_), action_ in found:
            resolved = self.run_action(action_, (step_, role_, key_), **kwargs)
            if resolved is not None:
                res[key_] = resolved
        return res

    def run_action(self, action, element, **kwargs):
        return (element, action, kwargs)


# rearranger.expand_actions(action='database') -> create batch
# rearranger.expand_actions(step='0h', role='input') -> create_input
# rearranger.expand_actions(step='0h', role='target') -> create_target


class RolloutDict(TreeDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for v in self.values():
            if v is None:
                continue
            if not isinstance(v, Rearranger):
                raise ValueError(f"Expected Rollout in each leaf of a RolloutDict, got {type(v)}")

    def next_input(self, step, database=None, previous_input=None, previous_output=None):
        role = "input"
        # fill with None
        database = database.each.copy() if database else TreeDict()
        previous_input = previous_input.each.copy() if previous_input else TreeDict()
        previous_output = previous_output.each.copy() if previous_output else TreeDict()
        res = TreeDict()
        for k, rollout in self.items():
            database_ = database.get(k)
            previous_input_ = previous_input.get(k)
            previous_output_ = previous_output.get(k)
            resolved = rollout(
                role,
                step,
                database=database_,
                previous_input=previous_input_,
                previous_output=previous_output_,
            )

            res[k] = resolved
        return res

    def next_target(self, step, database=None, previous_input=None, previous_output=None):
        role = "target"
        # fill with None
        database = database.each.copy() if database else TreeDict()
        previous_input = previous_input.each.copy() if previous_input else TreeDict()
        previous_output = previous_output.each.copy() if previous_output else TreeDict()
        res = TreeDict()
        for k, rollout in self.items():
            database_ = database.get(k)
            previous_input_ = previous_input.get(k)
            previous_output_ = previous_output.get(k)
            resolved = rollout(
                role,
                step,
                database=database_,
                previous_input=previous_input_,
                previous_output=previous_output_,
            )

            res[k] = resolved
        return res


class Rollout(Rearranger):
    def normalise_and_set(self, steps, input_steps, target_steps):
        if any(offset_to_timedelta(s) > datetime.timedelta(0) for s in input_steps):
            raise ValueError(f"All input steps must be negative, got {input_steps}")
        if any(offset_to_timedelta(s) < datetime.timedelta(0) for s in target_steps):
            raise ValueError(f"All target steps must be positive, got {target_steps}")
        if any(offset_to_timedelta(s) < datetime.timedelta(0) for s in steps):
            raise ValueError(f"All rollout steps must be positive, got {steps}")
        self.steps = [normalise_offset(s) for s in steps]
        self.input_steps = [normalise_offset(s) for s in input_steps]
        self.target_steps = [normalise_offset(s) for s in target_steps]
        return self.steps, self.input_steps, self.target_steps

    def run_action(self, action, element, database=None, previous_input=None, previous_output=None):
        step, role, key = element

        if action == "database":
            key_ = sum_offsets(step, key)
            try:
                return database[key_].copy()
            except (KeyError, TypeError) as e:
                e.add_note(f"Requesting {key} for {role} at step {step} -> action {action}")
                e.add_note(f"Tried to use {key_} ({step} + {key})")
                raise e

        if action == "previous_input":
            current_step = step
            previous_step = self.steps[self.steps.index(current_step) - 1]
            delta = substract_offsets(current_step, previous_step)
            key_in_previous_input = sum_offsets(key, delta)
            try:
                return previous_input[key_in_previous_input].copy()
            except (KeyError, TypeError) as e:
                e.add_note(f"Requesting {key} for {role} at step {step} -> action {action}")
                e.add_note(f"Delta: {delta} (current_step={current_step} - previous_step={previous_step})")
                e.add_note(f"Key {key_in_previous_input} not found in previous input {previous_input.keys()}")
                raise e

        if action == "previous_output":
            current_step = step
            previous_step = self.steps[self.steps.index(current_step) - 1]
            delta = substract_offsets(current_step, previous_step)
            key_in_previous_output = sum_offsets(key, delta)
            try:
                return previous_output[key_in_previous_output].copy()
            except (KeyError, TypeError) as e:
                e.add_note(f"Requesting {key} for {role} at step {step} -> action {action}")
                e.add_note(f"Delta: {delta} (current_step={current_step} - previous_step={previous_step})")
                e.add_note(f"Key {key_in_previous_output} not found in previous output {previous_output.keys()}")
                raise e

        if action == "none":
            return None

        raise ValueError(f"Unknown action '{action}' for {element}")

    def next(self, role, step, database=None, previous_input=None, previous_output=None):
        res = TreeDict()
        for k, action in self.items():
            database_ = database.get(k) if database else None
            previous_input_ = previous_input.get(k) if previous_input else None
            previous_output_ = previous_output.get(k) if previous_output else None
            resolved = self[k](database=database_, previous_input=previous_input_, previous_output=previous_output_)
            res[k] = resolved
        return res

    def __repr__(self):
        console = Console(record=True, width=120)
        tree = self.tree()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def tree(self, prefix=""):
        tree = _RichTree(prefix + " ♻️  Rollout")
        config_tree = tree.add("Configuration")
        config_tree.add(f"Rollout steps: {self.steps}")
        config_tree.add(f"Input steps: {self.input_steps}")
        config_tree.add(f"Target steps: {self.target_steps}")
        domain_tree = tree.add("Domain")
        for steps, roles, keys in self.domain:
            for role in roles:
                domain_tree.add(f"At rollout steps {steps} for '{role}' : {keys})")
        actions_tree = tree.add("Actions")
        for (steps, roles, keys), action in self.actions:
            for role in roles:
                actions_tree.add(f"At rollout steps {steps} for '{role}' : {keys} <- {action}")
        return tree


class ForcingRollout(Rollout):
    def __init__(self, steps, input_steps, target_steps):
        steps, input_steps, target_steps = self.normalise_and_set(steps, input_steps, target_steps)

        domain = [(steps, "input", input_steps), (steps, "target", target_steps)]
        # domain = [(steps, "input", input_steps)]
        # always take input from database for forcings
        actions = []
        actions.append((("*", "input", "*"), "database"))
        actions.append((("*", "target", "*"), "none"))

        super().__init__(domain, actions)


class DiagnosticRollout(Rollout):
    def __init__(self, steps, input_steps, target_steps):
        steps, input_steps, target_steps = self.normalise_and_set(steps, input_steps, target_steps)

        domain = [(steps, "input", input_steps), (steps, "target", target_steps)]
        # domain = [(steps, "target", target_steps)]
        # always take target from database
        actions = []
        actions.append((("*", "input", "*"), "none"))
        actions.append((("*", "target", "*"), "database"))
        super().__init__(domain, actions)


class PrognosticRollout(Rollout):
    def __init__(self, steps, input_steps, target_steps):
        steps, input_steps, target_steps = self.normalise_and_set(steps, input_steps, target_steps)

        domain = [(steps, "input", input_steps), (steps, "target", target_steps)]

        actions = []
        # always take target from database
        actions.append((("*", "target", "*"), "database"))

        # for the first step, take all the input from database
        actions.append(((steps[0], "input", input_steps), "database"))
        # for other steps take it from the previous input
        actions.append((("*", "input", input_steps[:-1]), "previous_input"))
        # except for the last one, take it from the previous output
        actions.append((("*", "input", input_steps[-1]), "previous_output"))

        super().__init__(domain, actions)


class RolloutSampleProvider(_DictSampleProvider):
    emoji = "♻️"
    label = "Rollout"

    def __init__(self, _context: Context, _parent, rollout, **template):
        super().__init__(_context, _parent)

        rollout_class = dict(prognostics=PrognosticRollout, diagnostics=DiagnosticRollout, forcings=ForcingRollout)[
            rollout
        ]
        self.rollout = rollout_class(**_context.rollout)

        self._samples = {}
        # rearranger.expand_actions(action='database') -> create batch
        for (step_, role_, key_), action in self.rollout.expand_actions(action="database"):
            assert action == "database"
            offset = sum_offsets(step_, key_)
            offset = offset_to_string(offset_to_timedelta(offset))
            sample_context = Context(_parent=_context, offset=offset, rollout_usage=role_)
            self._samples[offset] = _sample_provider_factory(sample_context, **template, _parent=self)

    def _get_rollout_info(self):
        return self.rollout


class MergeOffsetSampleProvider(_LoopSampleProvider):
    emoji = "M"
    label = "MergeOffset"

    def __init__(self, _context: Context, _parent, merge_offset_as_first_dimension):
        super().__init__(_context, _parent, for_each=merge_offset_as_first_dimension)

    def new_config(self, key, values, template):
        return dict(dictionary={str(value): {key: value, **template} for value in values})

    @property
    def static_metadata(self):
        res = super().static_metadata
        res["merge_me"] = self.values
        return res


class OffsetSampleProvider(SampleProvider):
    emoji = "⏱️"
    label = "Offset"
    keyword = "offset"

    emoji = "filter-emoji"
    label = "_Filter"

    keyword = None

    def __init__(self, _context: Context, _parent, offset, **kwargs):
        super().__init__(_context, _parent)

        self.value = offset_to_string(offset_to_timedelta(offset))

        new_context = Context(_parent=_context, offset=self.value)
        self._forward = _sample_provider_factory(new_context, **kwargs, _parent=_parent)

    @property
    def shape(self):
        return self._forward.shape

    def _get_item(self, request, item: int):
        return self._forward._get_item(request, item)

    def _get_static(self, request):
        warnings.warn("TODO: change the metadata for data after offset?")
        return self._forward._get_static(request)

    @property
    def static_metadata(self):
        return self._forward.static_metadata

    def _get_rollout_info(self):
        warnings.warn("TODO: change the metadata for data after offset?")
        return self._forward._get_rollout_info()

    @property
    def dataschema(self):
        return self._forward.dataschema

    def tree(self, prefix: str = ""):
        tree = self._forward.tree()
        tree.label = tree.label + f" ({self.emoji} {self.value})"
        return tree

    def __len__(self):
        return len(self._forward)


class TensorReshapeSampleProvider(ForwardSampleProvider):
    emoji = "🔄"
    label = "Reshape"

    def __init__(self, _context: Context, _parent, reshape):
        reshape = reshape.copy()
        dimensions = reshape.pop("dimensions")
        if not isinstance(dimensions, (list, tuple)):
            raise ValueError(f"Expected list/tuple for dimensions, got {type(dimensions)}: {dimensions}")
        if not all(isinstance(d, str) for d in dimensions):
            raise ValueError(f"Expected list/tuple of strings for dimensions, got {dimensions}")
        dimensions = tuple(dimensions)

        if "batch" in dimensions:
            raise ValueError("Cannot reshape on 'batch' dimension, it is implicit and always first.")

        sample = _sample_provider_factory(_context, _parent=self, **reshape)

        assert len(sample.static_metadata.batch) == 1, "Expected only one leaf in the batch metadata"
        self.initial_order = sample.static_metadata.batch.first["dimensions_order"]

        self.new_order = ("batch", *dimensions)

        for d in self.new_order:
            if d not in self.initial_order:
                raise ValueError(f"Dimension '{d}' not found in dataset for {sample}")

        initial_dims = [d if d in self.new_order else "1" for d in self.initial_order]
        to_dims = [_ for _ in self.new_order]

        assert initial_dims[0] == "batch", initial_dims
        assert to_dims[0] == "batch", to_dims
        initial_dims = initial_dims[1:]
        to_dims = to_dims[1:]

        self.einops_rearrange_str = f"{' '.join(initial_dims)} -> {' '.join(to_dims)}"

        super().__init__(_context, _parent, sample)

    def _update_box(self, box):
        box = box.copy()

        if "_initial_dimensions_order" in box:
            # this shouldn't be a problem to implement, but we don't need it yet
            # And we want to avoid silent bugs, where the previous '_initial_dimensions_order' is lost
            raise NotImplementedError(f"Reshaping twice is not implemented, got {box}")
        if "dimensions_order" in box:
            box["_initial_dimensions_order"] = self.initial_order
            box["dimensions_order"] = self.new_order
        if "shape" in box:
            raise NotImplementedError("TODO: implement shape update")

        if "data" in box:
            try:
                box["data"] = einops.rearrange(box["data"], self.einops_rearrange_str)
            except Exception as e:
                e.add_note(f"{e} while rearranging {(box['data'].shape)} from {self.einops_rearrange_str}")
                e.add_note(f"{self}")
                raise e
        return box

    def _get_static(self, request):
        box = self._forward._get_static(request)
        return self._update_box(box)

    @property
    def static_metadata(self):
        res = StaticMetadata()
        for k, v in self._forward.static_metadata.items():
            if k != "batch":
                res[k] = v
                continue
            assert len(v) == 1, "Expected only one leaf in the batch metadata"
            assert set(v.keys()) == {""}, f"Expected only one key '' in the batch metadata, got {v.keys()}"
            box = v[""]
            box = self._update_box(box)
            res["batch"] = TreeDict({"": box})
        return res

    def _get_item(self, request, item):
        box = self._forward._get_item(request, item)
        return self._update_box(box)

    @property
    def dataschema(self):
        schema = self._forward.dataschema
        schema["dimensions_order"] = self.new_order
        return schema

    def shape(self):
        raise NotImplementedError("Dead code here, remove all 'shape' methods")

    def tree(self, prefix=""):
        if not hasattr(self, "new_order"):
            return _RichTree(f"{prefix}{self.emoji} {self.label}")

        tree = _RichTree(f"{prefix}{self.emoji} {self.label} ({self.new_order})")
        tree.add(self._forward.tree(prefix=" "))

        return tree


class BoxSampleProvider(SampleProvider):
    emoji = "📦"
    label = "Data"

    i_offset = None
    dropped_samples = None

    _mutate = None

    def __init__(self, _context: Context, _parent, container):
        box = container
        box = box.copy()
        super().__init__(_context, _parent)
        if "dimensions" in box:
            # box.pop('dimensions')
            dimensions = box.pop("dimensions")
            self._mutate = dict(reshape=dict(dimensions=dimensions, container=box))
            return

        if "data_group" not in box:
            raise ValueError(f"Expected 'data_group' in box, got {box}")
        self.data_group = box.pop("data_group")
        self.variables = box.pop("variables", None)

    def mutate(self):
        if self._mutate:
            return _sample_provider_factory(self._context, _parent=self._parent, **self._mutate)
        return self

    def _get_static(self, request):
        assert request is None, f"Expected None for request, got {request}"
        request = dict(sharder=self.sharder)
        res = self.datahandler._get_static(request)
        res["_rollout"] = self.rollout
        res["rollout_usage"] = self.rollout_usage
        if self.offset is not None:
            res["_offset"] = offset_to_string(self.offset)
        return res

    @property
    def static_metadata(self):
        metadata = TreeDict({"": self._get_static(None)})
        return StaticMetadata(batch=metadata)

    def _get_rollout_info(self):
        return None

    def _get_item(self, request, item):
        actual_item = item + self.i_offset

        if actual_item < 0 or actual_item >= len(self.datahandler):
            msg = f"Item {item} ({actual_item}) is out of bounds with i_offset {self.i_offset}, lenght of the dataset is {len(self.datahandler)} and dropped_samples is {self.dropped_samples}."
            raise IndexError(msg)

        assert request is None, f"Expected None for request, got {request}"
        request = dict(sharder=self.sharder)
        return dict(self.datahandler._get_item(request, actual_item))

    @property
    def dataschema(self):
        return self.datahandler._dataschema

    @cached_property
    def datahandler(self):
        return DataHandler(
            self.data_group,
            variables=self.variables,
            sources=self._context.sources,
        )

    def tree(self, prefix: str = ""):
        def _(x):
            return offset_to_string(x)

        txt = f"{prefix}{self.emoji} {self.label} "
        txt += f"group={self.data_group} "
        txt += f"variables:{','.join(self.variables)}"
        txt += f" from {self.datahandler.dataset_name}"
        tree = _RichTree(txt)

        if os.environ.get("ANEMOI_CONFIG_VERBOSE_STRUCTURE"):
            if self.min_offset is not None:
                tree.add(f"global_min_offset={_(self.min_offset)}")
            if self.max_offset is not None:
                tree.add(f"global_max_offset={_(self.max_offset)}")
            tree.add(f"length={self.__len__()}")
            if self.i_offset is not None:
                explain = f"({_(self.offset)} - {_(self.min_offset)})/{_(self.frequency)} = {_(self.actual_offset)}/{_(self.frequency)} = {self.i_offset}"
                tree.add(f"i -> i + {self.i_offset} because {explain}")
        return tree

    def set_min_max_offsets(self, minimum=None, maximum=None, dropped_samples=None):
        self.min_offset = minimum
        self.max_offset = maximum
        self.dropped_samples = dropped_samples
        self.actual_offset = self.offset - self.min_offset

        if self.frequency != self._context.frequency:
            warnings.warn(
                f"Warning: Frequency mismatch: {offset_to_string(self.frequency)} != {offset_to_string(self._context.frequency)}. ",
            )
            warnings.warn(self)
            raise NotImplementedError(
                f"Frequency mismatch: {offset_to_string(self.frequency)} != {offset_to_string(self._context.frequency)}. "
                "For now, the frequency must match the context frequency."
                "This will be implemented in the future if needed.",
            )

        i_offset = seconds(self.actual_offset) / seconds(self.frequency)
        if i_offset != int(i_offset):
            warnings.warn(f"❌ {self}")
            msg = (
                f"Offset {offset_to_string(self.offset)} or {offset_to_string(self.min_offset)} is not a multiple of frequency {offset_to_string(self.frequency)}, for {self}. "
                f"i_offset = {i_offset} is not an integer."
            )
            raise ValueError(msg)
        self.i_offset = int(i_offset)

    def __len__(self):
        if self.dropped_samples is None:
            return None
        return len(self.datahandler) - self.dropped_samples


global SAVE_COUNTER
SAVE_COUNTER = 0


def SAVE(kwargs):
    global SAVE_COUNTER
    SAVE_COUNTER += 1
    import yaml

    with open(f"sample_provider_{SAVE_COUNTER}.yaml", "w") as f:
        f.write(yaml.dump(kwargs))


def _remove_omega_conf(x):
    assert not isinstance(x, SampleProvider), type(x)

    try:
        from omegaconf import DictConfig
        from omegaconf import ListConfig
        from omegaconf import OmegaConf

        if isinstance(x, DictConfig) or isinstance(x, ListConfig):
            warnings.warn("Received Omegaconf input for sample_provider_factory, converting to standard python types")
            x = OmegaConf.to_container(x, resolve=True)
        if isinstance(x, dict):
            x = {k: _remove_omega_conf(v) for k, v in x.items()}
        if isinstance(x, list):
            x = [_remove_omega_conf(v) for v in x]
        if isinstance(x, tuple):
            x = tuple(_remove_omega_conf(v) for v in x)
    except ImportError:
        pass

    return x


def sample_provider_factory(**kwargs):
    kwargs = _remove_omega_conf(kwargs)
    SAVE(kwargs)

    _context = Context(
        sources=kwargs.pop("sources", None),
        start=kwargs.pop("start", None),
        end=kwargs.pop("end", None),
        frequency=kwargs.pop("frequency"),
        sharder=kwargs.pop("sharder", None),
        rollout=kwargs.pop("rollout", None),
    )
    return _sample_provider_factory(**kwargs, _parent=None, _context=_context)


def _sample_provider_factory(_context=None, **kwargs):
    initial_kwargs = kwargs
    kwargs = kwargs.copy()

    assert _context is not None
    assert "_parent" in kwargs, kwargs

    if "offset" in kwargs:
        obj = OffsetSampleProvider(_context, **kwargs)
    elif "reshape" in kwargs:
        obj = TensorReshapeSampleProvider(_context, **kwargs)
    elif "dictionary" in kwargs:
        obj = DictSampleProvider(_context, **kwargs)
    elif "for_each" in kwargs:
        obj = DictionaryLoopSampleProvider(_context, **kwargs)
    elif "merge_offset_as_first_dimension" in kwargs:
        obj = MergeOffsetSampleProvider(_context, **kwargs)
    elif "rollout" in kwargs:
        obj = RolloutSampleProvider(_context, **kwargs)
    # elif "for_each_as_tuple" in kwargs:
    #     obj = TupleLoopSampleProvider(_context, **kwargs)
    elif "container" in kwargs:
        obj = BoxSampleProvider(_context, **kwargs)
    # elif "repeat" in kwargs:
    #     repeat = kwargs.pop("repeat")
    #     obj = _sample_provider_factory(_context, **kwargs)
    elif "structure" in kwargs:
        if isinstance(kwargs["structure"], SampleProvider):
            return kwargs["structure"]  # not mutate here?: todo: think about it
        if isinstance(kwargs["structure"], DictConfig):
            kwargs["structure"] = dict(kwargs["structure"])
        if isinstance(kwargs["structure"], dict):
            obj = _sample_provider_factory(_context, **kwargs["structure"])
        else:
            raise ValueError(
                f"Expected dictionary for 'structure', got {type(kwargs['structure'])}: {kwargs['structure']}",
            )
    else:
        assert False, f"Unknown sample type for kwargs {kwargs.keys()}"
    obj_ = None
    while obj != obj_:
        obj_ = obj
        obj = obj.mutate()
    del obj_

    if obj.is_root:
        # set the min/max offsets for the root sample provider
        nodes = obj._context._all_nodes
        all_offsets = [node.offset for node in nodes]
        all_offsets = [offset_to_timedelta(o) for o in all_offsets]
        all_offsets = all_offsets + [offset_to_timedelta("0h")]
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

    # print('✅ created obj : ' , obj)
    return obj


def test_custom(path):
    path = yaml.safe_load(path)
    with open(path) as f:
        yaml_str = f.read()
    CONFIG = yaml.safe_load(yaml_str)
    sample_config = CONFIG["sample"]
    sources_config = CONFIG["data"]

    do_something_with_this


def test_one(training_context):

    cfg_1 = """dictionary:
                    lowres:
                        container:
                          data_group: "era5"
                          variables: ["2t", "t_500", "tp"]
                          dimensions: ["values", "variables", "ensembles"]

                    highres:
                      merge_offset_as_first_dimension:
                        - offset: ["-6h", "+0h", "+6h", "+12h", "+18h", "+24h"]
                        - container:
                            data_group: "era5"
                            variables: ["2t", "t_500", "tp"]
                            dimensions: ["variables", "values"]
            """

    print("✅✅✅✅✅✅✅✅✅✅✅✅✅✅")
    config = yaml.safe_load(cfg_1)

    sp = sample_provider_factory(**training_context, **config)
    print(sp)
    schema = sp.dataschema

    # print(schema)
    # print(repr_schema(schema, "schema"))

    print("✅ sp.static_metadata.batch", sp.static_metadata.batch)

    batch_data = sp[1]
    batch_data = add_batch_dim_as_dataloder_would_do(batch_data)

    for k, v in batch_data.items():
        print(f" - {k}: {type(v)}")
    print("✅ Batch Data", batch_data)

    data = sp.static_metadata.batch + batch_data
    print("✅ Data full", data)

    assert isinstance(data, TreeDict), type(data)
    assert "data" in data["lowres"], data["lowres"].keys()
    assert "name_to_index" in data["lowres"], data["lowres"].keys()

    extra = data.each.pop("extra")
    print("Extra after pop extra", extra)

    # lat,lon = data.select_content("latitudes", "longitudes")
    lat = data.each["latitudes"]
    lon = data.each["longitudes"]
    print("Latitudes", lat)
    print("Longitudes", lon)

    def build_normaliser(statistics, **kwargs):
        mean = np.mean(statistics["mean"])

        def func(box):
            box = box.copy()
            box["data"] = box["data"] - mean
            return box

        return func

    normaliser = sp.static_metadata.batch.map_expanded(build_normaliser)

    print("Normaliser function", normaliser)
    n_data = normaliser.each(data)

    d = data.each["data"]
    n = n_data.each["data"]
    print_columns(f"Unnormalised data {d}", f"Normalised data {n}")


from rich.table import Table


def print_columns(*args):

    console = Console()
    table = Table(show_header=False, box=None)
    for a in args:
        table.add_column()
    table.add_row(*args)
    console.print(table)


def add_batch_dim_as_dataloder_would_do(data):
    res = TreeDict()
    for k, v in data.items():
        v = v.copy()
        v["data"] = np.expand_dims(v["data"], axis=0)
        res[k] = v
    return res


def test_two(training_context):
    cfg_2 = """dictionary:
                state:
                   dictionary:
                     prognostics:
                       rollout: prognostics
                       container:
                           dimensions: ["values", "variables"]
                           variables: ["2t", "t_500", "tp"]
                           data_group: "era5"

                     forcings:
                       rollout: forcings
                       container:
                           dimensions: ["values", "variables"]
                           variables: ["2t", "t_500", "tp"]
                           data_group: "era5"

                     diagnostics:
                       rollout: diagnostics
                       container:
                           dimensions: ["values", "variables"]
                           variables: ["2t", "t_500", "tp"]
                           data_group: "era5"

                  #observations:
                  #  tuple:
                  #    for_each:
                  #      - offset: ["-6h", "0h", "+6h"]
                  #    template:
                  #      tensor:
                  #        - variables: ["metop_a.scatss_1", "metop_a.scatss_2"]
            """

    rollout_config = yaml.safe_load(
        """
        rollout:
          steps: ["0h", "+6h", "+12h", "+18h", "+24h"]
          input_steps: ["-12h", "-6h", "0h"]
          target_steps: ["+6h", "12h"]
        """,
    )
    print("Rollout config", rollout_config)

    config = yaml.safe_load(cfg_2)
    sp = sample_provider_factory(**training_context, **config, **rollout_config)
    print("static_metadata", sp.static_metadata.batch)

    data = sp[1]
    data = add_batch_dim_as_dataloder_would_do(data)

    print(data)
    print("✅")
    print(sp.static_metadata.batch)

    data = sp.static_metadata.batch + data
    assert len(data) > 0, data
    print("Merged data", data)
    data = data.select_content(["_offset", "data", "_reference_date_str", "_offset"])
    # data = data.select_content(["_offset", "rollout_usage", "data"])
    print("Selected Data", data)
    assert len(data) > 0, data
    print("Rollout info : ", type(sp.rollout_info()), sp.rollout_info())
    rollout_steps = None
    for path, rollout in sp.rollout_info().items():
        rollout_steps = rollout.steps
        break
    print("Rollout steps", rollout_steps)

    rollout = sp.rollout_info()
    print("Rollout info", rollout)
    print(type(rollout.first))

    print("*******************")
    print("Rollout preparation")
    print("*******************")

    def age(v):
        return f"(age={v['_offset']})"

    for k, v in data.items():
        v["_tag"] = f"🟢 from batch {age(v)}"
    input = None
    output = None
    for i, step in enumerate(rollout_steps):
        input = rollout.next_input(step=step, database=data, previous_input=input, previous_output=output)
        target = rollout.next_target(step=step, database=data)
        # run model
        output = target
        print(f"-------- Rollout {i} {step=} --------")
        print_columns(
            input.each["_tag"].to_str(f"Input at step {i} : "),
            target.each["_tag"].to_str(f"Target at step {i} : "),
        )

        for k, v in input.items():
            v["_tag"] = f"🔵 prev_input {age(v)}"
        for k, v in output.items():
            v["_tag"] = f"🟠 prev_output {age(v)}"

    print()
    print("***************")
    print("Rollout applied")
    print("***************")

    for k, v in data.items():
        v["_tag"] = f"🟢 from batch {v['_offset']}"
    input = None
    output = None
    for i, step in enumerate(rollout_steps):
        input = rollout.next_input(step=step, database=data, previous_input=input, previous_output=output)
        target = rollout.next_target(step=step, database=data)
        # run model
        output = target
        print(f"-------- Rollout {i} {step=} --------")
        print_columns(
            input.unwrap("data").to_str(f"Input at step {i} : "),
            target.unwrap("data").to_str(f"Target at step {i} : "),
        )
        for k, v in input.items():
            v["_tag"] = f"🔵 from previous_input {v['_offset']}"
        for k, v in output.items():
            v["_tag"] = f"🟠 from previous_output {v['_offset']}"


def test_three(training_context):
    cfg_3 = """dictionary:
                  fields:
                    merge_offset_as_first_dimension:
                        - offset: ["-6h", "+0h"]
                        - container:
                            data_group: "era5"
                            variables: ["2t", "t_500", "tp"]
                            dimensions: ["variables", "values"]
                    #for_each:
                    #  - offset: ["-6h", "0h"]
                    #  - container:
                    #      data_group: "era5"
                    #      variables: ["2t", "t_500", "tp"]
            """
    config = yaml.safe_load(cfg_3)
    sp = sample_provider_factory(**training_context, **config)
    print(sp)
    print(f"Static metadata has {len(sp.static_metadata)} items: {list(sp.static_metadata.keys())}:")
    for k, v in sp.static_metadata.items():
        print(f" ✅  {k}: {v}")
    print("--------------")


def test():

    import sys

    if len(sys.argv) > 1 and not sys.argv[1].isdigit():
        return test_custom(sys.argv[1])

    source_yaml = """sources:
                         training:
                           era5:
                             dataset:
                               dataset: aifs-ea-an-oper-0001-mars-20p0-2016-2016-6h-v1
                               # dataset: aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8
                               set_group: era5
                           snow:
                             dataset: observations-testing-2018-2018-6h-v0
                           metop_a:
                             dataset: observations-testing-2018-2018-6h-v0
                             normaliser:
                                   "scatss_1": "mean-std"
                                   "scatss_2": "min-max"
                                   "scatss_3": {"name": "custom-normaliser", "theta": 0.5, "rho": 0.1}
                             imputer:
                                   "scatss_1": special
                                   "scatss_2": other
                                   "scatss_3": {"name": "custom-imputer", "theta": 0.5, "rho": 0.1}
                             extra:
                               user_key_1: a
                               user_key_2:
                                   1: foo
                                   2: bar
    """
    sources_config = yaml.safe_load(source_yaml)["sources"]["training"]
    print(sources_config)

    training_context = dict(sources=sources_config, start=None, end=None, frequency="6h")

    ONE = "1" in sys.argv
    TWO = "2" in sys.argv
    THREE = "3" in sys.argv
    if ONE:
        print("✅-✅")
        test_one(training_context)

    if TWO:
        print("✅--✅")
        test_two(training_context)

    if THREE:
        print("✅---✅")
        test_three(training_context)


if __name__ == "__main__":

    test()
