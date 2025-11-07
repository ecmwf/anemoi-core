# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import einops
from rich.tree import Tree as _RichTree

from anemoi.models.data_structure.dicts import DynamicDataDict
from anemoi.models.data_structure.dicts import StaticDataDict
from anemoi.models.data_structure.offsets import find_required_steps_for_rollout
from anemoi.models.data_structure.offsets import offset_to_np_timedelta
from anemoi.models.data_structure.sample_provider.base import Context
from anemoi.models.data_structure.sample_provider.base import Forward
from anemoi.models.data_structure.sample_provider.base import InsertStatic
from anemoi.models.data_structure.sample_provider.base import SampleProvider
from anemoi.models.data_structure.sample_provider.base import _sample_provider_factory

LOGGER = logging.getLogger(__name__)


def _merge_sublists(d):
    # merge a dict of lists into a single list
    res = []
    for v in d.values():
        if not isinstance(v, list):
            raise ValueError(f"Expected list for offsets, got {type(v)}: {v}")
        res += v
    return res


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

    def _getitem(self, data):
        res = DynamicDataDict()
        for k, v in self._forward._getitem(data).items():
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


class Container(SampleProvider):
    add_to_i = None
    multiply_i = None

    def __init__(self, _context: Context, container: dict):
        self.container = container
        self.variables = container["variables"]
        self.data_group = container["data_group"]
        self.extra_configs = container.get("extra_configs", {})
        self.dh = _context["data_handler"]
        self._offset = _context.get("offset", "0h")

    @property
    def _dates_block_in_dataset(self):
        return self.dh.dates_block(self.data_group)

    def filter_available_dates(self, dates_block):
        minus_offset = self._dates_block_in_dataset - self._offset
        return dates_block & minus_offset

    def update_index_offsets(self: "SampleProvider", dates_block):
        # Update the object's index offset from the overall dates_block and its own offset
        #
        #  Using upper case for sample dates and lower case for dataset dates:
        #
        #  (1) D = S + i . F         (D sample date, S sample start date, i sample index, F sample frequency)
        #  (2) d = s + j . f         (d dataset date, s dataset start date, j dataset index, f dataset frequency)
        #  (3) d = D + offset        required data is at date d (different from sample date D)
        #
        #  => j . f = d - s                         (2)
        #  => j . f = (D + offset) - s              (substituting d from (3))
        #  => j . f = (S + i . F + offset) - s      (substituting D from (1))
        #  => j . f = (S - s + offset) + i . F
        #  => j = (S - s + offset) / f + i . (F / f)
        #  => j = add_to_i + i . multiply_i
        #
        # add_to_i = (S - s + offset) / f
        # multiply_i = F / f

        in_dataset = self._dates_block_in_dataset
        offset = offset_to_np_timedelta(self._offset)

        add_to_i = (dates_block.start - in_dataset.start + offset) / in_dataset.frequency
        multiply_i = dates_block.frequency / in_dataset.frequency

        assert int(add_to_i) == add_to_i, add_to_i
        assert int(multiply_i) == multiply_i, multiply_i
        add_to_i = int(add_to_i)
        multiply_i = int(multiply_i)

        LOGGER.debug(f"Setting dynamic request for container ({self.variables}, {self._offset}):")
        LOGGER.debug(f"   offset  : {offset}")
        LOGGER.debug(f"   sample dates: {dates_block}")
        LOGGER.debug(f"   in dataset  : {in_dataset}")
        LOGGER.debug(f"   start difference (S - s + offset): {dates_block.start} - {in_dataset.start} + {offset}")
        LOGGER.debug(f"                                    : {dates_block.start - in_dataset.start + offset}")
        LOGGER.debug(f"   offset / f: {offset} / {in_dataset.frequency} = {offset / in_dataset.frequency}")
        LOGGER.debug(f"   => add_to_i: {add_to_i}, multiply_i: {multiply_i}")

        # update the dataset dates_block to the overall one
        # and provide the index offset and multiplier factor
        self.add_to_i = add_to_i
        self.multiply_i = multiply_i

    def register_read_pattern(self):
        self.dh.register_read_pattern(
            self,
            self.data_group,
            variables=self.variables,
            add_to_i=self.add_to_i,
            multiply_i=self.multiply_i,
        )

    @property
    def static(self):
        self._static = self.dh.static(self.data_group, variables=self.variables).copy()
        extra_configs = {**self._static.pop("extra_configs", {}), **self.extra_configs}
        if extra_configs:
            self._static["extra_configs"] = extra_configs
        return self._static

    def _getitem(self, data):
        return self.dh.get_item(self, data)

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
                return InsertStatic.new(_context, forward=forward, static=dict(variables_categories=categories))

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
                return InsertStatic.new(_context, forward=forward, static=dict(offsets_categories=categories))

            case {"dimensions": list() as dims} if all(isinstance(d, str) for d in dims):
                # found "dimensions" key with simple format (a list of string)
                # rearrange the tensors accordingly to match these dimensions
                dimensions = container.pop("dimensions")
                forward = _sample_provider_factory(_context, container)
                return Rearrange.new(_context, forward=forward, dimensions=dimensions)

            case {"dimensions": [["offsets"], *dimensions]}:
                from anemoi.models.data_structure.sample_provider.stacks import StackAsLists

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
                from anemoi.models.data_structure.sample_provider.stacks import Stack

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

    def _tree(self, prefix=None):
        name = ""  # self.__class__.__name__
        if prefix:
            name = f"{prefix}: {name}"
        tree = _RichTree(name)
        for k, v in self.container.items():
            tree.add(f"{k}: {v}")
        tree.add(f"j = i * {self.multiply_i} + {self.add_to_i} (offset: {self._offset})")
        return tree
