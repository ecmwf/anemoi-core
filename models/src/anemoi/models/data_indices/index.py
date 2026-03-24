# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from dataclasses import dataclass

from anemoi.models.data_indices.ds_tensor import InputTensorIndex
from anemoi.models.data_indices.ds_tensor import OutputTensorIndex


class BaseIndex:
    """Base class for data and model indices."""

    def __init__(self):
        self.input = NotImplementedError
        self.output = NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, BaseIndex):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.input == other.input and self.output == other.output

    def __repr__(self):
        return f"{self.__class__.__name__}(input={self.input}, output={self.output})"

    def __getitem__(self, key):
        return getattr(self, key)

    def todict(self):
        return {
            "input_0": self.input[0].todict(),
            "input_1": self.input[1].todict(),
            "output": self.output.todict(),
        }

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar(f"!{data.__class__.__name__}", repr(data))


@dataclass
class DataIndex(BaseIndex):
    """Indexing for data variables."""

    _diagnostic: list
    _forcing: list
    name_to_index_input: dict
    name_to_index_output: dict

    def __post_init__(self):

        self.input = []
        for i in range(len(self.name_to_index_input)):
            if i == 0:
                diagnostic_i = self._diagnostic
            else:
                diagnostic_i = []
            self.input.append(
                InputTensorIndex(
                    includes=self._forcing,
                    excludes=diagnostic_i,
                    name_to_index=self.name_to_index_input[i],
                )
            )

        self.output = OutputTensorIndex(
            includes=self._diagnostic,
            excludes=self._forcing,
            name_to_index=self.name_to_index_output,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"diagnostic={self.input}, "
            f"forcing={self.output}, "
            f"name_to_index_input={self.name_to_index_input})"
        )


@dataclass
class ModelIndex(BaseIndex):
    """Indexing for model variables."""

    _diagnostic: list
    _forcing: list
    _name_to_index_model_input: list
    _name_to_index_model_output: dict

    def __post_init__(self):
        self.input = []
        for i in range(len(self._name_to_index_model_input)):
            self.input.append(
                InputTensorIndex(
                    includes=self._forcing,
                    excludes=[],
                    name_to_index=self._name_to_index_model_input[i],
                )
            )

        self.output = OutputTensorIndex(
            includes=self._diagnostic,
            excludes=[],
            name_to_index=self._name_to_index_model_output,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(diagnostic={self.input}, forcing={self.output}, "
            f"name_to_index_model_input={self._name_to_index_model_input}, "
            f"name_to_index_model_output={self._name_to_index_model_output})"
        )
