# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.models.data_indices.tensor import InputTensorIndex
from anemoi.models.data_indices.tensor import OutputTensorIndex


class BaseIndex:
    """Base class for data and model indices."""

    def __init__(self) -> None:
        self.input = NotImplementedError
        self.output = NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, BaseIndex):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.input == other.input and self.output == other.output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input={self.input}, output={self.output})"

    def __getitem__(self, key):
        return getattr(self, key)

    def todict(self):
        return {
            "input": self.input.todict(),
            "output": self.output.todict(),
        }

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar(f"!{data.__class__.__name__}", repr(data))


class DataIndex(BaseIndex):
    """Indexing for data variables."""

    def __init__(self, diagnostic, forcing, name_to_index, target=[]) -> None:
        self._diagnostic = diagnostic
        self._forcing = forcing
        self._target = target
        self._prognostic = [v for v in name_to_index.keys() if v not in set(forcing + diagnostic + target)]
        self._name_to_index = name_to_index
        self.input = InputTensorIndex(
            includes=forcing + self._prognostic,
            forcing=forcing,
            target=target,
            diagnostic=diagnostic,
            prognostic=self._prognostic,
            name_to_index=name_to_index,
        )

        self.output = OutputTensorIndex(
            includes=diagnostic + self._prognostic + target,
            forcing=forcing,
            target=target,
            diagnostic=diagnostic,
            prognostic=self._prognostic,
            name_to_index=name_to_index,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(diagnostic={self.output}, forcing={self.input}, name_to_index={self._name_to_index})"


class ModelIndex(BaseIndex):
    """Indexing for model variables."""

    def __init__(
        self,
        diagnostic,
        forcing,
        name_to_index_model_input,
        name_to_index_model_output,
        target=[],
    ) -> None:
        self._diagnostic = diagnostic
        self._forcing = forcing
        self._target = target
        self._prognostic = [v for v in name_to_index_model_input.keys() if v not in set(forcing + diagnostic + target)]
        self._name_to_index_model_input = name_to_index_model_input
        self._name_to_index_model_output = name_to_index_model_output
        self.input = InputTensorIndex(
            includes=forcing + self._prognostic,
            forcing=forcing,
            target=target,
            diagnostic=diagnostic,
            prognostic=self._prognostic,
            name_to_index=name_to_index_model_input,
        )

        self.output = OutputTensorIndex(
            includes=diagnostic + self._prognostic,
            forcing=forcing,
            target=target,
            diagnostic=diagnostic,
            prognostic=self._prognostic,
            name_to_index=name_to_index_model_output,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(diagnostic={self.input}, forcing={self.output}, "
            f"name_to_index_model_input={self._name_to_index_model_input}, "
            f"name_to_index_model_output={self._name_to_index_model_output})"
        )
