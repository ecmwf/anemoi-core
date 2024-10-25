# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import operator

import yaml
from omegaconf import OmegaConf

from anemoi.models.data_indices.index import BaseIndex
from anemoi.models.data_indices.index import DataIndex
from anemoi.models.data_indices.index import ModelIndex
from anemoi.models.data_indices.tensor import BaseTensorIndex
from anemoi.models.data_indices.tensor import InputTensorIndex
from anemoi.models.data_indices.tensor import OutputTensorIndex


class IndexCollection:
    """Collection of data and model indices."""

    def __init__(self, config, name_to_index) -> None:
        self.config = OmegaConf.to_container(config, resolve=True)
        self.name_to_index = dict(sorted(name_to_index.items(), key=operator.itemgetter(1)))
        self.forcing = [] if config.data.forcing is None else OmegaConf.to_container(config.data.forcing, resolve=True)
        self.diagnostic = (
            [] if config.data.diagnostic is None else OmegaConf.to_container(config.data.diagnostic, resolve=True)
        )
        # config.data.remapped is an optional dictionary with every remapper as one entry
        self.remapped = (
            dict()
            if config.data.get("remapped") is None
            else OmegaConf.to_container(config.data.remapped, resolve=True)
        )
        self.forcing_remapped = self.forcing.copy()

        assert set(self.diagnostic).isdisjoint(self.forcing), (
            f"Diagnostic and forcing variables overlap: {set(self.diagnostic).intersection(self.forcing)}. ",
            "Please drop them at a dataset-level to exclude them from the training data.",
        )
        assert set(self.remapped).isdisjoint(self.diagnostic), (
            "Remapped variable overlap with diagnostic variables. Not implemented.",
        )
        assert set(self.remapped).issubset(self.name_to_index), (
            "Remapping a variable that does not exist in the dataset. Check for typos: ",
            f"{set(self.remapped).difference(self.name_to_index)}",
        )
        name_to_index_model_input = {
            name: i for i, name in enumerate(key for key in self.name_to_index if key not in self.diagnostic)
        }
        name_to_index_model_output = {
            name: i for i, name in enumerate(key for key in self.name_to_index if key not in self.forcing)
        }
        # remove remapped variables from internal data and model indices
        name_to_index_internal_data_input = {
            name: i for i, name in enumerate(key for key in self.name_to_index if key not in self.remapped)
        }
        name_to_index_internal_model_input = {
            name: i for i, name in enumerate(key for key in name_to_index_model_input if key not in self.remapped)
        }
        name_to_index_internal_model_output = {
            name: i for i, name in enumerate(key for key in name_to_index_model_output if key not in self.remapped)
        }
        # for all variables to be remapped we add the resulting remapped variables to the end of the tensors
        # keep track of that in the index collections
        for key in self.remapped:
            for mapped in self.remapped[key]:
                # add index of remapped variables to dictionary
                name_to_index_internal_model_input[mapped] = len(name_to_index_internal_model_input)
                name_to_index_internal_data_input[mapped] = len(name_to_index_internal_data_input)
                if key not in self.forcing:
                    # do not include forcing variables in the remapped model output
                    name_to_index_internal_model_output[mapped] = len(name_to_index_internal_model_output)
                else:
                    # add remapped forcing variables to forcing_remapped
                    self.forcing_remapped += [mapped]
            if key in self.forcing:
                # if key is in forcing we need to remove it from forcing_remapped after remapped variables have been added
                self.forcing_remapped.remove(key)

        self.data = DataIndex(self.diagnostic, self.forcing, self.name_to_index)
        self.internal_data = DataIndex(
            self.diagnostic,
            self.forcing_remapped,
            name_to_index_internal_data_input,
        )  # internal after the remapping applied to data (training)
        self.model = ModelIndex(self.diagnostic, self.forcing, name_to_index_model_input, name_to_index_model_output)
        self.internal_model = ModelIndex(
            self.diagnostic,
            self.forcing_remapped,
            name_to_index_internal_model_input,
            name_to_index_internal_model_output,
        )  # internal after the remapping applied to model (inference)

    def __repr__(self) -> str:
        return f"IndexCollection(config={self.config}, name_to_index={self.name_to_index})"

    def __eq__(self, other):
        if not isinstance(other, IndexCollection):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return (
            self.model == other.model
            and self.data == other.data
            and self.internal_model == other.internal_model
            and self.internal_data == other.internal_data
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def todict(self):
        return {
            "data": self.data.todict(),
            "model": self.model.todict(),
            "internal_model": self.internal_model.todict(),
            "internal_data": self.internal_data.todict(),
        }

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar(f"!{data.__class__.__name__}", repr(data))


for cls in [BaseTensorIndex, InputTensorIndex, OutputTensorIndex, BaseIndex, DataIndex, ModelIndex, IndexCollection]:
    yaml.add_representer(cls, cls.representer)
