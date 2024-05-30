# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

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

        self.forcing = [] if config.data.forcing is None else OmegaConf.to_container(config.data.forcing, resolve=True)
        self.diagnostic = (
            [] if config.data.diagnostic is None else OmegaConf.to_container(config.data.diagnostic, resolve=True)
        )

        assert set(self.diagnostic).isdisjoint(self.forcing), (
            f"Diagnostic and forcing variables overlap: {set(self.diagnostic).intersection(self.forcing)}. ",
            "Please drop them at a dataset-level to exclude them from the training data.",
        )
        self.name_to_index = dict(sorted(name_to_index.items(), key=operator.itemgetter(1)))
        name_to_index_model_input = {
            name: i for i, name in enumerate(key for key in self.name_to_index if key not in self.diagnostic)
        }
        name_to_index_model_output = {
            name: i for i, name in enumerate(key for key in self.name_to_index if key not in self.forcing)
        }

        self.data = DataIndex(self.diagnostic, self.forcing, self.name_to_index)
        self.model = ModelIndex(self.diagnostic, self.forcing, name_to_index_model_input, name_to_index_model_output)

    def __repr__(self) -> str:
        return f"IndexCollection(config={self.config}, name_to_index={self.name_to_index})"

    def __eq__(self, other):
        if not isinstance(other, IndexCollection):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.model == other.model and self.data == other.data

    def __getitem__(self, key):
        return getattr(self, key)

    def todict(self):
        return {
            "data": self.data.todict(),
            "model": self.model.todict(),
        }

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar(f"!{data.__class__.__name__}", repr(data))


for cls in [BaseTensorIndex, InputTensorIndex, OutputTensorIndex, BaseIndex, DataIndex, ModelIndex, IndexCollection]:
    yaml.add_representer(cls, cls.representer)
