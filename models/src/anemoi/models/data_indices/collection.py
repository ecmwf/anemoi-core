# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import operator

import yaml
from omegaconf import OmegaConf

from anemoi.models.data_indices.index import BaseIndex
from anemoi.models.data_indices.index import DataIndex
from anemoi.models.data_indices.index import ModelIndex
from anemoi.models.data_indices.tensor import BaseTensorIndex
from anemoi.models.data_indices.tensor import InputTensorIndex
from anemoi.models.data_indices.tensor import OutputTensorIndex

LOGGER = logging.getLogger(__name__)


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

    def _compare_variables(self, ckpt_name_to_index: dict[str, int], data_name_to_index: dict[str, int]) -> None:
        """Compare the order of the variables in the model from checkpoint and the data.

        Parameters
        ----------
        data_name_to_index : dict[str, int]
            The dictionary mapping variable names to their indices in the data.

        Raises
        ------
        ValueError
            If the variable order in the model and data is verifiably different.
        """
        if ckpt_name_to_index is None:
            LOGGER.info("No variable order to compare. Skipping variable order check.")
            return

        if ckpt_name_to_index == data_name_to_index:
            LOGGER.info("The order of the variables in the model matches the order in the data.")
            LOGGER.debug("%s, %s", self._model_name_to_index, data_name_to_index)
            return

        keys1 = set(ckpt_name_to_index.keys())
        keys2 = set(data_name_to_index.keys())

        error_msg = ""

        # Find keys unique to each dictionary
        only_in_model = {key: ckpt_name_to_index[key] for key in (keys1 - keys2)}
        only_in_data = {key: data_name_to_index[key] for key in (keys2 - keys1)}

        # Find common keys
        common_keys = keys1 & keys2

        # Compare values for common keys
        different_values = {
            k: (ckpt_name_to_index[k], data_name_to_index[k])
            for k in common_keys
            if ckpt_name_to_index[k] != data_name_to_index[k]
        }

        LOGGER.warning(
            "The variables in the model do not match the variables in the data. "
            "If you're fine-tuning or pre-training, you may have to adjust the "
            "variable order and naming in your config.",
        )
        if only_in_model:
            LOGGER.warning("Variables only in model: %s", only_in_model)
        if only_in_data:
            LOGGER.warning("Variables only in data: %s", only_in_data)
        if set(only_in_model.values()) == set(only_in_data.values()):
            # This checks if the order is the same, but the naming is different. This is not be treated as an error.
            LOGGER.warning(
                "The variable naming is different, but the order appears to be the same. Continuing with training.",
            )
        else:
            # If the renamed variables are not in the same index locations, raise an error.
            error_msg += (
                "The variable order in the model and data is different.\n"
                "Please adjust the variable order in your config, you may need to "
                "use the 'reorder' and 'rename' key in the dataloader config.\n"
                "Refer to the Anemoi Datasets documentation for more information.\n"
            )
        if different_values:
            # If the variables are named the same but in different order, raise an error.
            error_msg += (
                f"Detected a different sort order of the same variables: {different_values}.\n"
                "Please adjust the variable order in your config, you may need to use the "
                f"'reorder' key in the dataloader config. With:\n `reorder: {ckpt_name_to_index}`\n"
            )

        if error_msg:
            LOGGER.error(error_msg)
            raise ValueError(error_msg)


for cls in [BaseTensorIndex, InputTensorIndex, OutputTensorIndex, BaseIndex, DataIndex, ModelIndex, IndexCollection]:
    yaml.add_representer(cls, cls.representer)
